"""A safe core for evaluating expressions a language model writes. It
never calls eval(). New statement types get added by extending
eval_node/exec_stmt -- never by relaxing the whitelist check below.

The one rule that has to hold no matter how much this grows: parsing
uses only ast.parse, dispatch goes only through an explicit whitelist
of node types and names, and eval() is never called on anything a
model produced.

For the reasoning behind each design choice below -- what was tried
first, what broke, what was verified live before being trusted -- see
notes/DAILY_SUMMARY.md. This docstring states the current rules, not
their history.

Grammar (informal EBNF):

    program    := statement*
    statement  := assign | if_stmt | while_stmt | for_stmt | func_def | expr_stmt
    func_def   := "def" NAME "(" (NAME ("," NAME)*)? ")" ":" NEWLINE INDENT statement+ DEDENT
    assign     := NAME "=" expr
    if_stmt    := "if" expr ":" NEWLINE INDENT statement+ DEDENT
                  ("else" ":" NEWLINE INDENT statement+ DEDENT)?
    while_stmt := "while" expr ":" NEWLINE INDENT statement+ DEDENT
    for_stmt   := "for" NAME "in" expr ":" NEWLINE INDENT statement+ DEDENT
    expr_stmt  := expr
    expr       := call | method_call | compare | arith | boolean | unary
                  | subscript | attribute | list_expr | dict_expr | ternary
                  | NAME | literal
    call       := NAME "(" (expr ("," expr)*)? ")"
    method_call := expr "." NAME "(" (expr ("," expr)*)? ")"
    compare    := expr (("==" | "!=" | "<" | "<=" | ">" | ">=" | "in" | "not in") expr)+
    arith      := expr ("+" | "-" | "*" | "/" | "//" | "%" | "**") expr
    boolean    := expr ("and" | "or") expr
    unary      := ("-" | "+" | "not") expr
    subscript  := expr "[" expr "]"
    attribute  := expr "." NAME
    list_expr  := "[" (expr ("," expr)*)? "]"
    dict_expr  := "{" (expr ":" expr ("," expr ":" expr)*)? "}"
    ternary    := expr "if" expr "else" expr
    literal    := NUMBER | STRING | "True" | "False" | "None"

Note: method_call is only valid when expr evaluates to a string, and
only for the fixed whitelist in _STRING_METHODS -- a runtime check,
not grammar-enforced (same as "list index must be an integer").

Two separate namespaces: `allowed` (whitelisted callables, checked at
call sites) and `env` (variables bound by assignment, checked at
name-read sites). A bare reference to a whitelisted function name is
not a variable read, and is rejected.

**Capability tracking.** Every value carries two independent tags,
Trust (UNTRUSTED/TRUSTED) and Secrecy (SECRET/PUBLIC) -- independent
because a value can be trustworthy but secret (an API key) or
untrustworthy but not sensitive (an attacker's email body). Both are
threaded through eval_node/exec_stmt directly, no side table.

- `sources`: return value always UNTRUSTED, regardless of arguments.
- `privileged`: refuses to run (raises CapabilityError) if any
  argument is UNTRUSTED, or if pc_trust is UNTRUSTED (see below).
- `sanitizers`: return value always TRUSTED, regardless of arguments
  -- the only way to turn UNTRUSTED back into TRUSTED. `sources` wins
  if a name is listed in both.
- `confidential`/`sinks`/`declassifiers`: the same three rules,
  mirrored for Secrecy (ConfidentialityError instead of
  CapabilityError). `confidential` wins over `declassifiers`.
- An ordinary function (none of the above) propagates trust/secrecy
  from its arguments: UNTRUSTED/SECRET in, UNTRUSTED/SECRET out,
  through any chain of calls.

The six sets are bundled into one `_Capabilities` object rather than
six separate parameters, so a positional-argument mistake at a
recursive call site can't silently drop one check with no error.

**Loop bounds.** MAX_WHILE_ITERATIONS caps one loop. A shared
`_IterationBudget`, created once per run() call and threaded through
every exec_stmt call, caps total iterations across the whole program
-- so nested loops can't each individually stay under the per-loop cap
while multiplying past a sane total.

**Containers.** Lists/dicts carry trust and secrecy per element, not
one tag for the whole collection: a list stores (value, Trust,
Secrecy) triples; a dict stores a
(value, value_trust, value_secrecy, key_trust, key_secrecy) 5-tuple
per entry (the key itself stays a bare raw value, since real dict
lookup needs it to equal what a caller looks up). `for key in
some_dict:` gives each key its own key_trust/key_secrecy, not one
aggregate tag for the dict. A plain list/dict returned by a whitelisted
function gets auto-wrapped: every element takes the call's own
trust/secrecy uniformly, since the interpreter never watched it get
built; a function needing real per-element precision can opt out by
returning already-tagged pairs itself.

A sanitizer/declassifier clears only the *outer* tag on whatever it
returns -- if it passes a tagged container through unchanged, nested
elements keep their own tags. `_has_untrusted`/`_has_secret` check
recursively everywhere an argument is inspected, so a container can't
be laundered by clearing just its top-level label.

**Implicit flow (pc_trust/pc_secrecy).** Explicit-flow checking alone
misses a tagged value that decides which branch runs without ever
appearing as an argument. Every eval_node/exec_stmt call threads a
pc_trust/pc_secrecy value -- the trust/secrecy of the control flow
that led here. Entering an if/while/for whose condition or iterable
was UNTRUSTED (or SECRET) raises pc_trust (or pc_secrecy) for that
block; a privileged/sink call made under a raised pc is refused
regardless of its own arguments. Deliberately narrow: this gates
privileged/sink calls only, it doesn't retroactively taint every value
computed under a raised pc (most practical taint trackers don't
attempt that either, since it tends to make a system too restrictive
to use).

**Operators.** Arithmetic/comparison/boolean results combine (join)
their operands' trust and secrecy. `and`/`or`/chained comparisons
short-circuit like real Python -- an operand skipped by short-circuit
never contributes a tag. `**` additionally caps the exponent's
magnitude (MAX_EXPONENT) before running, since a huge exponent can
hang the process with no loop for MAX_WHILE_ITERATIONS to bound and
no exception to catch. A malformed operation (divide by zero, etc.)
raises whatever ordinary Python exception it would outside this
interpreter -- the whitelist boundary governs what's allowed to run,
not every mistake a legal operation can still make.

**run() vs. Session/run_turn().** run() executes a whole program
blind (the model writes every statement before any run). Session/
run_turn() run one statement at a time against persisted state (env,
budget), so a driving harness can show a real result before the next
statement. pc_trust/pc_secrecy reset fresh for each run_turn() call,
the same as every top-level statement already resets them inside a
single run() call. A Session's budget is shared across turns the same
way run()'s budget is shared across loops in one program.

**Function definitions** (`def name(params): body`) are deliberately
minimal: positional parameters only, no defaults/*args/**kwargs/
keyword-only/decorators/return-type annotation, no `return` statement
-- a function's result is whatever its last statement evaluates to,
the same convention run()/if/while/for already use. A name colliding
with an `allowed` entry is rejected at definition time. A function
body runs against a fresh, isolated local env -- no closures, no
access to the caller's variables. Recursion (direct or mutual) is
refused outright rather than depth-limited, since nothing else bounds
Python's own call-stack depth. A function call inherits the call
site's own pc_trust/pc_secrecy into its body (not reset to
TRUSTED/PUBLIC) -- it's a continuation of the caller's control-flow
context, not a fresh entry point the way run()/run_turn() top-level
statements are; skipping this would let `if untrusted_cond: my_func()`
run an unconditional privileged call inside my_func() unblocked.
"""

from __future__ import annotations

import ast
import operator
from enum import Enum
from typing import Any, Callable

MAX_WHILE_ITERATIONS = 1000
MAX_TOTAL_ITERATIONS = 50000
MAX_EXPONENT = 10_000

_COMPARE_OPS: dict[type, Callable[[Any, Any], bool]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda a, b: _contains(a, b),
    ast.NotIn: lambda a, b: not _contains(a, b),
}

_BINOP_OPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_STRING_METHODS: frozenset[str] = frozenset({
    "startswith", "endswith",
    "strip", "lstrip", "rstrip",
    "lower", "upper",
    "replace", "split", "find", "count",
})
# Deliberately excluded, not just not-yet-added:
#   format/format_map -- format-string injection ("{0.__class__}").
#   encode/decode -- no bytes type in this language's value model.
#   join -- takes an iterable argument whose elements would each need
#     their own trust/secrecy; every method here takes only plain
#     string/int arguments.
#   any regex method -- catastrophic backtracking is a resource-
#     exhaustion vector MAX_WHILE_ITERATIONS/MAX_EXPONENT don't cover.

_UNARY_OPS: dict[type, Callable[[Any], Any]] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Not: operator.not_,
}


class Trust(Enum):
    TRUSTED = "trusted"
    UNTRUSTED = "untrusted"


class Secrecy(Enum):
    PUBLIC = "public"
    SECRET = "secret"


class InterpreterError(Exception):
    pass


class CapabilityError(InterpreterError):
    """An UNTRUSTED value reached a privileged operation (as an
    argument, or via pc_trust). Subclass of InterpreterError."""


class ConfidentialityError(InterpreterError):
    """A SECRET value reached a sink operation (as an argument, or via
    pc_secrecy). Subclass of InterpreterError."""


class _Capabilities:
    """Bundles the six whitelist policy sets (see module docstring)
    into one object, threaded through eval_node/exec_stmt."""

    __slots__ = ("sources", "privileged", "sanitizers", "confidential", "sinks", "declassifiers")

    def __init__(
        self,
        sources: frozenset[str],
        privileged: frozenset[str],
        sanitizers: frozenset[str],
        confidential: frozenset[str],
        sinks: frozenset[str],
        declassifiers: frozenset[str],
    ):
        self.sources = sources
        self.privileged = privileged
        self.sanitizers = sanitizers
        self.confidential = confidential
        self.sinks = sinks
        self.declassifiers = declassifiers


class _IterationBudget:
    """Counts loop iterations across an entire run() call (see module
    docstring's "Loop bounds")."""

    def __init__(self, limit: int):
        self.limit = limit
        self.used = 0

    def consume(self) -> None:
        self.used += 1
        if self.used > self.limit:
            raise InterpreterError(
                f"program exceeded {self.limit} total loop iterations across all loops"
            )


class _FunctionRegistry:
    """Holds user-defined function definitions, the recursion guard
    (call_stack), and a reference to the shared _IterationBudget. One
    registry per run()/run_turn() call, never shared across runs."""

    __slots__ = ("defs", "call_stack", "budget")

    def __init__(self, budget: _IterationBudget):
        self.defs: dict[str, ast.FunctionDef] = {}
        self.call_stack: set[str] = set()
        self.budget = budget


def _call_user_function(
    node: ast.Call,
    name: str,
    allowed: dict[str, Callable],
    env: dict[str, tuple[Any, Trust, Secrecy]],
    caps: "_Capabilities",
    pc_trust: Trust,
    pc_secrecy: Secrecy,
    functions: "_FunctionRegistry",
) -> tuple[Any, Trust, Secrecy]:
    """Calls a user-defined function. Recursion is refused outright
    (call_stack tracks every in-progress function, catching mutual
    recursion too); pc_trust/pc_secrecy carry into the function body
    unchanged rather than resetting; the body runs against a fresh,
    isolated env (no closures). See module docstring's "Function
    definitions" for why on all three."""
    func_def = functions.defs[name]
    if name in functions.call_stack:
        raise InterpreterError(f"recursive function calls are not supported: {name!r}")

    arg_results = [
        eval_node(a, allowed, env, caps, pc_trust, pc_secrecy, functions) for a in node.args
    ]
    kwarg_results = {
        kw.arg: eval_node(kw.value, allowed, env, caps, pc_trust, pc_secrecy, functions)
        for kw in node.keywords
    }

    param_names = [p.arg for p in func_def.args.args]
    if len(node.args) > len(param_names):
        raise InterpreterError(
            f"{name}() takes {len(param_names)} argument(s) but "
            f"{len(node.args)} positional argument(s) were given"
        )
    positional_names = param_names[: len(node.args)]
    remaining_names = set(param_names[len(node.args):])
    provided_kwargs = set(kwarg_results)
    duplicate = provided_kwargs & set(positional_names)
    if duplicate:
        raise InterpreterError(
            f"{name}() got multiple values for argument(s): {', '.join(sorted(duplicate))}"
        )
    unexpected = provided_kwargs - remaining_names
    if unexpected:
        raise InterpreterError(
            f"{name}() got unexpected keyword argument(s): {', '.join(sorted(unexpected))}"
        )
    missing = remaining_names - provided_kwargs
    if missing:
        raise InterpreterError(
            f"{name}() missing required argument(s): {', '.join(sorted(missing))}"
        )

    local_env: dict[str, tuple[Any, Trust, Secrecy]] = {}
    for pname, arg_result in zip(positional_names, arg_results):
        local_env[pname] = arg_result
    for kw in node.keywords:
        local_env[kw.arg] = kwarg_results[kw.arg]

    functions.call_stack.add(name)
    try:
        result = None
        for stmt in func_def.body:
            result = exec_stmt(
                stmt, allowed, local_env, caps, pc_trust, pc_secrecy,
                functions.budget, functions,
            )
    finally:
        functions.call_stack.discard(name)
    return result if result is not None else (None, Trust.TRUSTED, Secrecy.PUBLIC)


def _call_string_method(
    node: ast.Call,
    allowed: dict[str, Callable],
    env: dict[str, tuple[Any, Trust, Secrecy]],
    caps: "_Capabilities",
    pc_trust: Trust,
    pc_secrecy: Secrecy,
    functions: "_FunctionRegistry | None",
) -> tuple[Any, Trust, Secrecy]:
    """Calls a whitelisted method (`_STRING_METHODS`) on a string
    value, e.g. `transaction.date.startswith("2022-03")`. Scoped to
    strings only -- `isinstance(obj, str)` below is the real security
    boundary: it stops this from becoming "call any method on any
    object" (a real returned object could carry unaudited side
    effects) and stops it reaching a *tagged* list/dict, where a
    mutating method would corrupt the tagged-triple/5-tuple invariant.
    No pc_trust/pc_secrecy inheritance needed, unlike function calls
    -- these are pure real `str` methods with no access to
    `allowed`/`caps`, so none can hide a privileged/sink call. Result
    trust/secrecy is the join of the receiver's and every argument's
    tags, same rule as ast.BinOp/ast.Compare."""
    method_name = node.func.attr
    obj, obj_trust, obj_secrecy = eval_node(node.func.value, allowed, env, caps, pc_trust, pc_secrecy, functions)
    if not isinstance(obj, str):
        raise InterpreterError(
            f"method calls are only supported on strings, got {type(obj).__name__!r}"
        )
    if method_name not in _STRING_METHODS:
        raise InterpreterError(f"unsupported string method: {method_name!r}")
    if node.keywords:
        raise InterpreterError(f"{method_name}() does not accept keyword arguments")
    arg_results = [
        eval_node(a, allowed, env, caps, pc_trust, pc_secrecy, functions) for a in node.args
    ]
    result_trust = (
        Trust.UNTRUSTED
        if obj_trust == Trust.UNTRUSTED or any(t == Trust.UNTRUSTED for _, t, _ in arg_results)
        else Trust.TRUSTED
    )
    result_secrecy = (
        Secrecy.SECRET
        if obj_secrecy == Secrecy.SECRET or any(s == Secrecy.SECRET for _, _, s in arg_results)
        else Secrecy.PUBLIC
    )
    args = [unwrap_value(v) for v, _, _ in arg_results]
    try:
        result = getattr(obj, method_name)(*args)
    except (TypeError, ValueError) as exc:
        raise InterpreterError(f"{method_name}() failed: {exc}") from exc
    if isinstance(result, list):
        result = [(item, result_trust, result_secrecy) for item in result]
    return result, result_trust, result_secrecy


def _is_tagged_list(value: Any) -> bool:
    """True if value is already shaped like a list this interpreter
    builds: a list of (value, Trust, Secrecy) triples. Empty list
    counts as tagged, vacuously."""
    return isinstance(value, list) and all(
        isinstance(item, tuple)
        and len(item) == 3
        and isinstance(item[1], Trust)
        and isinstance(item[2], Secrecy)
        for item in value
    )


def _is_tagged_dict(value: Any) -> bool:
    """True if value is already shaped like a dict this interpreter
    builds: every value is a
    (value, value_trust, value_secrecy, key_trust, key_secrecy)
    5-tuple. Empty dict counts as tagged, vacuously. The key itself
    stays raw (unwrapped) -- see module docstring's "Containers"."""
    return isinstance(value, dict) and all(
        isinstance(v, tuple)
        and len(v) == 5
        and isinstance(v[1], Trust)
        and isinstance(v[2], Secrecy)
        and isinstance(v[3], Trust)
        and isinstance(v[4], Secrecy)
        for v in value.values()
    )


def _contains(a: Any, b: Any) -> bool:
    """Implements `a in b`. A tagged list's elements are triples, not
    bare values, so they're unwrapped before comparing -- plain
    Python `in` would otherwise always return False. Dict keys and
    strings are never wrapped, so they fall through unchanged."""
    if _is_tagged_list(b):
        return any(a == v for v, _, _ in b)
    return a in b


def unwrap_value(value: Any) -> Any:
    """Recursively strips Trust/Secrecy tags from a tagged list or
    dict, down to plain Python values. Used at every boundary where
    the interpreter's internal bookkeeping must never leak past: an
    external function call's arguments, and run()/run_turn()'s own
    return value."""
    if _is_tagged_list(value):
        return [unwrap_value(v) for v, _, _ in value]
    if _is_tagged_dict(value):
        return {k: unwrap_value(v) for k, (v, _, _, _, _) in value.items()}
    return value


def _has_untrusted(value: Any, trust: Trust) -> bool:
    """True if trust is UNTRUSTED, or value is a tagged list/dict with
    an untrusted element anywhere, recursively -- the laundering check
    from module docstring's "Containers", used everywhere an
    argument's trust is inspected."""
    if trust == Trust.UNTRUSTED:
        return True
    if isinstance(value, list) and _is_tagged_list(value):
        return any(_has_untrusted(v, t) for v, t, _ in value)
    if isinstance(value, dict) and _is_tagged_dict(value):
        # kt checked directly, not recursively: a dict key must be
        # hashable, so it can never itself be a nested tagged list/dict.
        return any(
            _has_untrusted(v, vt) or kt == Trust.UNTRUSTED
            for v, vt, _, kt, _ in value.values()
        )
    return False


def _has_secret(value: Any, secrecy: Secrecy) -> bool:
    """The confidentiality counterpart to _has_untrusted."""
    if secrecy == Secrecy.SECRET:
        return True
    if isinstance(value, list) and _is_tagged_list(value):
        return any(_has_secret(v, s) for v, _, s in value)
    if isinstance(value, dict) and _is_tagged_dict(value):
        # Same reasoning as _has_untrusted above.
        return any(
            _has_secret(v, vs) or ks == Secrecy.SECRET
            for v, _, vs, _, ks in value.values()
        )
    return False


def eval_node(
    node: ast.AST,
    allowed: dict[str, Callable],
    env: dict[str, tuple[Any, Trust, Secrecy]],
    caps: _Capabilities,
    pc_trust: Trust = Trust.TRUSTED,
    pc_secrecy: Secrecy = Secrecy.PUBLIC,
    functions: "_FunctionRegistry | None" = None,
) -> tuple[Any, Trust, Secrecy]:
    """Evaluates a single expression node. Each case is explicit;
    anything else raises rather than falling back to a general
    evaluator. Returns (value, trust, secrecy), never a bare value.
    pc_trust/pc_secrecy are the control flow's own tags that led here
    -- see module docstring's "Implicit flow"."""
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute):
            return _call_string_method(node, allowed, env, caps, pc_trust, pc_secrecy, functions)
        if not isinstance(node.func, ast.Name):
            raise InterpreterError("calls must be a plain function name")
        name = node.func.id
        if name not in allowed:
            if functions is not None and name in functions.defs:
                return _call_user_function(node, name, allowed, env, caps, pc_trust, pc_secrecy, functions)
            raise InterpreterError(f"unknown or disallowed name: {name}")
        arg_results = [
            eval_node(a, allowed, env, caps, pc_trust, pc_secrecy, functions) for a in node.args
        ]
        kwarg_results = {
            kw.arg: eval_node(kw.value, allowed, env, caps, pc_trust, pc_secrecy, functions)
            for kw in node.keywords
        }
        all_args = arg_results + list(kwarg_results.values())
        any_untrusted = any(_has_untrusted(v, t) for v, t, _ in all_args)
        any_secret = any(_has_secret(v, s) for v, _, s in all_args)
        if name in caps.privileged and any_untrusted:
            raise CapabilityError(
                f"privileged operation {name!r} called with an untrusted argument"
            )
        if name in caps.privileged and pc_trust == Trust.UNTRUSTED:
            raise CapabilityError(
                f"privileged operation {name!r} called from a branch whose "
                "condition depended on untrusted data"
            )
        if name in caps.sinks and any_secret:
            raise ConfidentialityError(
                f"sink operation {name!r} called with a secret argument"
            )
        if name in caps.sinks and pc_secrecy == Secrecy.SECRET:
            raise ConfidentialityError(
                f"sink operation {name!r} called from a branch whose "
                "condition depended on secret data"
            )
        args = [unwrap_value(v) for v, _, _ in arg_results]
        kwargs = {k: unwrap_value(v) for k, (v, _, _) in kwarg_results.items()}
        # External functions see/return plain values only. If a
        # function hands back the exact same list/dict object it was
        # given (identity, not equality), its original per-element
        # tags are restored below instead of auto-wrapping -- auto-
        # wrap would relabel every nested element with the call's
        # outer trust and launder anything still untrusted/secret
        # inside. Scoped to this one call, by object identity; doesn't
        # cover a function that mutates the container in place.
        passthrough = {
            id(v): orig
            for v, orig in list(zip(args, arg_results)) + [(kwargs[k], kwarg_results[k]) for k in kwargs]
            if _is_tagged_list(orig[0]) or _is_tagged_dict(orig[0])
        }
        result = allowed[name](*args, **kwargs)
        if id(result) in passthrough:
            result = passthrough[id(result)][0]
        if name in caps.sources:
            result_trust = Trust.UNTRUSTED
        elif name in caps.sanitizers:
            result_trust = Trust.TRUSTED
        elif any_untrusted:
            result_trust = Trust.UNTRUSTED
        else:
            result_trust = Trust.TRUSTED
        if name in caps.confidential:
            result_secrecy = Secrecy.SECRET
        elif name in caps.declassifiers:
            result_secrecy = Secrecy.PUBLIC
        elif any_secret:
            result_secrecy = Secrecy.SECRET
        else:
            result_secrecy = Secrecy.PUBLIC
        if isinstance(result, list) and not _is_tagged_list(result):
            result = [(item, result_trust, result_secrecy) for item in result]
        elif isinstance(result, dict) and not _is_tagged_dict(result):
            # Uniform, not precise, same as list auto-wrap: every key
            # and value gets the call's own result_trust/result_secrecy.
            result = {
                k: (v, result_trust, result_secrecy, result_trust, result_secrecy)
                for k, v in result.items()
            }
        return result, result_trust, result_secrecy
    if isinstance(node, ast.Constant):
        return node.value, Trust.TRUSTED, Secrecy.PUBLIC
    if isinstance(node, ast.Name):
        if not isinstance(node.ctx, ast.Load):
            raise InterpreterError(f"unsupported name usage: {ast.dump(node)}")
        if node.id not in env:
            raise InterpreterError(f"undefined variable: {node.id}")
        return env[node.id]
    if isinstance(node, ast.Compare):
        for op_node in node.ops:
            if type(op_node) not in _COMPARE_OPS:
                raise InterpreterError(f"unsupported comparison operator: {type(op_node).__name__}")
        result = True
        result_trust = Trust.TRUSTED
        result_secrecy = Secrecy.PUBLIC
        prev, prev_trust, prev_secrecy = eval_node(node.left, allowed, env, caps, pc_trust, pc_secrecy, functions)
        result_trust = Trust.UNTRUSTED if prev_trust == Trust.UNTRUSTED else result_trust
        result_secrecy = Secrecy.SECRET if prev_secrecy == Secrecy.SECRET else result_secrecy
        for op_node, comparator_node in zip(node.ops, node.comparators):
            curr, curr_trust, curr_secrecy = eval_node(
                comparator_node, allowed, env, caps, pc_trust, pc_secrecy, functions
            )
            result_trust = Trust.UNTRUSTED if curr_trust == Trust.UNTRUSTED else result_trust
            result_secrecy = Secrecy.SECRET if curr_secrecy == Secrecy.SECRET else result_secrecy
            if not _COMPARE_OPS[type(op_node)](prev, curr):
                result = False
                break
            prev, prev_trust, prev_secrecy = curr, curr_trust, curr_secrecy
        return result, result_trust, result_secrecy
    if isinstance(node, ast.BoolOp):
        op_type = type(node.op)
        result = None
        result_trust = Trust.TRUSTED
        result_secrecy = Secrecy.PUBLIC
        for value_node in node.values:
            result, value_trust, value_secrecy = eval_node(
                value_node, allowed, env, caps, pc_trust, pc_secrecy, functions
            )
            result_trust = Trust.UNTRUSTED if value_trust == Trust.UNTRUSTED else result_trust
            result_secrecy = Secrecy.SECRET if value_secrecy == Secrecy.SECRET else result_secrecy
            if op_type is ast.And and not result:
                break
            if op_type is ast.Or and result:
                break
        return result, result_trust, result_secrecy
    if isinstance(node, ast.IfExp):
        # Ternary -- the expression-level twin of ast.If's
        # implicit-flow protection, applied to whichever single branch
        # is actually evaluated (lazy, so the unchosen branch's side
        # effects never run). The chosen branch's own trust/secrecy is
        # not additionally tainted by the test's tag, matching ast.If's
        # choice not to retroactively taint a branch's own assignment.
        test_value, test_trust, test_secrecy = eval_node(
            node.test, allowed, env, caps, pc_trust, pc_secrecy, functions
        )
        branch_pc_trust = Trust.UNTRUSTED if pc_trust == Trust.UNTRUSTED or test_trust == Trust.UNTRUSTED else Trust.TRUSTED
        branch_pc_secrecy = Secrecy.SECRET if pc_secrecy == Secrecy.SECRET or test_secrecy == Secrecy.SECRET else Secrecy.PUBLIC
        chosen = node.body if test_value else node.orelse
        return eval_node(chosen, allowed, env, caps, branch_pc_trust, branch_pc_secrecy, functions)
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise InterpreterError(f"unsupported unary operator: {op_type.__name__}")
        operand, operand_trust, operand_secrecy = eval_node(
            node.operand, allowed, env, caps, pc_trust, pc_secrecy, functions
        )
        return _UNARY_OPS[op_type](operand), operand_trust, operand_secrecy
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BINOP_OPS:
            raise InterpreterError(f"unsupported arithmetic operator: {op_type.__name__}")
        left, left_trust, left_secrecy = eval_node(node.left, allowed, env, caps, pc_trust, pc_secrecy, functions)
        right, right_trust, right_secrecy = eval_node(node.right, allowed, env, caps, pc_trust, pc_secrecy, functions)
        if op_type is ast.Pow and isinstance(right, (int, float)) and abs(right) > MAX_EXPONENT:
            raise InterpreterError(f"exponent magnitude exceeds {MAX_EXPONENT}")
        binop_trust = Trust.UNTRUSTED if Trust.UNTRUSTED in (left_trust, right_trust) else Trust.TRUSTED
        binop_secrecy = Secrecy.SECRET if Secrecy.SECRET in (left_secrecy, right_secrecy) else Secrecy.PUBLIC
        return _BINOP_OPS[op_type](left, right), binop_trust, binop_secrecy
    if isinstance(node, ast.List):
        elements = [eval_node(e, allowed, env, caps, pc_trust, pc_secrecy, functions) for e in node.elts]
        list_trust = Trust.UNTRUSTED if any(t == Trust.UNTRUSTED for _, t, _ in elements) else Trust.TRUSTED
        list_secrecy = Secrecy.SECRET if any(s == Secrecy.SECRET for _, _, s in elements) else Secrecy.PUBLIC
        return elements, list_trust, list_secrecy
    if isinstance(node, ast.Dict):
        if any(k is None for k in node.keys):
            raise InterpreterError("dict unpacking (**) is not supported")
        entries = []
        for key_node, value_node in zip(node.keys, node.values):
            key, key_trust, key_secrecy = eval_node(key_node, allowed, env, caps, pc_trust, pc_secrecy, functions)
            value_result = eval_node(value_node, allowed, env, caps, pc_trust, pc_secrecy, functions)
            entries.append((key, key_trust, key_secrecy, value_result))
        # Entry: (value, value_trust, value_secrecy, key_trust,
        # key_secrecy). The key stays the bare raw value so plain-key
        # lookup keeps working; see _is_tagged_dict's docstring.
        result_dict: dict[Any, tuple[Any, Trust, Secrecy, Trust, Secrecy]] = {}
        for key, key_trust, key_secrecy, (value, value_trust, value_secrecy) in entries:
            try:
                result_dict[key] = (value, value_trust, value_secrecy, key_trust, key_secrecy)
            except TypeError as exc:
                raise InterpreterError(f"unhashable dict key: {key!r}") from exc
        dict_trust = Trust.TRUSTED
        dict_secrecy = Secrecy.PUBLIC
        for _, key_trust, key_secrecy, (_, value_trust, value_secrecy) in entries:
            if key_trust == Trust.UNTRUSTED or value_trust == Trust.UNTRUSTED:
                dict_trust = Trust.UNTRUSTED
            if key_secrecy == Secrecy.SECRET or value_secrecy == Secrecy.SECRET:
                dict_secrecy = Secrecy.SECRET
        return result_dict, dict_trust, dict_secrecy
    if isinstance(node, ast.Subscript):
        if not isinstance(node.ctx, ast.Load):
            raise InterpreterError(f"unsupported subscript usage: {ast.dump(node)}")
        container, _, _ = eval_node(node.value, allowed, env, caps, pc_trust, pc_secrecy, functions)
        index, _, _ = eval_node(node.slice, allowed, env, caps, pc_trust, pc_secrecy, functions)
        if _is_tagged_list(container):
            if not isinstance(index, int) or isinstance(index, bool):
                raise InterpreterError("list index must be an integer")
            try:
                return container[index]
            except IndexError as exc:
                raise InterpreterError(f"list index out of range: {index}") from exc
        if _is_tagged_dict(container):
            try:
                entry = container[index]
            except KeyError as exc:
                raise InterpreterError(f"dict has no key: {index!r}") from exc
            except TypeError as exc:
                raise InterpreterError(f"unhashable dict key: {index!r}") from exc
            # entry: (value, value_trust, value_secrecy, key_trust,
            # key_secrecy). Indexing returns the value; the key's tag
            # matters for iteration (ast.For), not for lookup.
            value, value_trust, value_secrecy, _key_trust, _key_secrecy = entry
            return value, value_trust, value_secrecy
        raise InterpreterError(
            "subscripting requires a list or dict built with a literal (or "
            "a function in `allowed` returning one) -- a plain value has "
            "no per-element trust or secrecy to read"
        )
    if isinstance(node, ast.Attribute):
        if not isinstance(node.ctx, ast.Load):
            raise InterpreterError(f"unsupported attribute usage: {ast.dump(node)}")
        # Reads a named field off a real returned object (e.g. a
        # Transaction's .amount). Trust/secrecy pass through unchanged
        # from the base object, same principle as subscripting a tagged
        # container's elements. Can't become a call-through-attribute
        # escape: ast.Call only dispatches by a literal name looked up
        # in `allowed`, never against a value reached via attribute
        # access. Dunder/private names are still blocked below as
        # defense in depth (__dict__, __class__, and a property/
        # __getattr__ override running code as a side effect of a plain
        # read) -- a real but currently unmitigated risk this project's
        # own tools (dataclasses, pydantic models) don't exercise.
        obj, trust, secrecy = eval_node(node.value, allowed, env, caps, pc_trust, pc_secrecy, functions)
        if node.attr.startswith("_"):
            raise InterpreterError(f"attribute access to {node.attr!r} is not allowed")
        try:
            value = getattr(obj, node.attr)
        except AttributeError as exc:
            raise InterpreterError(f"{type(obj).__name__!r} object has no attribute {node.attr!r}") from exc
        if callable(value):
            raise InterpreterError(
                f"attribute {node.attr!r} is a method, not a field -- method calls are not supported"
            )
        return value, trust, secrecy
    raise InterpreterError(f"unsupported expression: {ast.dump(node)}")


def exec_stmt(
    node: ast.stmt,
    allowed: dict[str, Callable],
    env: dict[str, tuple[Any, Trust, Secrecy]],
    caps: _Capabilities,
    pc_trust: Trust = Trust.TRUSTED,
    pc_secrecy: Secrecy = Secrecy.PUBLIC,
    budget: _IterationBudget | None = None,
    functions: "_FunctionRegistry | None" = None,
) -> tuple[Any, Trust, Secrecy] | None:
    """Executes a single statement node. Returns (value, trust,
    secrecy) for an expr_stmt, None otherwise. Unsupported statement
    types raise rather than being silently skipped. budget is the
    shared iteration counter, or None to skip the global cap."""
    if isinstance(node, ast.Assign):
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            raise InterpreterError("assignment must have a single plain-name target")
        env[node.targets[0].id] = eval_node(node.value, allowed, env, caps, pc_trust, pc_secrecy, functions)
        return None
    if isinstance(node, ast.If):
        test_value, test_trust, test_secrecy = eval_node(
            node.test, allowed, env, caps, pc_trust, pc_secrecy, functions
        )
        branch_pc_trust = Trust.UNTRUSTED if pc_trust == Trust.UNTRUSTED or test_trust == Trust.UNTRUSTED else Trust.TRUSTED
        branch_pc_secrecy = Secrecy.SECRET if pc_secrecy == Secrecy.SECRET or test_secrecy == Secrecy.SECRET else Secrecy.PUBLIC
        branch = node.body if test_value else node.orelse
        result = None
        for stmt in branch:
            result = exec_stmt(stmt, allowed, env, caps, branch_pc_trust, branch_pc_secrecy, budget, functions)
        return result
    if isinstance(node, ast.While):
        if node.orelse:
            raise InterpreterError("while/else is not supported")
        result = None
        iterations = 0
        test_value, test_trust, test_secrecy = eval_node(
            node.test, allowed, env, caps, pc_trust, pc_secrecy, functions
        )
        while test_value:
            iterations += 1
            if iterations > MAX_WHILE_ITERATIONS:
                raise InterpreterError(
                    f"while loop exceeded {MAX_WHILE_ITERATIONS} iterations"
                )
            if budget is not None:
                budget.consume()
            body_pc_trust = Trust.UNTRUSTED if pc_trust == Trust.UNTRUSTED or test_trust == Trust.UNTRUSTED else Trust.TRUSTED
            body_pc_secrecy = Secrecy.SECRET if pc_secrecy == Secrecy.SECRET or test_secrecy == Secrecy.SECRET else Secrecy.PUBLIC
            for stmt in node.body:
                result = exec_stmt(stmt, allowed, env, caps, body_pc_trust, body_pc_secrecy, budget, functions)
            test_value, test_trust, test_secrecy = eval_node(
                node.test, allowed, env, caps, pc_trust, pc_secrecy, functions
            )
        return result
    if isinstance(node, ast.For):
        if node.orelse:
            raise InterpreterError("for/else is not supported")
        if not isinstance(node.target, ast.Name):
            raise InterpreterError("for loop target must be a single plain name")
        iterable, iterable_trust, iterable_secrecy = eval_node(
            node.iter, allowed, env, caps, pc_trust, pc_secrecy, functions
        )
        if _is_tagged_list(iterable):
            tagged = iterable
        elif _is_tagged_dict(iterable):
            # Iterates keys, matching Python's `for x in some_dict:`.
            # Each key keeps its own trust/secrecy, not the dict's
            # aggregate tag -- see _is_tagged_dict's docstring.
            tagged = [(key, key_trust, key_secrecy) for key, (_, _, _, key_trust, key_secrecy) in iterable.items()]
        else:
            raise InterpreterError(
                "a for loop's iterable requires a list or dict built with a "
                "literal (or a function in `allowed` returning one) -- a "
                "plain value has no per-element trust or secrecy to read"
            )
        body_pc_trust = Trust.UNTRUSTED if pc_trust == Trust.UNTRUSTED or iterable_trust == Trust.UNTRUSTED else Trust.TRUSTED
        body_pc_secrecy = Secrecy.SECRET if pc_secrecy == Secrecy.SECRET or iterable_secrecy == Secrecy.SECRET else Secrecy.PUBLIC
        result = None
        for element in tagged:
            if budget is not None:
                budget.consume()
            env[node.target.id] = element
            for stmt in node.body:
                result = exec_stmt(stmt, allowed, env, caps, body_pc_trust, body_pc_secrecy, budget, functions)
        return result
    if isinstance(node, ast.Expr):
        return eval_node(node.value, allowed, env, caps, pc_trust, pc_secrecy, functions)
    if isinstance(node, ast.FunctionDef):
        # `def name(params): body` -- plain positional parameters only,
        # no defaults/*args/**kwargs/keyword-only/decorators/return-
        # type annotation, and no `return` statement. A function's
        # result is whatever its last statement evaluates to, the same
        # convention run()/run_turn()/if/while/for already use.
        if node.decorator_list:
            raise InterpreterError("function decorators are not supported")
        if node.returns is not None:
            raise InterpreterError("function return type annotations are not supported")
        args = node.args
        if args.vararg or args.kwarg or args.kwonlyargs or args.posonlyargs or args.defaults or args.kw_defaults:
            raise InterpreterError(
                "function definitions only support plain positional "
                "parameters -- no *args, **kwargs, keyword-only "
                "parameters, or default values"
            )
        if node.name in allowed:
            raise InterpreterError(
                f"cannot define a function named {node.name!r} -- it "
                "collides with a whitelisted name"
            )
        if functions is None:
            raise InterpreterError("function definitions are not supported in this context")
        functions.defs[node.name] = node
        return None
    raise InterpreterError(f"unsupported statement: {ast.dump(node)}")


def run(
    source: str,
    allowed: dict[str, Callable],
    env: dict[str, tuple[Any, Trust, Secrecy]] | None = None,
    *,
    sources: frozenset[str] = frozenset(),
    privileged: frozenset[str] = frozenset(),
    sanitizers: frozenset[str] = frozenset(),
    confidential: frozenset[str] = frozenset(),
    sinks: frozenset[str] = frozenset(),
    declassifiers: frozenset[str] = frozenset(),
) -> Any:
    """Parses source (exec mode, never eval()) and runs each top-level
    statement through exec_stmt. Returns the bare value of the last
    expression statement (Trust/Secrecy unwrapped here), or None if it
    ended on an assignment or empty branch. Raises InterpreterError on
    a parse failure, CapabilityError/ConfidentialityError per the
    rules in the module docstring. One fresh _IterationBudget is
    shared across every loop in the whole program."""
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise InterpreterError(f"could not parse: {exc}") from exc
    if env is None:
        env = {}
    caps = _Capabilities(sources, privileged, sanitizers, confidential, sinks, declassifiers)
    budget = _IterationBudget(MAX_TOTAL_ITERATIONS)
    functions = _FunctionRegistry(budget)
    result = None
    for stmt in tree.body:
        result = exec_stmt(stmt, allowed, env, caps, Trust.TRUSTED, Secrecy.PUBLIC, budget, functions)
    return unwrap_value(result[0]) if result is not None else None


class Session:
    """Holds the env and iteration budget that persist across
    run_turn() calls for one task. Two Sessions are always
    independent; nothing here is process-global."""

    def __init__(self, budget_limit: int = MAX_TOTAL_ITERATIONS):
        self.env: dict[str, tuple[Any, Trust, Secrecy]] = {}
        self.budget = _IterationBudget(budget_limit)
        self.functions = _FunctionRegistry(self.budget)


def run_turn(
    session: Session,
    source: str,
    allowed: dict[str, Callable],
    *,
    sources: frozenset[str] = frozenset(),
    privileged: frozenset[str] = frozenset(),
    sanitizers: frozenset[str] = frozenset(),
    confidential: frozenset[str] = frozenset(),
    sinks: frozenset[str] = frozenset(),
    declassifiers: frozenset[str] = frozenset(),
) -> Any:
    """Like run(), but source must be exactly one top-level statement,
    executed against session's accumulated env and budget (raises
    InterpreterError otherwise). pc_trust/pc_secrecy start fresh each
    call, same as every top-level statement in run(). Capability rules
    are passed per call, same as run(); only the env and budget carry
    over between calls."""
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise InterpreterError(f"could not parse: {exc}") from exc
    if len(tree.body) != 1:
        raise InterpreterError(
            f"a turn must be exactly one statement, got {len(tree.body)}"
        )
    caps = _Capabilities(sources, privileged, sanitizers, confidential, sinks, declassifiers)
    result = exec_stmt(
        tree.body[0], allowed, session.env, caps, Trust.TRUSTED, Secrecy.PUBLIC,
        session.budget, session.functions,
    )
    return unwrap_value(result[0]) if result is not None else None
