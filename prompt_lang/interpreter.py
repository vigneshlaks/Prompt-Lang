"""A safe core for evaluating expressions a language model writes, without
ever calling eval(). This is the seed of a real interpreter, ported from
provenance-ac's agent_demo/agent_loop.py, where the same pattern parsed
and dispatched a single tool call safely. Here it's meant to grow into the
actual interpreter for a constrained language: more statement types get
added by extending eval_node/exec_stmt, not by relaxing the whitelist
check.

The rule that has to hold no matter how much this grows: parsing uses only
ast.parse, dispatch goes only through an explicit whitelist of node types
and names, and eval() is never called on anything a model produced.

Toy grammar supported so far (informal EBNF):

    program    := statement*
    statement  := assign | if_stmt | while_stmt | for_stmt | expr_stmt
    assign     := NAME "=" expr
    if_stmt    := "if" expr ":" NEWLINE INDENT statement+ DEDENT
                  ("else" ":" NEWLINE INDENT statement+ DEDENT)?
    while_stmt := "while" expr ":" NEWLINE INDENT statement+ DEDENT
    for_stmt   := "for" NAME "in" expr ":" NEWLINE INDENT statement+ DEDENT
    expr_stmt  := expr
    expr       := call | compare | arith | boolean | unary | subscript
                  | attribute | list_expr | dict_expr | ternary | NAME | literal
    call       := NAME "(" (expr ("," expr)*)? ")"
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

Two separate namespaces: `allowed` (whitelisted callables, checked at call
sites only) and `env` (variables bound by assignment, checked at name-read
sites only). A bare reference to a whitelisted function name is not a
variable read and is rejected, same as before this file supported
variables at all.

Capability tracking covers two independent questions about every value:
integrity (is this data trustworthy enough to safely drive a privileged
action) and confidentiality (is this data sensitive enough that it must
not reach an untrusted destination). They're tracked as two separate tags,
Trust and Secrecy, because they're genuinely independent -- a value can be
trustworthy but secret (a real API key), or untrustworthy but not remotely
sensitive (an attacker's email body). Both tags are threaded through
eval_node and exec_stmt rather than attached via object identity -- there's
no side table, because the interpreter owns every value's lifecycle itself.

Integrity: `sources` names a subset of `allowed` whose return value is
always UNTRUSTED, regardless of its arguments. `privileged` names a subset
of `allowed` that refuses to run at all if any argument is UNTRUSTED,
raising CapabilityError before the underlying function is ever called. An
ordinary (non-source, non-privileged, non-sanitizer) function propagates
trust from its arguments: UNTRUSTED in, UNTRUSTED out, through any chain
of calls, since eval_node evaluates arguments recursively before combining
their trust. `sanitizers` names a subset of `allowed` whose return value
is always TRUSTED, regardless of its arguments -- the one deliberate,
explicitly declared way for a program to turn UNTRUSTED back into TRUSTED.
Only a name explicitly listed in `sanitizers` has this effect; nothing
about a function's behavior implies it. If a name is in both `sources` and
`sanitizers`, `sources` wins, since the untrusted outcome is the safer one
to prefer when the configuration itself is contradictory.

Confidentiality is the mirror image, checked in the opposite direction.
`confidential` names a subset of `allowed` whose return value is always
SECRET, regardless of its arguments. `sinks` names a subset of `allowed`
that refuses to run at all if any argument is SECRET, raising
ConfidentialityError -- the confidentiality counterpart to CapabilityError.
Secrecy propagates through ordinary functions the same way trust does:
SECRET in, SECRET out, through any chain of calls. `declassifiers` names a
subset of `allowed` whose return value is always PUBLIC regardless of its
arguments -- the deliberate, explicitly declared way to clear a secrecy
tag, the confidentiality counterpart to `sanitizers`. If a name is in both
`confidential` and `declassifiers`, `confidential` wins, for the same
fail-safe reason `sources` wins over `sanitizers`.

The six capability sets above are bundled into one `_Capabilities` object
rather than threaded as six separate parameters through every recursive
eval_node/exec_stmt call. That's not decoration: with six independent
whitelist categories, a positional-parameter mistake in one of the many
recursive call sites would silently break a security check without
raising any error at all, and this is exactly the kind of mistake a test
suite that doesn't specifically probe the broken category wouldn't catch.
run()'s public keyword arguments are unaffected by this -- sources,
privileged, sanitizers, confidential, sinks, and declassifiers are still
passed the same way they always were; run() bundles them into a
_Capabilities instance once and threads that single object down instead.

while loops are bounded two ways. MAX_WHILE_ITERATIONS caps a single loop
-- a condition that never goes false raises InterpreterError instead of
running forever. That alone doesn't bound total program execution, since
each while node resets its own counter independently: two loops nested
inside each other can each individually stay under the cap while the
program as a whole does far more work than the cap suggests (found by
deliberately trying it -- 300 outer iterations times 300 inner iterations
ran 90,000 total operations with neither loop ever exceeding 1000). A
shared _IterationBudget, created once per run() call and threaded through
every exec_stmt call, counts iterations across the whole program --
every while pass and every for pass consumes from it -- so nesting can't
multiply past a single fixed total.

Lists and dicts both carry trust and secrecy per element, not as one tag
for the whole collection. A list literal evaluates each element the
normal way and stores the full list of (value, Trust, Secrecy) triples as
its value. A dict literal stores, per entry, a
(value, value_trust, value_secrecy, key_trust, key_secrecy) 5-tuple --
the key itself stays a bare raw value (real dict lookup by a plain key
has to keep working, and a tagged key would never equal the bare value
someone looks it up by), with its own trust/secrecy carried alongside
rather than folded into the key. Indexing and for loops both work on
lists and dicts: indexing a dict reads an entry's value and its own
trust/secrecy; `for key in some_dict:` iterates keys, each with its own
precise key_trust/key_secrecy rather than one aggregate tag for the
whole dict -- found worth doing precisely, not with the cheaper
shortcut of tagging every key with the dict's own overall trust,
because that shortcut would let one untrusted entry drag every other,
individually fine key down with it, an over-restriction cost with a
real, already-observed precedent in this project (see notes on the
implicit-flow pc_trust mitigation elsewhere in this file). A container
mixing trusted and untrusted, or public and secret, items keeps each
item's own tags instead of collapsing to one.

A plain Python list or dict handed back by a function in `allowed` has no
per-element tags of its own -- the interpreter never watched it get
built. Rather than reject it, eval_node auto-wraps it: every element gets
the same trust and secrecy the call as a whole already earned, turning a
plain container into the same tagged shape a literal produces. This is
uniform, not precise -- every element in an auto-wrapped container shares
one trust value and one secrecy value, since the interpreter has no
finer-grained information to give it. A function that needs real
per-element precision can opt out by returning already-tagged pairs
itself; auto-wrap detects that shape and leaves it untouched.

A sanitizer (or declassifier) clears the outer tag on whatever it
returns, but if it just passes a tagged container through unchanged, the
tags on the individual elements inside are untouched -- the outer tag and
the nested tags can disagree. _has_untrusted and _has_secret both check
this recursively, everywhere an argument's trust or secrecy is inspected
(the privileged/sink gates and ordinary propagation), so a container
can't be laundered by clearing only its own top-level label while what's
nested inside stays untrusted or secret.

Neither integrity nor confidentiality tracking is complete on explicit
data flow alone: does a tagged value reach a call as an argument. Neither
sees implicit flow by default: a tagged value can decide which branch of
an if/while runs without ever appearing as an argument, and a call in that
branch that takes no matching-tagged argument (or no argument at all) is
invisible to the argument check. Both sides now have a scoped, partial
mitigation for this, tracked in parallel: every eval_node/exec_stmt call
threads a pc_trust value and a pc_secrecy value -- the trust and secrecy
of the control flow that led to this point in the program. Entering an if
branch, a while body, or a for body whose controlling condition or
iterable was UNTRUSTED raises pc_trust to UNTRUSTED for that block, and
likewise raises pc_secrecy to SECRET if the condition or iterable was
SECRET. A privileged call made while pc_trust is UNTRUSTED is refused
regardless of its own arguments; a sink call made while pc_secrecy is
SECRET is refused the same way. This is deliberately narrow on both
sides -- it gates privileged/sink calls only, it does not retroactively
taint every value computed under a raised pc the way a complete
implicit-flow system would (assigning to a variable that already exists,
for instance, isn't currently affected). Real implicit-flow tracking done
properly tends to make a system very restrictive, since almost anything
computed inside a branch on tagged data ends up needing the same
treatment; most practical taint trackers (Perl's taint mode among them)
deliberately don't attempt it for exactly that reason. This mitigation is
a bounded compromise, not a complete solution, scoped to the concrete
cases found by deliberately testing for them: a privileged or sink call
with no matching-tagged arguments, reachable only through a branch a
tagged value controlled.

Arithmetic (+, -, *, /, //, %, **) is handled the same way comparisons
are: a small dict maps AST operator node types to the matching function
from the `operator` module, and the result's trust and secrecy are the
join of its two operands, exactly like ast.Compare already computes. A
malformed operation (dividing by zero, adding a number to a string) is
not caught or converted into an InterpreterError -- it raises whatever
ordinary Python exception it would outside this interpreter, the same
way a whitelisted function called with the wrong argument types already
does. The whitelist boundary is about what's allowed to run, not about
catching every mistake a legal operation can still make.

** (exponentiation) is the one arithmetic operator that gets an extra
check first: a single expression like x ** huge_number can hang the
process or exhaust memory in one step, with no exception to catch and
no loop for MAX_WHILE_ITERATIONS to bound -- the same class of concern
loops had before that cap existed, just faster and with no iteration to
count. MAX_EXPONENT caps the exponent's magnitude before ** ever runs;
the base is unrestricted, since a large base raised to a small exponent
is cheap (squaring a big number is not the danger; a small number raised
to a huge one is).

ast.Compare now supports chained comparisons (0 <= x <= 100), not just
a single pair. Each operand is still evaluated exactly once, left to
right, matching real Python semantics -- x() < y() < z() calls y() once,
not twice, even though y's value is used in two comparisons. The whole
chain short-circuits the moment one comparison is false, the same way
Python's own chained comparisons do, and only operands actually
evaluated before the short-circuit contribute to the result's trust and
secrecy; an operand skipped because an earlier comparison already failed
carries no information into the result.

Boolean operators (and, or) short-circuit the same way: `and` stops at
the first falsy operand, `or` stops at the first truthy one, evaluating
operands left to right and stopping as soon as the outcome is decided,
exactly like real Python. Trust and secrecy accumulate only from
operands that were actually evaluated -- a later operand skipped by
short-circuiting was never read, so its tag can't matter to the result.

Unary operators (-x, +x, not x) pass the single operand's trust and
secrecy straight through unchanged; there's only one input; there's
nothing to join. Adding unary minus specifically also fixes a real gap,
not just a missing operator: ast.parse never folds a negative literal
into a single constant -- `-5` parses as UnaryOp(USub, Constant(5)), two
nodes, not one -- so before this, the interpreter had no negative number
literals at all, not even as simple as `-5` on its own.

run() executes a whole program blind: a model writes every statement
before any of them run, never seeing a real result along the way.
Session and run_turn() are the alternative -- one statement at a time,
against state (env, the iteration budget) that persists across calls,
so a driving harness can show the real result of one statement back to
a model before asking it to write the next one. This doesn't change
eval_node or exec_stmt at all; it only changes what creates env and the
budget and when. pc_trust and pc_secrecy still start fresh
(TRUSTED/PUBLIC) for each statement passed to run_turn(), which isn't a
new rule -- run()'s own top-level loop already resets both for every
top-level statement in a single program, so run_turn() just lets each
of those arrive in its own call instead of all together in one source
string. A Session's budget is shared across every turn the same way
run()'s budget is shared across every loop in one program, closing the
same class of gap the same way: many turns, each individually
unremarkable, can't multiply past a fixed total any more than many
nested loops could.
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
    """Raised when an UNTRUSTED value reaches a privileged operation,
    either directly as an argument or by controlling which branch a
    privileged call sits in. A subclass of InterpreterError so existing
    callers that catch the whitelist-boundary error also catch this,
    while code that cares can still distinguish the two."""


class ConfidentialityError(InterpreterError):
    """Raised when a SECRET value reaches a sink operation, either
    directly as an argument or by controlling which branch a sink call
    sits in -- the confidentiality counterpart to CapabilityError. A
    subclass of InterpreterError so existing callers that catch the
    whitelist-boundary error also catch this, while code that cares can
    still distinguish it from CapabilityError, its integrity twin."""


class _Capabilities:
    """Bundles the six whitelist policy sets threaded through every
    recursive eval_node/exec_stmt call, so adding one more capability
    category means adding one field here, not one more positional
    parameter at every one of the many recursive call sites -- exactly
    the kind of place a silent ordering mistake would break a security
    check without anyone noticing."""

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
    """Counts loop iterations across an entire run() call, not just
    within one loop. MAX_WHILE_ITERATIONS bounds a single while loop, but
    that alone doesn't bound total program execution, since nested loops
    each reset their own counter independently. One shared budget per
    run() call closes that gap."""

    def __init__(self, limit: int):
        self.limit = limit
        self.used = 0

    def consume(self) -> None:
        self.used += 1
        if self.used > self.limit:
            raise InterpreterError(
                f"program exceeded {self.limit} total loop iterations across all loops"
            )


def _is_tagged_list(value: Any) -> bool:
    """True if value is already shaped like a list this interpreter
    builds: a list of (value, Trust, Secrecy) triples. An empty list
    counts as tagged, vacuously -- there's nothing in it with the wrong
    shape."""
    return isinstance(value, list) and all(
        isinstance(item, tuple)
        and len(item) == 3
        and isinstance(item[1], Trust)
        and isinstance(item[2], Secrecy)
        for item in value
    )


def _is_tagged_dict(value: Any) -> bool:
    """True if value is already shaped like a dict this interpreter
    builds: a dict whose every value is a
    (value, value_trust, value_secrecy, key_trust, key_secrecy) 5-tuple.
    An empty dict counts as tagged, vacuously.

    The key itself is deliberately stored raw, not wrapped -- Python
    dict lookup requires the stored key to equal whatever a caller
    looks it up by (`d["x"]`), and a tagged tuple would never equal the
    bare string being looked up. The key's own trust/secrecy rides
    alongside the value in the same 5-tuple instead, so `for key in
    some_dict:` can give each key its own precise tag (see ast.For)
    without breaking lookup by raw key."""
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
    """Implements `a in b` for this interpreter's own value shapes. A
    tagged list's elements are (value, Trust, Secrecy) triples, not bare
    values -- Python's own `in` would compare `a` against those triples
    directly and always come back False, so a list's elements are
    unwrapped to their raw values first. A tagged dict's keys are already
    stored raw (only its values are triples, see ast.Dict's handling
    below), and strings are never wrapped internally at all, so both fall
    straight through to plain Python `in` unchanged."""
    if _is_tagged_list(b):
        return any(a == v for v, _, _ in b)
    return a in b


def unwrap_value(value: Any) -> Any:
    """Recursively strips Trust/Secrecy tags from a tagged list or dict,
    down to plain Python values. Found necessary, not assumed, while
    wiring up real external functions (AgentDojo's real tools) that take
    list arguments (e.g. `restaurant_names: list[str]`): a list literal
    passed as a call argument was being handed to the callee with its
    elements still as raw (value, Trust, Secrecy) triples, since the
    ast.Call case only ever unwrapped the outermost triple, not what a
    list/dict value nested inside it. The same bug existed at run()'s and
    run_turn()'s own top-level return -- both promise an unwrapped value
    to the caller but only unwrapped one layer. Calling an external
    function, and returning a value out of run()/run_turn(), are both
    real boundaries where the interpreter's internal bookkeeping must
    never leak past, the same principle already applied to run()'s
    return -- this closes the gap where it wasn't actually applied for
    list/dict values specifically."""
    if _is_tagged_list(value):
        return [unwrap_value(v) for v, _, _ in value]
    if _is_tagged_dict(value):
        return {k: unwrap_value(v) for k, (v, _, _, _, _) in value.items()}
    return value


def _has_untrusted(value: Any, trust: Trust) -> bool:
    """True if trust itself is UNTRUSTED, or value is a tagged list or
    dict containing an untrusted element anywhere, recursively. A
    container's own outer trust tag can be overridden by a source or
    sanitizer independent of what's nested inside it -- a sanitizer that
    merely passes a tagged container through leaves its inner tags
    untouched while its outer tag flips to TRUSTED. Every place that
    gates on an argument's trust checks this instead of the bare outer
    tag, so a container can't be laundered by clearing only its own
    top-level label."""
    if trust == Trust.UNTRUSTED:
        return True
    if isinstance(value, list) and _is_tagged_list(value):
        return any(_has_untrusted(v, t) for v, t, _ in value)
    if isinstance(value, dict) and _is_tagged_dict(value):
        # Keys can be individually untrusted too, now that each entry
        # carries its own key_trust separately from the value's trust
        # -- checked directly (kt == Trust.UNTRUSTED), not recursively,
        # since a dict key must be hashable and therefore can never
        # itself be a nested tagged list/dict.
        return any(
            _has_untrusted(v, vt) or kt == Trust.UNTRUSTED
            for v, vt, _, kt, _ in value.values()
        )
    return False


def _has_secret(value: Any, secrecy: Secrecy) -> bool:
    """The confidentiality counterpart to _has_untrusted: true if secrecy
    itself is SECRET, or value is a tagged list or dict containing a
    secret element anywhere, recursively. Same reasoning as
    _has_untrusted -- a declassifier clearing a container's outer tag
    doesn't clear what's nested inside it."""
    if secrecy == Secrecy.SECRET:
        return True
    if isinstance(value, list) and _is_tagged_list(value):
        return any(_has_secret(v, s) for v, _, s in value)
    if isinstance(value, dict) and _is_tagged_dict(value):
        # Same reasoning as _has_untrusted: a key's own secrecy is
        # checked directly, not recursively -- a dict key can't itself
        # be a nested tagged container, since it must be hashable.
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
) -> tuple[Any, Trust, Secrecy]:
    """Evaluates a single expression node: a call, a constant, a variable
    read, a comparison, a list literal, a dict literal, or a subscript.
    Each case is explicit; anything else raises rather than falling back
    to a general evaluator. Returns (value, trust, secrecy), not a bare
    value -- every expression in this language carries both tags
    alongside its value. pc_trust and pc_secrecy are the trust and
    secrecy of the control flow that led here (see the module docstring's
    implicit-flow section) -- an UNTRUSTED pc_trust means this code is
    running inside a branch an untrusted value controlled, and a
    privileged call made under it is refused regardless of its own
    arguments; a SECRET pc_secrecy does the same for sink calls."""
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise InterpreterError("calls must be a plain function name")
        name = node.func.id
        if name not in allowed:
            raise InterpreterError(f"unknown or disallowed name: {name}")
        arg_results = [
            eval_node(a, allowed, env, caps, pc_trust, pc_secrecy) for a in node.args
        ]
        kwarg_results = {
            kw.arg: eval_node(kw.value, allowed, env, caps, pc_trust, pc_secrecy)
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
        # A real external function only ever sees and returns plain
        # values now, never this interpreter's internal tags -- but if
        # it hands back the exact same list/dict object it was given (a
        # passthrough, e.g. `def identity_sanitizer(x): return x`), the
        # original per-element tags are restored below rather than
        # falling through to auto-wrap, which would uniformly relabel
        # every nested element with the call's own outer trust and
        # silently launder whatever was actually still untrusted/secret
        # inside it -- the exact container-laundering shape already
        # found and fixed once, reachable again through a different
        # path (a real external function merely passing a container
        # through unchanged) once callees stopped seeing internal tags.
        # Detected by object identity, scoped to this one call only, not
        # a persistent tracking scheme -- so it doesn't share
        # provenance-ac's fragility against re-serialization. Narrow by
        # design: a function that mutates the container in place instead
        # of returning it untouched isn't covered, the same "opt out by
        # returning already-tagged pairs yourself" escape hatch other
        # imprecise auto-wrap cases already document applies here too.
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
            # Uniform, not precise, same principle as list auto-wrap:
            # a plain dict from a real function has no per-key/per-value
            # trust information of its own, so both the key and the
            # value in every entry get the call's own result_trust/
            # result_secrecy -- the interpreter never watched this dict
            # get built, so it has nothing finer-grained to give it.
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
        prev, prev_trust, prev_secrecy = eval_node(node.left, allowed, env, caps, pc_trust, pc_secrecy)
        result_trust = Trust.UNTRUSTED if prev_trust == Trust.UNTRUSTED else result_trust
        result_secrecy = Secrecy.SECRET if prev_secrecy == Secrecy.SECRET else result_secrecy
        for op_node, comparator_node in zip(node.ops, node.comparators):
            curr, curr_trust, curr_secrecy = eval_node(
                comparator_node, allowed, env, caps, pc_trust, pc_secrecy
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
                value_node, allowed, env, caps, pc_trust, pc_secrecy
            )
            result_trust = Trust.UNTRUSTED if value_trust == Trust.UNTRUSTED else result_trust
            result_secrecy = Secrecy.SECRET if value_secrecy == Secrecy.SECRET else result_secrecy
            if op_type is ast.And and not result:
                break
            if op_type is ast.Or and result:
                break
        return result, result_trust, result_secrecy
    if isinstance(node, ast.IfExp):
        # Ternary (`body if test else orelse`) -- the expression-level
        # twin of ast.If's own implicit-flow protection, deliberately
        # not skipped: a privileged/sink call could otherwise hide
        # inside a branch, gated by an untrusted test, invisible to any
        # check that only inspects arguments -- exactly the class of
        # bug pc_trust/pc_secrecy already exist to close for the
        # statement form (`send_to_attacker()` running with zero
        # errors purely because untrusted content picked the branch).
        # ast.IfExp is a different AST node from ast.If and was never
        # covered by that protection just by ast.If existing -- this
        # mirrors its branch_pc_trust/branch_pc_secrecy elevation
        # exactly, applied to whichever single branch actually gets
        # evaluated (lazily, matching real Python ternary semantics and
        # this interpreter's existing BoolOp short-circuit behavior --
        # the unchosen branch's side effects, including any call in
        # it, must never run).
        #
        # The chosen branch's own returned trust/secrecy is NOT
        # additionally forced untrusted/secret by the test's own tag --
        # consistent with ast.If's own documented choice not to
        # retroactively taint a plain assignment made inside a branch.
        # Picking between two already-safe values based on an untrusted
        # condition doesn't itself leak anything; what has to be
        # stopped is a privileged/sink call reachable only through that
        # choice, which branch_pc_trust/branch_pc_secrecy already
        # covers.
        test_value, test_trust, test_secrecy = eval_node(
            node.test, allowed, env, caps, pc_trust, pc_secrecy
        )
        branch_pc_trust = Trust.UNTRUSTED if pc_trust == Trust.UNTRUSTED or test_trust == Trust.UNTRUSTED else Trust.TRUSTED
        branch_pc_secrecy = Secrecy.SECRET if pc_secrecy == Secrecy.SECRET or test_secrecy == Secrecy.SECRET else Secrecy.PUBLIC
        chosen = node.body if test_value else node.orelse
        return eval_node(chosen, allowed, env, caps, branch_pc_trust, branch_pc_secrecy)
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise InterpreterError(f"unsupported unary operator: {op_type.__name__}")
        operand, operand_trust, operand_secrecy = eval_node(
            node.operand, allowed, env, caps, pc_trust, pc_secrecy
        )
        return _UNARY_OPS[op_type](operand), operand_trust, operand_secrecy
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BINOP_OPS:
            raise InterpreterError(f"unsupported arithmetic operator: {op_type.__name__}")
        left, left_trust, left_secrecy = eval_node(node.left, allowed, env, caps, pc_trust, pc_secrecy)
        right, right_trust, right_secrecy = eval_node(node.right, allowed, env, caps, pc_trust, pc_secrecy)
        if op_type is ast.Pow and isinstance(right, (int, float)) and abs(right) > MAX_EXPONENT:
            raise InterpreterError(f"exponent magnitude exceeds {MAX_EXPONENT}")
        binop_trust = Trust.UNTRUSTED if Trust.UNTRUSTED in (left_trust, right_trust) else Trust.TRUSTED
        binop_secrecy = Secrecy.SECRET if Secrecy.SECRET in (left_secrecy, right_secrecy) else Secrecy.PUBLIC
        return _BINOP_OPS[op_type](left, right), binop_trust, binop_secrecy
    if isinstance(node, ast.List):
        elements = [eval_node(e, allowed, env, caps, pc_trust, pc_secrecy) for e in node.elts]
        list_trust = Trust.UNTRUSTED if any(t == Trust.UNTRUSTED for _, t, _ in elements) else Trust.TRUSTED
        list_secrecy = Secrecy.SECRET if any(s == Secrecy.SECRET for _, _, s in elements) else Secrecy.PUBLIC
        return elements, list_trust, list_secrecy
    if isinstance(node, ast.Dict):
        if any(k is None for k in node.keys):
            raise InterpreterError("dict unpacking (**) is not supported")
        entries = []
        for key_node, value_node in zip(node.keys, node.values):
            key, key_trust, key_secrecy = eval_node(key_node, allowed, env, caps, pc_trust, pc_secrecy)
            value_result = eval_node(value_node, allowed, env, caps, pc_trust, pc_secrecy)
            entries.append((key, key_trust, key_secrecy, value_result))
        # Each entry stores (value, value_trust, value_secrecy,
        # key_trust, key_secrecy) -- the key itself stays the bare raw
        # value as the actual Python dict key (lookup by a plain key
        # has to keep working), with its own trust/secrecy carried
        # alongside rather than folded into the key. See
        # _is_tagged_dict's docstring for why a tagged key can't be the
        # stored key itself.
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
        container, _, _ = eval_node(node.value, allowed, env, caps, pc_trust, pc_secrecy)
        index, _, _ = eval_node(node.slice, allowed, env, caps, pc_trust, pc_secrecy)
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
            # entry is the 5-tuple (value, value_trust, value_secrecy,
            # key_trust, key_secrecy) -- indexing produces the value,
            # not the key, so only the first three fields are the
            # actual expression result here; the key's own tag matters
            # for iteration (ast.For), not for a lookup by that key.
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
        # Reading a named field off a real returned object (e.g. an
        # AgentDojo Transaction's .amount) -- found necessary live, not
        # anticipated: a plain returned object is neither a tagged list
        # nor a tagged dict, so ast.Subscript already correctly refuses
        # it, and until now there was no other way to read a field off
        # it at all. Trust/secrecy propagate straight through unchanged
        # from the base object -- reading a field doesn't establish any
        # new information, the same principle subscripting already
        # follows for a tagged container's elements.
        #
        # Security note, checked before shipping this, not assumed:
        # this can never become a call-through-attribute-chain escape.
        # ast.Call only ever dispatches by looking up a *literal*
        # whitelisted name in `allowed` (`node.func` must be a plain
        # ast.Name, and the lookup is always against `allowed`, never
        # against `env`) -- and ast.Name as a value expression only
        # ever reads from `env`, never from `allowed`. Whitelisted
        # callables are never reachable as values at all, so there is
        # no way for a value reached via attribute access, however
        # deeply chained, to ever end up called. Dunder/private names
        # are still blocked below anyway, as defense in depth against
        # information exposure (__dict__, __class__, and similar), and
        # because a property or __getattr__ override on a real returned
        # object could in principle run arbitrary code as a side effect
        # of a plain attribute read -- a real, currently unmitigated
        # risk for any object a whitelisted function might return, not
        # just for dunder names, but one this project's own tools
        # (plain data classes and pydantic models) don't actually
        # exercise today.
        obj, trust, secrecy = eval_node(node.value, allowed, env, caps, pc_trust, pc_secrecy)
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
) -> tuple[Any, Trust, Secrecy] | None:
    """Executes a single statement node: an assignment, a conditional, a
    while loop, a for loop, or an expression statement. Returns
    (value, trust, secrecy) for an expr_stmt, None otherwise. Unsupported
    statement types (imports, def, class, ...) raise rather than being
    silently skipped. pc_trust and pc_secrecy are the trust and secrecy
    of the control flow that led here (see the module docstring's
    implicit-flow section); budget is the shared iteration counter for
    the whole run() call, or None if the caller doesn't want the global
    cap enforced."""
    if isinstance(node, ast.Assign):
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            raise InterpreterError("assignment must have a single plain-name target")
        env[node.targets[0].id] = eval_node(node.value, allowed, env, caps, pc_trust, pc_secrecy)
        return None
    if isinstance(node, ast.If):
        test_value, test_trust, test_secrecy = eval_node(
            node.test, allowed, env, caps, pc_trust, pc_secrecy
        )
        branch_pc_trust = Trust.UNTRUSTED if pc_trust == Trust.UNTRUSTED or test_trust == Trust.UNTRUSTED else Trust.TRUSTED
        branch_pc_secrecy = Secrecy.SECRET if pc_secrecy == Secrecy.SECRET or test_secrecy == Secrecy.SECRET else Secrecy.PUBLIC
        branch = node.body if test_value else node.orelse
        result = None
        for stmt in branch:
            result = exec_stmt(stmt, allowed, env, caps, branch_pc_trust, branch_pc_secrecy, budget)
        return result
    if isinstance(node, ast.While):
        if node.orelse:
            raise InterpreterError("while/else is not supported")
        result = None
        iterations = 0
        test_value, test_trust, test_secrecy = eval_node(
            node.test, allowed, env, caps, pc_trust, pc_secrecy
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
                result = exec_stmt(stmt, allowed, env, caps, body_pc_trust, body_pc_secrecy, budget)
            test_value, test_trust, test_secrecy = eval_node(
                node.test, allowed, env, caps, pc_trust, pc_secrecy
            )
        return result
    if isinstance(node, ast.For):
        if node.orelse:
            raise InterpreterError("for/else is not supported")
        if not isinstance(node.target, ast.Name):
            raise InterpreterError("for loop target must be a single plain name")
        iterable, iterable_trust, iterable_secrecy = eval_node(
            node.iter, allowed, env, caps, pc_trust, pc_secrecy
        )
        if _is_tagged_list(iterable):
            tagged = iterable
        elif _is_tagged_dict(iterable):
            # Iterating a dict walks its keys, matching real Python's
            # `for x in some_dict:` -- each key gets its own precise
            # trust/secrecy (key_trust, key_secrecy), stored alongside
            # its value in the 5-tuple entry, not the dict's one
            # aggregate tag. This is the whole point of storing keys
            # this way (see _is_tagged_dict's docstring): a key from a
            # dict with one untrusted entry doesn't drag every other
            # key down with it.
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
                result = exec_stmt(stmt, allowed, env, caps, body_pc_trust, body_pc_secrecy, budget)
        return result
    if isinstance(node, ast.Expr):
        return eval_node(node.value, allowed, env, caps, pc_trust, pc_secrecy)
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
    """Parses source with ast.parse in exec mode (so multiple statements
    and assignment/if are syntactically available) and executes each
    top-level statement through exec_stmt. Never calls eval() on the
    source itself. Returns the bare value of the last expression
    statement (the Trust and Secrecy tags are unwrapped here, not exposed
    to callers), or None if the program ended on an assignment or empty
    branch. A malformed program raises InterpreterError; an UNTRUSTED
    value reaching a name in `privileged`, either as an argument or by
    controlling which branch it's called from, raises CapabilityError; a
    SECRET value reaching a name in `sinks` the same two ways raises
    ConfidentialityError. A fresh _IterationBudget is created for this
    call and shared across every loop the program runs, bounding total
    iterations across the whole program, not just within any single
    loop."""
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise InterpreterError(f"could not parse: {exc}") from exc
    if env is None:
        env = {}
    caps = _Capabilities(sources, privileged, sanitizers, confidential, sinks, declassifiers)
    budget = _IterationBudget(MAX_TOTAL_ITERATIONS)
    result = None
    for stmt in tree.body:
        result = exec_stmt(stmt, allowed, env, caps, Trust.TRUSTED, Secrecy.PUBLIC, budget)
    return unwrap_value(result[0]) if result is not None else None


class Session:
    """Holds the state that persists across multiple turns of
    incremental execution: variable bindings and the total iteration
    budget for one task. run() executes a whole program blind -- the
    model writes every statement before any of them run, never seeing a
    real result along the way. run_turn() is the alternative: one
    statement at a time, against a Session's accumulated state, so a
    driving harness can show the real result back to the model before
    asking for the next statement. Two Sessions are always independent;
    nothing here is process-global."""

    def __init__(self, budget_limit: int = MAX_TOTAL_ITERATIONS):
        self.env: dict[str, tuple[Any, Trust, Secrecy]] = {}
        self.budget = _IterationBudget(budget_limit)


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
    """Parses source as exactly one top-level statement and executes it
    against session's accumulated env and budget, returning the bare
    value the same way run() does. Raises InterpreterError if source
    contains anything other than exactly one statement -- a turn is one
    statement, not a program, and that's the entire difference from
    run(). pc_trust and pc_secrecy start fresh (TRUSTED/PUBLIC) for the
    statement, the same way every top-level statement inside a single
    run() call already does -- run_turn() doesn't change that behavior,
    it just lets each top-level statement arrive in its own call instead
    of all together in one source string. Capability rules
    (sources/privileged/etc.) are passed per call, same as run(), so a
    driving harness can keep them fixed across a task's turns or change
    them if it has a reason to; the budget and env are what's shared."""
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
        tree.body[0], allowed, session.env, caps, Trust.TRUSTED, Secrecy.PUBLIC, session.budget
    )
    return unwrap_value(result[0]) if result is not None else None
