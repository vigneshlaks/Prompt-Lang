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
    expr       := call | compare | subscript | list_expr | NAME | literal
    call       := NAME "(" (expr ("," expr)*)? ")"
    compare    := expr ("==" | "!=" | "<" | "<=" | ">" | ">=") expr
    subscript  := expr "[" expr "]"
    list_expr  := "[" (expr ("," expr)*)? "]"
    dict_expr  := "{" (expr ":" expr ("," expr ":" expr)*)? "}"
    literal    := NUMBER | STRING | "True" | "False" | "None"

Two separate namespaces: `allowed` (whitelisted callables, checked at call
sites only) and `env` (variables bound by assignment, checked at name-read
sites only). A bare reference to a whitelisted function name is not a
variable read and is rejected, same as before this file supported
variables at all.

Capability tracking: every value this interpreter produces carries a
Trust tag (TRUSTED or UNTRUSTED) alongside it, threaded through eval_node
and exec_stmt rather than attached via object identity -- there's no
side table, because the interpreter owns every value's lifecycle itself.
`sources` names a subset of `allowed` whose return value is always
UNTRUSTED, regardless of its arguments. `privileged` names a subset of
`allowed` that refuses to run at all if any argument is UNTRUSTED, raising
CapabilityError before the underlying function is ever called. An
ordinary (non-source, non-privileged, non-sanitizer) function propagates
trust from its arguments: UNTRUSTED in, UNTRUSTED out, through any chain
of calls, since eval_node evaluates arguments recursively before
combining their trust. `sanitizers` names a subset of `allowed` whose
return value is always TRUSTED, regardless of its arguments -- the one
deliberate, explicitly declared way for a program to turn UNTRUSTED back
into TRUSTED. Only a name explicitly listed in `sanitizers` has this
effect; nothing about a function's behavior implies it. If a name is in
both `sources` and `sanitizers`, `sources` wins, since the untrusted
outcome is the safer one to prefer when the configuration itself is
contradictory. Confidentiality (the reverse direction: stopping sensitive
data from reaching an untrusted sink) is explicitly not handled yet.

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

Lists carry trust per element, not as one tag for the whole collection.
A list literal evaluates each element the normal way and stores the full
list of (value, Trust) pairs as its value; indexing and for loops both
read that structure directly, so a list mixing trusted and untrusted
items keeps each item's own tag instead of collapsing to one.

A plain Python list handed back by a function in `allowed` has no
per-element tags of its own -- the interpreter never watched it get
built. Rather than reject it, eval_node auto-wraps it: every element
gets the same trust the call as a whole already earned (from `sources`,
`sanitizers`, or argument propagation, the usual rules), turning a plain
list into the same (value, Trust) pair shape a list literal produces.
This is uniform, not precise -- every element in an auto-wrapped list
shares one trust value, since the interpreter has no finer-grained
information to give it. A function that needs real per-element
precision (a list of items with genuinely different trust levels) can
opt out of the uniform treatment by returning already-tagged pairs
itself; auto-wrap detects that shape and leaves it untouched.

A sanitizer clears the outer trust tag on whatever it returns, but if it
just passes a tagged list through unchanged, the tags on the individual
elements inside are untouched -- the outer tag and the nested tags can
disagree. _has_untrusted checks both, recursively, everywhere an
argument's trust is inspected (the privileged gate and ordinary
propagation), so a container can't be laundered by clearing only its own
top-level label while what's nested inside stays untrusted.

Dicts work the same way lists do, keeping the lesson from that finding
rather than relearning it: a dict literal evaluates each key and value
normally and stores {key: (value, Trust)} as its value, so a dict mixing
trusted and untrusted values keeps each one's own tag. Its own outer tag
is the join of every key's and value's trust. Subscripting a dict reads
that structure directly, the same as a list. A plain Python dict handed
back by a function in `allowed` gets auto-wrapped exactly like a plain
list does -- every value tagged uniformly with the call's own trust,
opt-out available by returning already-tagged pairs. _has_untrusted and
the tagged-shape checks cover both lists and dicts from the start, so
the sanitizer-laundering gap already fixed for lists doesn't reopen for
dicts as a second, separate bug to find later. Dicts do not yet support
iteration (`for k in some_dict`) -- only literal construction and
indexed lookup.

Everything above tracks explicit data flow: does an untrusted value reach
a privileged call as an argument. It does not, by itself, track implicit
flow: untrusted data can freely decide which branch of an if/while runs
without ever appearing as an argument, and if the privileged call in that
branch takes no untrusted argument (or no argument at all), the argument
check has nothing to see. A scoped, partial mitigation exists for this:
every eval_node/exec_stmt call threads a pc_trust value (the trust of the
control flow that led to this point in the program). Entering an if
branch, a while body, or a for body whose controlling condition or
iterable was UNTRUSTED raises pc_trust to UNTRUSTED for that block; a
privileged call made while pc_trust is UNTRUSTED is refused, regardless
of its own arguments. This is deliberately narrow -- it gates privileged
calls only, it does not retroactively taint every value computed under a
raised pc_trust the way a complete implicit-flow system would (assigning
to a variable that already exists, for instance, isn't currently
affected). Real implicit-flow tracking done properly tends to make a
system very restrictive, since almost anything computed inside a branch
on untrusted data ends up needing to be treated as untrusted too; most
practical taint trackers (Perl's taint mode among them) deliberately
don't attempt it for exactly that reason. This mitigation is a bounded
compromise, not a complete solution, and is scoped to the one case found
by deliberately testing it: a privileged call with no untrusted
arguments, reachable only through a branch an untrusted value controlled.
"""

from __future__ import annotations

import ast
import operator
from enum import Enum
from typing import Any, Callable

MAX_WHILE_ITERATIONS = 1000
MAX_TOTAL_ITERATIONS = 50000

_COMPARE_OPS: dict[type, Callable[[Any, Any], bool]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}


class Trust(Enum):
    TRUSTED = "trusted"
    UNTRUSTED = "untrusted"


class InterpreterError(Exception):
    pass


class CapabilityError(InterpreterError):
    """Raised when an UNTRUSTED value reaches a privileged operation,
    either directly as an argument or by controlling which branch a
    privileged call sits in. A subclass of InterpreterError so existing
    callers that catch the whitelist-boundary error also catch this,
    while code that cares can still distinguish the two."""


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
    builds: a list of (value, Trust) pairs. An empty list counts as
    tagged, vacuously -- there's nothing in it with the wrong shape."""
    return isinstance(value, list) and all(
        isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], Trust)
        for item in value
    )


def _is_tagged_dict(value: Any) -> bool:
    """True if value is already shaped like a dict this interpreter
    builds: a dict whose every value is a (value, Trust) pair. An empty
    dict counts as tagged, vacuously."""
    return isinstance(value, dict) and all(
        isinstance(v, tuple) and len(v) == 2 and isinstance(v[1], Trust)
        for v in value.values()
    )


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
        return any(_has_untrusted(v, t) for v, t in value)
    if isinstance(value, dict) and _is_tagged_dict(value):
        return any(_has_untrusted(v, t) for v, t in value.values())
    return False


def _as_tagged_list(value: Any, context: str) -> list[tuple[Any, Trust]]:
    """Checks that value is shaped like a list this interpreter itself
    builds: a list of (value, Trust) pairs. Raises InterpreterError with a
    clear explanation rather than letting a plain Python list fail with a
    confusing unpacking error somewhere downstream. Calls in `allowed`
    that return a plain list get it auto-wrapped into this shape before
    it ever reaches here (see eval_node's ast.Call case), so this only
    rejects a list that was never given a trust shape at all."""
    if not _is_tagged_list(value):
        raise InterpreterError(
            f"{context} requires a list built with a list literal (or a "
            "function in `allowed` returning one) -- a plain list has no "
            "per-element trust to read"
        )
    return value


def eval_node(
    node: ast.AST,
    allowed: dict[str, Callable],
    env: dict[str, tuple[Any, Trust]],
    sources: frozenset[str] = frozenset(),
    privileged: frozenset[str] = frozenset(),
    sanitizers: frozenset[str] = frozenset(),
    pc_trust: Trust = Trust.TRUSTED,
) -> tuple[Any, Trust]:
    """Evaluates a single expression node: a call, a constant, a variable
    read, a comparison, a list literal, or a subscript. Each case is
    explicit; anything else raises rather than falling back to a general
    evaluator. Returns (value, trust), not a bare value -- every
    expression in this language carries a Trust tag alongside its value.
    pc_trust is the trust of the control flow that led here (see the
    module docstring's implicit-flow section) -- UNTRUSTED means this
    code is running inside a branch an untrusted value controlled, and a
    privileged call made under it is refused regardless of its own
    arguments."""
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise InterpreterError("calls must be a plain function name")
        name = node.func.id
        if name not in allowed:
            raise InterpreterError(f"unknown or disallowed name: {name}")
        arg_results = [
            eval_node(a, allowed, env, sources, privileged, sanitizers, pc_trust)
            for a in node.args
        ]
        kwarg_results = {
            kw.arg: eval_node(kw.value, allowed, env, sources, privileged, sanitizers, pc_trust)
            for kw in node.keywords
        }
        all_args = arg_results + list(kwarg_results.values())
        any_untrusted = any(_has_untrusted(v, t) for v, t in all_args)
        if name in privileged and any_untrusted:
            raise CapabilityError(
                f"privileged operation {name!r} called with an untrusted argument"
            )
        if name in privileged and pc_trust == Trust.UNTRUSTED:
            raise CapabilityError(
                f"privileged operation {name!r} called from a branch whose "
                "condition depended on untrusted data"
            )
        args = [v for v, _ in arg_results]
        kwargs = {k: v for k, (v, _) in kwarg_results.items()}
        result = allowed[name](*args, **kwargs)
        if name in sources:
            result_trust = Trust.UNTRUSTED
        elif name in sanitizers:
            result_trust = Trust.TRUSTED
        elif any_untrusted:
            result_trust = Trust.UNTRUSTED
        else:
            result_trust = Trust.TRUSTED
        if isinstance(result, list) and not _is_tagged_list(result):
            result = [(item, result_trust) for item in result]
        elif isinstance(result, dict) and not _is_tagged_dict(result):
            result = {k: (v, result_trust) for k, v in result.items()}
        return result, result_trust
    if isinstance(node, ast.Constant):
        return node.value, Trust.TRUSTED
    if isinstance(node, ast.Name):
        if not isinstance(node.ctx, ast.Load):
            raise InterpreterError(f"unsupported name usage: {ast.dump(node)}")
        if node.id not in env:
            raise InterpreterError(f"undefined variable: {node.id}")
        return env[node.id]
    if isinstance(node, ast.Compare):
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise InterpreterError("only single comparisons are supported, e.g. x == 1")
        op_type = type(node.ops[0])
        if op_type not in _COMPARE_OPS:
            raise InterpreterError(f"unsupported comparison operator: {op_type.__name__}")
        left, left_trust = eval_node(node.left, allowed, env, sources, privileged, sanitizers, pc_trust)
        right, right_trust = eval_node(
            node.comparators[0], allowed, env, sources, privileged, sanitizers, pc_trust
        )
        compare_trust = Trust.UNTRUSTED if Trust.UNTRUSTED in (left_trust, right_trust) else Trust.TRUSTED
        return _COMPARE_OPS[op_type](left, right), compare_trust
    if isinstance(node, ast.List):
        elements = [
            eval_node(e, allowed, env, sources, privileged, sanitizers, pc_trust) for e in node.elts
        ]
        list_trust = Trust.UNTRUSTED if any(t == Trust.UNTRUSTED for _, t in elements) else Trust.TRUSTED
        return elements, list_trust
    if isinstance(node, ast.Dict):
        if any(k is None for k in node.keys):
            raise InterpreterError("dict unpacking (**) is not supported")
        entries = []
        for key_node, value_node in zip(node.keys, node.values):
            key, key_trust = eval_node(key_node, allowed, env, sources, privileged, sanitizers, pc_trust)
            value_result = eval_node(value_node, allowed, env, sources, privileged, sanitizers, pc_trust)
            entries.append((key, key_trust, value_result))
        result_dict: dict[Any, tuple[Any, Trust]] = {}
        for key, _, value_result in entries:
            try:
                result_dict[key] = value_result
            except TypeError as exc:
                raise InterpreterError(f"unhashable dict key: {key!r}") from exc
        dict_trust = Trust.TRUSTED
        for _, key_trust, (_, value_trust) in entries:
            if key_trust == Trust.UNTRUSTED or value_trust == Trust.UNTRUSTED:
                dict_trust = Trust.UNTRUSTED
                break
        return result_dict, dict_trust
    if isinstance(node, ast.Subscript):
        if not isinstance(node.ctx, ast.Load):
            raise InterpreterError(f"unsupported subscript usage: {ast.dump(node)}")
        container, _ = eval_node(node.value, allowed, env, sources, privileged, sanitizers, pc_trust)
        index, _ = eval_node(node.slice, allowed, env, sources, privileged, sanitizers, pc_trust)
        if _is_tagged_list(container):
            if not isinstance(index, int) or isinstance(index, bool):
                raise InterpreterError("list index must be an integer")
            try:
                return container[index]
            except IndexError as exc:
                raise InterpreterError(f"list index out of range: {index}") from exc
        if _is_tagged_dict(container):
            try:
                return container[index]
            except KeyError as exc:
                raise InterpreterError(f"dict has no key: {index!r}") from exc
            except TypeError as exc:
                raise InterpreterError(f"unhashable dict key: {index!r}") from exc
        raise InterpreterError(
            "subscripting requires a list or dict built with a literal (or "
            "a function in `allowed` returning one) -- a plain value has "
            "no per-element trust to read"
        )
    raise InterpreterError(f"unsupported expression: {ast.dump(node)}")


def exec_stmt(
    node: ast.stmt,
    allowed: dict[str, Callable],
    env: dict[str, tuple[Any, Trust]],
    sources: frozenset[str] = frozenset(),
    privileged: frozenset[str] = frozenset(),
    sanitizers: frozenset[str] = frozenset(),
    pc_trust: Trust = Trust.TRUSTED,
    budget: _IterationBudget | None = None,
) -> tuple[Any, Trust] | None:
    """Executes a single statement node: an assignment, a conditional, a
    while loop, a for loop, or an expression statement. Returns
    (value, trust) for an expr_stmt, None otherwise. Unsupported
    statement types (imports, def, class, ...) raise rather than being
    silently skipped. pc_trust is the trust of the control flow that led
    here (see the module docstring's implicit-flow section); budget is
    the shared iteration counter for the whole run() call, or None if the
    caller doesn't want the global cap enforced."""
    if isinstance(node, ast.Assign):
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            raise InterpreterError("assignment must have a single plain-name target")
        env[node.targets[0].id] = eval_node(
            node.value, allowed, env, sources, privileged, sanitizers, pc_trust
        )
        return None
    if isinstance(node, ast.If):
        test_value, test_trust = eval_node(
            node.test, allowed, env, sources, privileged, sanitizers, pc_trust
        )
        branch_pc = Trust.UNTRUSTED if pc_trust == Trust.UNTRUSTED or test_trust == Trust.UNTRUSTED else Trust.TRUSTED
        branch = node.body if test_value else node.orelse
        result = None
        for stmt in branch:
            result = exec_stmt(stmt, allowed, env, sources, privileged, sanitizers, branch_pc, budget)
        return result
    if isinstance(node, ast.While):
        if node.orelse:
            raise InterpreterError("while/else is not supported")
        result = None
        iterations = 0
        test_value, test_trust = eval_node(
            node.test, allowed, env, sources, privileged, sanitizers, pc_trust
        )
        while test_value:
            iterations += 1
            if iterations > MAX_WHILE_ITERATIONS:
                raise InterpreterError(
                    f"while loop exceeded {MAX_WHILE_ITERATIONS} iterations"
                )
            if budget is not None:
                budget.consume()
            body_pc = Trust.UNTRUSTED if pc_trust == Trust.UNTRUSTED or test_trust == Trust.UNTRUSTED else Trust.TRUSTED
            for stmt in node.body:
                result = exec_stmt(stmt, allowed, env, sources, privileged, sanitizers, body_pc, budget)
            test_value, test_trust = eval_node(
                node.test, allowed, env, sources, privileged, sanitizers, pc_trust
            )
        return result
    if isinstance(node, ast.For):
        if node.orelse:
            raise InterpreterError("for/else is not supported")
        if not isinstance(node.target, ast.Name):
            raise InterpreterError("for loop target must be a single plain name")
        iterable, iterable_trust = eval_node(
            node.iter, allowed, env, sources, privileged, sanitizers, pc_trust
        )
        tagged = _as_tagged_list(iterable, "a for loop's iterable")
        body_pc = Trust.UNTRUSTED if pc_trust == Trust.UNTRUSTED or iterable_trust == Trust.UNTRUSTED else Trust.TRUSTED
        result = None
        for element in tagged:
            if budget is not None:
                budget.consume()
            env[node.target.id] = element
            for stmt in node.body:
                result = exec_stmt(stmt, allowed, env, sources, privileged, sanitizers, body_pc, budget)
        return result
    if isinstance(node, ast.Expr):
        return eval_node(node.value, allowed, env, sources, privileged, sanitizers, pc_trust)
    raise InterpreterError(f"unsupported statement: {ast.dump(node)}")


def run(
    source: str,
    allowed: dict[str, Callable],
    env: dict[str, tuple[Any, Trust]] | None = None,
    *,
    sources: frozenset[str] = frozenset(),
    privileged: frozenset[str] = frozenset(),
    sanitizers: frozenset[str] = frozenset(),
) -> Any:
    """Parses source with ast.parse in exec mode (so multiple statements
    and assignment/if are syntactically available) and executes each
    top-level statement through exec_stmt. Never calls eval() on the
    source itself. Returns the bare value of the last expression
    statement (the Trust tag is unwrapped here, not exposed to callers),
    or None if the program ended on an assignment or empty branch. A
    malformed program raises InterpreterError; an UNTRUSTED value reaching
    a name in `privileged`, either as an argument or by controlling which
    branch it's called from, raises CapabilityError, a subclass of it.
    A fresh _IterationBudget is created for this call and shared across
    every loop the program runs, bounding total iterations across the
    whole program, not just within any single loop."""
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise InterpreterError(f"could not parse: {exc}") from exc
    if env is None:
        env = {}
    budget = _IterationBudget(MAX_TOTAL_ITERATIONS)
    result = None
    for stmt in tree.body:
        result = exec_stmt(
            stmt, allowed, env, sources, privileged, sanitizers, Trust.TRUSTED, budget
        )
    return result[0] if result is not None else None
