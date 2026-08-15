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
    statement  := assign | if_stmt | while_stmt | expr_stmt
    assign     := NAME "=" expr
    if_stmt    := "if" expr ":" NEWLINE INDENT statement+ DEDENT
                  ("else" ":" NEWLINE INDENT statement+ DEDENT)?
    while_stmt := "while" expr ":" NEWLINE INDENT statement+ DEDENT
    expr_stmt  := expr
    expr       := call | compare | NAME | literal
    call       := NAME "(" (expr ("," expr)*)? ")"
    compare    := expr ("==" | "!=" | "<" | "<=" | ">" | ">=") expr
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

while loops are bounded: a loop that runs past MAX_WHILE_ITERATIONS raises
InterpreterError rather than running forever. An agent-executed language
should not be able to hang the process it runs in just because a model
wrote a condition that never goes false.
"""

from __future__ import annotations

import ast
import operator
from enum import Enum
from typing import Any, Callable

MAX_WHILE_ITERATIONS = 1000

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
    """Raised when an UNTRUSTED value reaches a privileged operation. A
    subclass of InterpreterError so existing callers that catch the
    whitelist-boundary error also catch this, while code that cares can
    still distinguish the two."""


def eval_node(
    node: ast.AST,
    allowed: dict[str, Callable],
    env: dict[str, tuple[Any, Trust]],
    sources: frozenset[str] = frozenset(),
    privileged: frozenset[str] = frozenset(),
    sanitizers: frozenset[str] = frozenset(),
) -> tuple[Any, Trust]:
    """Evaluates a single expression node: a call, a constant, a variable
    read, or a single comparison. Each case is explicit; anything else
    raises rather than falling back to a general evaluator. Returns
    (value, trust), not a bare value -- every expression in this
    language carries a Trust tag alongside its value."""
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise InterpreterError("calls must be a plain function name")
        name = node.func.id
        if name not in allowed:
            raise InterpreterError(f"unknown or disallowed name: {name}")
        arg_results = [
            eval_node(a, allowed, env, sources, privileged, sanitizers) for a in node.args
        ]
        kwarg_results = {
            kw.arg: eval_node(kw.value, allowed, env, sources, privileged, sanitizers)
            for kw in node.keywords
        }
        arg_trusts = [t for _, t in arg_results] + [t for _, t in kwarg_results.values()]
        if name in privileged and Trust.UNTRUSTED in arg_trusts:
            raise CapabilityError(
                f"privileged operation {name!r} called with an untrusted argument"
            )
        args = [v for v, _ in arg_results]
        kwargs = {k: v for k, (v, _) in kwarg_results.items()}
        result = allowed[name](*args, **kwargs)
        if name in sources:
            result_trust = Trust.UNTRUSTED
        elif name in sanitizers:
            result_trust = Trust.TRUSTED
        elif Trust.UNTRUSTED in arg_trusts:
            result_trust = Trust.UNTRUSTED
        else:
            result_trust = Trust.TRUSTED
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
        left, _ = eval_node(node.left, allowed, env, sources, privileged, sanitizers)
        right, _ = eval_node(node.comparators[0], allowed, env, sources, privileged, sanitizers)
        return _COMPARE_OPS[op_type](left, right), Trust.TRUSTED
    raise InterpreterError(f"unsupported expression: {ast.dump(node)}")


def exec_stmt(
    node: ast.stmt,
    allowed: dict[str, Callable],
    env: dict[str, tuple[Any, Trust]],
    sources: frozenset[str] = frozenset(),
    privileged: frozenset[str] = frozenset(),
    sanitizers: frozenset[str] = frozenset(),
) -> tuple[Any, Trust] | None:
    """Executes a single statement node: an assignment, a conditional, a
    while loop, or an expression statement. Returns (value, trust) for an
    expr_stmt, None otherwise. Unsupported statement types (imports, def,
    for, class, ...) raise rather than being silently skipped."""
    if isinstance(node, ast.Assign):
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            raise InterpreterError("assignment must have a single plain-name target")
        env[node.targets[0].id] = eval_node(
            node.value, allowed, env, sources, privileged, sanitizers
        )
        return None
    if isinstance(node, ast.If):
        test_value, _ = eval_node(node.test, allowed, env, sources, privileged, sanitizers)
        branch = node.body if test_value else node.orelse
        result = None
        for stmt in branch:
            result = exec_stmt(stmt, allowed, env, sources, privileged, sanitizers)
        return result
    if isinstance(node, ast.While):
        if node.orelse:
            raise InterpreterError("while/else is not supported")
        result = None
        iterations = 0
        test_value, _ = eval_node(node.test, allowed, env, sources, privileged, sanitizers)
        while test_value:
            iterations += 1
            if iterations > MAX_WHILE_ITERATIONS:
                raise InterpreterError(
                    f"while loop exceeded {MAX_WHILE_ITERATIONS} iterations"
                )
            for stmt in node.body:
                result = exec_stmt(stmt, allowed, env, sources, privileged, sanitizers)
            test_value, _ = eval_node(node.test, allowed, env, sources, privileged, sanitizers)
        return result
    if isinstance(node, ast.Expr):
        return eval_node(node.value, allowed, env, sources, privileged, sanitizers)
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
    malformed program raises InterpreterError; an UNTRUSTED value reaching a
    name in `privileged` raises CapabilityError, a subclass of it."""
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise InterpreterError(f"could not parse: {exc}") from exc
    if env is None:
        env = {}
    result = None
    for stmt in tree.body:
        result = exec_stmt(stmt, allowed, env, sources, privileged, sanitizers)
    return result[0] if result is not None else None
