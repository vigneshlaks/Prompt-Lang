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
    statement  := assign | if_stmt | expr_stmt
    assign     := NAME "=" expr
    if_stmt    := "if" expr ":" NEWLINE INDENT statement+ DEDENT
                  ("else" ":" NEWLINE INDENT statement+ DEDENT)?
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
"""

from __future__ import annotations

import ast
import operator
from typing import Any, Callable

_COMPARE_OPS: dict[type, Callable[[Any, Any], bool]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}


class InterpreterError(Exception):
    pass


def eval_node(node: ast.AST, allowed: dict[str, Callable], env: dict[str, Any]) -> Any:
    """Evaluates a single expression node: a call, a constant, a variable
    read, or a single comparison. Each case is explicit; anything else
    raises rather than falling back to a general evaluator."""
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise InterpreterError("calls must be a plain function name")
        name = node.func.id
        if name not in allowed:
            raise InterpreterError(f"unknown or disallowed name: {name}")
        args = [eval_node(a, allowed, env) for a in node.args]
        kwargs = {kw.arg: eval_node(kw.value, allowed, env) for kw in node.keywords}
        return allowed[name](*args, **kwargs)
    if isinstance(node, ast.Constant):
        return node.value
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
        left = eval_node(node.left, allowed, env)
        right = eval_node(node.comparators[0], allowed, env)
        return _COMPARE_OPS[op_type](left, right)
    raise InterpreterError(f"unsupported expression: {ast.dump(node)}")


def exec_stmt(node: ast.stmt, allowed: dict[str, Callable], env: dict[str, Any]) -> Any:
    """Executes a single statement node: an assignment, a conditional, or
    an expression statement. Returns the expression's value for an
    expr_stmt, None otherwise. Unsupported statement types (imports, def,
    for, while, class, ...) raise rather than being silently skipped."""
    if isinstance(node, ast.Assign):
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            raise InterpreterError("assignment must have a single plain-name target")
        env[node.targets[0].id] = eval_node(node.value, allowed, env)
        return None
    if isinstance(node, ast.If):
        branch = node.body if eval_node(node.test, allowed, env) else node.orelse
        result = None
        for stmt in branch:
            result = exec_stmt(stmt, allowed, env)
        return result
    if isinstance(node, ast.Expr):
        return eval_node(node.value, allowed, env)
    raise InterpreterError(f"unsupported statement: {ast.dump(node)}")


def run(source: str, allowed: dict[str, Callable], env: dict[str, Any] | None = None) -> Any:
    """Parses source with ast.parse in exec mode (so multiple statements
    and assignment/if are syntactically available) and executes each
    top-level statement through exec_stmt. Never calls eval() on the
    source itself. Returns the value of the last expression statement, or
    None if the program ended on an assignment or empty branch. A
    malformed program raises InterpreterError rather than executing
    anything."""
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise InterpreterError(f"could not parse: {exc}") from exc
    if env is None:
        env = {}
    result = None
    for stmt in tree.body:
        result = exec_stmt(stmt, allowed, env)
    return result
