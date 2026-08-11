"""A safe core for evaluating expressions a language model writes, without
ever calling eval(). This is the seed of a real interpreter, ported from
provenance-ac's agent_demo/agent_loop.py, where the same pattern parsed
and dispatched a single tool call safely. Here it's meant to grow into the
actual interpreter for a constrained language: more statement types
(assignment, conditionals, loops) get added by extending eval_node, not by
relaxing the whitelist check.

The rule that has to hold no matter how much this grows: parsing uses only
ast.parse, dispatch goes only through an explicit whitelist of names, and
eval() is never called on anything a model produced.
"""

from __future__ import annotations

import ast
from typing import Any, Callable


class InterpreterError(Exception):
    pass


def eval_node(node: ast.AST, allowed: dict[str, Callable]) -> Any:
    """Evaluates a single AST node against a whitelist of callables. Only
    a plain function call or a constant is supported right now. This is
    where new node types get added as the language grows: an ast.Assign
    branch for variable assignment, an ast.If branch for conditionals, and
    so on, each one explicit rather than falling back to a general
    evaluator."""
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise InterpreterError("calls must be a plain function name")
        name = node.func.id
        if name not in allowed:
            raise InterpreterError(f"unknown or disallowed name: {name}")
        args = [eval_node(a, allowed) for a in node.args]
        kwargs = {kw.arg: eval_node(kw.value, allowed) for kw in node.keywords}
        return allowed[name](*args, **kwargs)
    if isinstance(node, ast.Constant):
        return node.value
    raise InterpreterError(f"unsupported expression: {ast.dump(node)}")


def run(source: str, allowed: dict[str, Callable]) -> Any:
    """Parses one line of source with ast.parse, in eval mode, and
    evaluates it through eval_node. Never calls eval() on the source
    itself. A malformed program raises InterpreterError rather than
    executing anything."""
    try:
        tree = ast.parse(source, mode="eval").body
    except SyntaxError as exc:
        raise InterpreterError(f"could not parse: {exc}") from exc
    return eval_node(tree, allowed)
