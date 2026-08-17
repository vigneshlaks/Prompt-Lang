"""Feasibility test: can an open-weight model reliably produce valid
programs in prompt-lang's toy grammar? Meant to run before investing
further in the capability system. Logs three things per attempt, not
one: did it parse (valid Python syntax), did it run (passed the
interpreter's whitelist and executed without InterpreterError), and was
it correct (the right whitelisted functions got called, in the right
branch).

Requires Ollama running locally (`ollama serve`, usually automatic after
`brew install ollama`) with at least one model pulled, e.g.:

    ollama pull llama3.1:8b

Usage (from the repo root):
    python3 experiments/feasibility_test.py
    python3 experiments/feasibility_test.py --models llama3.1:8b qwen2.5:7b --reps 5
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prompt_lang.interpreter import InterpreterError, run

OLLAMA_URL = "http://localhost:11434/api/generate"

GRAMMAR = """\
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
"""

FEW_SHOT = """\
Example task: "If the score is at least 50, call pass_test(). Otherwise \
call fail_test()."
Example program:
if get_score() >= 50:
    pass_test()
else:
    fail_test()

Example task: "Get a value with get_value(). While it is less than 3, \
reassign it to the result of bump(value). Then call done(value)."
Example program:
x = get_value()
while x < 3:
    x = bump(x)
done(x)
"""


@dataclass
class Task:
    name: str
    description: str
    functions: list[str]
    make_allowed: Callable[[list[tuple[str, tuple, dict]]], dict[str, Callable]]
    check: Callable[[list[tuple[str, tuple, dict]]], bool]


def _logging_stub(name: str, calls: list, return_value: Any = None) -> Callable:
    def stub(*args, **kwargs):
        calls.append((name, args, kwargs))
        return return_value

    stub.__name__ = name
    return stub


def _balance_task() -> Task:
    def make_allowed(calls):
        return {
            "get_balance": _logging_stub("get_balance", calls, return_value=150),
            "approve": _logging_stub("approve", calls),
            "deny": _logging_stub("deny", calls),
        }

    def check(calls):
        names = [c[0] for c in calls]
        return names == ["get_balance", "approve"]

    return Task(
        name="balance_approval",
        description=(
            "Check the account balance with get_balance(). If it is at "
            "least 100, call approve(). Otherwise call deny()."
        ),
        functions=["get_balance", "approve", "deny"],
        make_allowed=make_allowed,
        check=check,
    )


def _admin_task() -> Task:
    def make_allowed(calls):
        return {
            "is_admin": _logging_stub("is_admin", calls, return_value=False),
            "grant": _logging_stub("grant", calls),
            "reject": _logging_stub("reject", calls),
        }

    def check(calls):
        names = [c[0] for c in calls]
        return names == ["is_admin", "reject"]

    return Task(
        name="admin_gate",
        description=(
            "Check is_admin(). If it is true, call grant(). Otherwise "
            "call reject()."
        ),
        functions=["is_admin", "grant", "reject"],
        make_allowed=make_allowed,
        check=check,
    )


def _chain_task() -> Task:
    def make_allowed(calls):
        return {
            "get_input": _logging_stub("get_input", calls, return_value=5),
            "add_one": lambda x: x + 1,
            "report": _logging_stub("report", calls),
        }

    def check(calls):
        report_calls = [c for c in calls if c[0] == "report"]
        return len(report_calls) == 1 and report_calls[0][1] == (6,)

    return Task(
        name="assign_chain",
        description=(
            "Get a value with get_input(), store it in a variable, add "
            "one to it with add_one(), and pass the result to report()."
        ),
        functions=["get_input", "add_one", "report"],
        make_allowed=make_allowed,
        check=check,
    )


def _score_equality_task() -> Task:
    def make_allowed(calls):
        return {
            "get_score": _logging_stub("get_score", calls, return_value=100),
            "celebrate": _logging_stub("celebrate", calls),
            "retry": _logging_stub("retry", calls),
        }

    def check(calls):
        names = [c[0] for c in calls]
        return names == ["get_score", "celebrate"]

    return Task(
        name="score_equality",
        description=(
            "Check get_score(). If it equals 100, call celebrate(). "
            "Otherwise call retry()."
        ),
        functions=["get_score", "celebrate", "retry"],
        make_allowed=make_allowed,
        check=check,
    )


def _branching_compute_task() -> Task:
    def make_allowed(calls):
        return {
            "get_x": _logging_stub("get_x", calls, return_value=20),
            "compute_large": lambda x: x * 10,
            "compute_small": lambda x: x,
            "finish": _logging_stub("finish", calls),
        }

    def check(calls):
        finish_calls = [c for c in calls if c[0] == "finish"]
        return len(finish_calls) == 1 and finish_calls[0][1] == (200,)

    return Task(
        name="branching_compute",
        description=(
            "Get x with get_x(). If x is greater than 10, store the "
            "result of compute_large(x) in a variable. Otherwise store "
            "the result of compute_small(x). Then pass that variable to "
            "finish()."
        ),
        functions=["get_x", "compute_large", "compute_small", "finish"],
        make_allowed=make_allowed,
        check=check,
    )


TASKS = [
    _balance_task(),
    _admin_task(),
    _chain_task(),
    _score_equality_task(),
    _branching_compute_task(),
]


def build_prompt(task: Task) -> str:
    return f"""\
You write programs in a tiny constrained language, not Python. Only this \
grammar is valid:

{GRAMMAR}
The only functions available for this task are: {", ".join(task.functions)}.
Do not use any other functions, operators, or statements (no loops, no \
imports, no arithmetic operators, no attribute access).

{FEW_SHOT}
Task: {task.description}

Respond with only the program, no explanation, no markdown fences.
"""


def extract_code(text: str) -> str:
    fence = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return text.strip()


def call_model(model: str, prompt: str, timeout: float = 60.0) -> str:
    resp = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def attempt(model: str, task: Task) -> dict:
    prompt = build_prompt(task)
    raw = call_model(model, prompt)
    source = extract_code(raw)

    parsed = True
    try:
        ast.parse(source, mode="exec")
    except SyntaxError:
        parsed = False

    calls: list[tuple[str, tuple, dict]] = []
    allowed = task.make_allowed(calls)
    ran = False
    error = None
    try:
        run(source, allowed)
        ran = True
    except InterpreterError as exc:
        error = f"whitelist boundary: {exc}"
    except Exception as exc:
        # A whitelisted call with the wrong arity/type (e.g. approve(True)
        # when approve takes no args) raises a normal Python exception,
        # not InterpreterError -- that's legal-per-whitelist but still a
        # failed attempt, distinct from hitting the grammar boundary.
        error = f"runtime error in whitelisted call: {exc!r}"

    correct = ran and task.check(calls)

    return {
        "model": model,
        "task": task.name,
        "parsed": parsed,
        "ran": ran,
        "correct": correct,
        "error": error,
        "raw_response": raw,
        "extracted_source": source,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["llama3.1:8b", "qwen2.5:7b", "mistral:7b"],
    )
    parser.add_argument("--reps", type=int, default=5)
    default_out = Path(__file__).resolve().parent / "feasibility_results.jsonl"
    parser.add_argument("--out", default=str(default_out), help="output log path")
    args = parser.parse_args()

    out_path = Path(args.out)
    results = []
    with out_path.open("w") as f:
        for model in args.models:
            for task in TASKS:
                for rep in range(args.reps):
                    try:
                        result = attempt(model, task)
                    except requests.RequestException as exc:
                        result = {
                            "model": model,
                            "task": task.name,
                            "parsed": False,
                            "ran": False,
                            "correct": False,
                            "error": f"request failed: {exc}",
                            "raw_response": "",
                            "extracted_source": "",
                        }
                    result["rep"] = rep
                    results.append(result)
                    f.write(json.dumps(result) + "\n")
                    f.flush()
                    tag = "OK" if result["correct"] else ("ran" if result["ran"] else ("parsed" if result["parsed"] else "FAIL"))
                    print(f"{model:16s} {task.name:20s} rep={rep} {tag}")

    print("\n--- summary ---")
    for model in args.models:
        model_results = [r for r in results if r["model"] == model]
        n = len(model_results)
        if n == 0:
            continue
        parsed_rate = sum(r["parsed"] for r in model_results) / n
        ran_rate = sum(r["ran"] for r in model_results) / n
        correct_rate = sum(r["correct"] for r in model_results) / n
        print(
            f"{model:16s} parsed={parsed_rate:.0%} ran={ran_rate:.0%} "
            f"correct={correct_rate:.0%} (n={n})"
        )
    print(f"\nfull log: {out_path}")


if __name__ == "__main__":
    main()
