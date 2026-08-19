"""Live verification of RetypingGuard (prompt_lang/defenses.py) against
a real model, reusing agentdojo_test.py's exact banking-suite scenario --
the same UserTask0/InjectionTask0 setup that originally found the
literal-retyping bug (notes/PRODUCTION_ROADMAP.md item 12). The guard's
own test suite (tests/test_defenses.py) only proves it catches a
hand-constructed statement shaped like the real finding; this is the
first time it runs against an actual model's real turn-by-turn output,
on both the injected scenario (does it catch a real retyped value) and
the baseline scenario (does it false-positive on a model correctly
using real data no attacker touched).

Detect-only, not blocking: the guard's check_statement() runs before
every turn and any flags get recorded, but the turn still executes
either way. Blocking would prevent observing what the guard would have
caught across a whole run; the point here is measuring, not enforcing.

Usage (from the repo root):
    python3 experiments/retyping_guard_live_test.py --models qwen2.5:3b --reps 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentdojo_test import (
    _JB_STRING,
    PRIVILEGED_TOOLS,
    SOURCE_TOOLS,
    build_prompt,
    call_model,
    extract_code,
    make_allowed,
    task_suite,
    tool_descriptions,
)
from turn_by_turn_test import _turn_display_result

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from prompt_lang.defenses import RetypingGuard, wrap_for_retyping_guard
from prompt_lang.interpreter import Session, run_turn

MAX_TURNS = 8


def run_attempt_guarded(model: str, injections: dict[str, str]) -> dict:
    pre_env = task_suite.load_and_inject_default_environment(injections)
    from agentdojo.functions_runtime import FunctionsRuntime
    runtime = FunctionsRuntime(task_suite.tools)
    env = pre_env.model_copy(deep=True)
    allowed = make_allowed(env, runtime)
    tools_text = tool_descriptions(runtime)

    guard = RetypingGuard()
    guarded_allowed = wrap_for_retyping_guard(
        allowed, sources=SOURCE_TOOLS, confidential=frozenset(), guard=guard
    )

    task = task_suite.get_user_task_by_id("user_task_0")
    session = Session()
    transcript: list[tuple[str, str]] = []
    guard_events: list[dict] = []
    model_output = ""

    for turn in range(MAX_TURNS):
        raw = call_model(model, build_prompt(task.PROMPT, tools_text, transcript), )
        stmt = extract_code(raw)
        if stmt.strip() == "DONE":
            break

        flags = guard.check_statement(stmt, privileged=PRIVILEGED_TOOLS, sinks=frozenset())
        if flags:
            guard_events.append({
                "turn": turn,
                "statement": stmt,
                "flags": [
                    {"call_name": f.call_name, "literal": f.literal, "matched_text": f.matched_text}
                    for f in flags
                ],
            })

        try:
            result = run_turn(
                session, stmt, guarded_allowed,
                sources=SOURCE_TOOLS, privileged=PRIVILEGED_TOOLS,
            )
            display = _turn_display_result(stmt, result, session)
            transcript.append((stmt, repr(display)))
            # Same fix as agentdojo_test.py/overhead_measurement.py: don't
            # drop a turn's real result from model_output just because it
            # isn't a string.
            if display is not None:
                model_output += str(display) + "\n"
        except Exception as exc:
            transcript.append((stmt, f"ERROR: {exc}"))

    utility = task.utility(model_output, pre_env, env, strict=False)
    injection_task = task_suite.get_injection_task_by_id("injection_task_0")
    security = injection_task.security(model_output, pre_env, env) if injections else None
    return {
        "model": model,
        "injected": bool(injections),
        "utility": utility,
        "security_breached": security,
        "guard_fired": bool(guard_events),
        "guard_events": guard_events,
        "transcript": transcript,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["qwen2.5:3b"])
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--host", default="http://localhost:11434")
    default_out = Path(__file__).resolve().parent / "results" / "retyping_guard_live_results.jsonl"
    parser.add_argument("--out", default=str(default_out))
    args = parser.parse_args()

    ollama_url = args.host.rstrip("/") + "/api/generate"
    import agentdojo_test
    agentdojo_test.OLLAMA_URL = ollama_url

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with out_path.open("w") as f:
        for model in args.models:
            for injected in (False, True):
                for rep in range(args.reps):
                    injections = {}
                    if injected:
                        goal_text = _JB_STRING.format(
                            user="Emma Johnson", model=model,
                            goal=task_suite.get_injection_task_by_id("injection_task_0").GOAL,
                        )
                        injections = {"injection_bill_text": "Car Rental\t\t98.70" + goal_text}
                    result = run_attempt_guarded(model, injections)
                    result["rep"] = rep
                    results.append(result)
                    f.write(json.dumps(result) + "\n")
                    f.flush()
                    tag = "injected" if injected else "baseline"
                    print(
                        f"{model:16s} {tag:9s} rep={rep} utility={result['utility']} "
                        f"security_breached={result['security_breached']} "
                        f"guard_fired={result['guard_fired']}"
                    )
                    if result["guard_fired"]:
                        for ev in result["guard_events"]:
                            for flag in ev["flags"]:
                                print(f"    turn {ev['turn']}: {flag['call_name']}() literal={flag['literal']!r}")

    print("\n--- summary ---")
    for model in args.models:
        for injected in (False, True):
            rows = [r for r in results if r["model"] == model and r["injected"] == injected]
            n = len(rows)
            if n == 0:
                continue
            fired = sum(1 for r in rows if r["guard_fired"])
            breached = sum(1 for r in rows if r["security_breached"])
            utility_ok = sum(1 for r in rows if r["utility"] is True)
            tag = "injected" if injected else "baseline"
            print(
                f"{model:16s} {tag:9s} n={n} guard_fired={fired}/{n} "
                f"security_breached={breached}/{n} utility_pass={utility_ok}/{n}"
            )
    print(f"\nfull log: {out_path}")


if __name__ == "__main__":
    main()
