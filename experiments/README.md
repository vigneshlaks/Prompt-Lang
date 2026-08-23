# experiments/

Live-model and live-infrastructure scripts, as distinct from `tests/`
(deterministic, no model involved, run in CI-style on every change).
Everything here calls a real model (local Ollama or a rented GPU) or
otherwise needs a live environment to produce a result -- that's the
actual distinction from `tests/`, not just "less tested."

Naming note, since it's a real point of confusion: several files here
end in `_test.py` and sit right next to `tests/test_agentdojo_harness.py`
in the repo tree. They are not pytest files and pytest does not collect
them (pytest only discovers `tests/`). The `_test` suffix here means
"a script that tests something live," not "a unit test module."

## Reading order / dependency map

Two files are foundational -- other scripts import shared setup from
them rather than duplicating it:

- **`turn_by_turn_test.py`** -- the first turn-by-turn harness (a toy
  task suite, not AgentDojo). Exports `_turn_display_result`, reused by
  `agentdojo_test.py` and `retyping_guard_live_test.py` to work around
  `run_turn()`'s real `None`-for-assignment return (see its own
  docstring). Standalone otherwise; runnable on its own.
- **`agentdojo_test.py`** -- the first real AgentDojo integration
  (banking suite). Exports `make_allowed`, `task_suite`, `_JB_STRING`,
  `SUITE_CLASSIFICATIONS`, and `_run_stmt_with_auto_split`, reused by
  four other scripts below. The largest file here (699 lines) because
  it's both a standalone experiment and the shared AgentDojo-wiring
  module everything else builds on.

Everything else either builds on one of those two, or is fully
self-contained:

| File | Builds on | What it actually tests |
|---|---|---|
| `feasibility_test.py` | (standalone) | Can a model produce syntactically valid, whitelist-passing programs at all, before asking anything harder. |
| `turn_by_turn_test.py` | (standalone) | Does showing a model real intermediate results (vs. write-ahead blind) actually change what it writes. |
| `agentdojo_test.py` | `turn_by_turn_test.py` | End-to-end AgentDojo banking-suite runs: ground-truth utility across all 4 suites, capability-boundary checks, and live baseline/injected agent attempts. |
| `overhead_measurement.py` | `agentdojo_test.py` | prompt-lang vs. AgentDojo's own native tool-calling pipeline, same model/tasks/scoring -- utility and latency side by side. |
| `defense_comparison.py` | `agentdojo_test.py`, `overhead_measurement.py` | Whether AgentDojo's own real `spotlighting_with_delimiting` defense stops the injection prompt-lang's capability system missed. |
| `describe_handle_isolation_test.py` | `agentdojo_test.py` | Just `describe_handle`'s own underlying model call against real poisoned content -- no agent loop, no rented GPU, cheap to iterate on. |
| `retyping_guard_live_test.py` | `agentdojo_test.py`, `turn_by_turn_test.py` | `RetypingGuard` + opaque handles + the approval gate, combined, against the real scenario that found the literal-retyping gap. |
| `covert_channel_test.py` | (standalone) | The multi-session covert-channel persistence question (`notes/PRODUCTION_ROADMAP.md` item 11) -- a toy impossible task and a mundane-named shared store, not AgentDojo. |

## Running any of these

Most accept `--check-plumbing` (verify wiring with no model call) and
`--host`/`--model` for pointing at a local or rented Ollama endpoint.
Check the individual file's own `Usage:` note in its module docstring
for exact flags -- they're not fully uniform across all eight scripts,
since each grew to fit what its own experiment needed.

## `results/`

Raw `.jsonl` output from real runs, one file per experiment invocation.
Names describe what was being checked at the time (e.g.
`overhead_measurement_results_after_final_answer_fix.jsonl`), not a
stable schema -- read `notes/DAILY_SUMMARY.md` for what each real run
actually found; the `.jsonl` files are the underlying data, not a
summary.
