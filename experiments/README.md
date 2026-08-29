# experiments/

Live-model scripts, as opposed to `tests/` (deterministic, no model
calls). Everything here needs a real Ollama instance, local or rented.

These used to end in `_test.py`, which read as pytest modules sitting
next to `tests/test_agentdojo_harness.py`. Renamed to `_live.py` to
remove that ambiguity; `pytest.ini` also scopes `testpaths = tests` as
a backstop.

## Reading order

Two files are foundational; other scripts import shared setup from
them:

- **`turn_by_turn_live.py`**, the first turn-by-turn harness (a toy
  task suite, not AgentDojo). Exports `_turn_display_result`, reused by
  `agentdojo_live.py` and `retyping_guard_live.py`.
- **`agentdojo_live.py`**, the first real AgentDojo integration
  (banking suite). Exports `make_allowed`, `task_suite`, `_JB_STRING`,
  `SUITE_CLASSIFICATIONS`, and `_run_stmt_with_auto_split`, reused by
  four other scripts below.

| File | Builds on | What it tests |
|---|---|---|
| `feasibility_live.py` | (standalone) | Can a model produce valid, whitelist-passing programs at all. |
| `turn_by_turn_live.py` | (standalone) | Does showing real intermediate results change what a model writes. |
| `agentdojo_live.py` | `turn_by_turn_live.py` | End-to-end AgentDojo banking runs: ground-truth utility, capability-boundary checks, live baseline/injected attempts. |
| `overhead_measurement.py` | `agentdojo_live.py` | prompt-lang vs. AgentDojo's native tool-calling, utility and latency side by side. |
| `defense_comparison.py` | `agentdojo_live.py`, `overhead_measurement.py` | Does AgentDojo's `spotlighting_with_delimiting` defense stop the injection prompt-lang missed. |
| `describe_handle_isolation_live.py` | `agentdojo_live.py` | `describe_handle`'s own model call against real poisoned content, no agent loop needed. |
| `retyping_guard_live.py` | `agentdojo_live.py`, `turn_by_turn_live.py` | `RetypingGuard` + opaque handles + the approval gate, combined. |
| `covert_channel_live.py` | (standalone) | The multi-session covert-channel question (`notes/PRODUCTION_ROADMAP.md` item 11). |

## Running these

Most accept `--check-plumbing` (verify wiring, no model call) and
`--host`/`--model`. Flags aren't fully uniform; check each script's own
`Usage:` docstring.

## `results/`

Raw `.jsonl` output, one file per run. `MANIFEST.md` marks which file
in each family is the current result versus a superseded checkpoint
(moved to `results/checkpoints/`, nothing deleted). `visualize.py`
builds `results/report.html` from the current files; regenerate with
`python3 experiments/results/visualize.py`.
