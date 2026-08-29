# results/ manifest

Which file in each experiment family is the current result, and what got
moved to `checkpoints/`.

| Family | Current | Moved to `checkpoints/` | Why |
|---|---|---|---|
| feasibility | `feasibility_results.jsonl`, `feasibility_results_loop.jsonl` | n/a | Two different task sets, both current. |
| turn-by-turn | `turn_by_turn_results_32b_v2.jsonl`, `turn_by_turn_results_72b.jsonl` | `turn_by_turn_results.jsonl`, `turn_by_turn_results_32b.jsonl` | Base file predates a harness fix (it showed `None` instead of real assigned values). `_32b.jsonl` predates a later `in`/`not in` fix, superseded by `_32b_v2.jsonl` (93.3% → 96.7% correct, same tasks/model). |
| AgentDojo | `agentdojo_ground_truth_results.jsonl`, `agentdojo_boundary_results.jsonl`, `agentdojo_live_results.jsonl` | n/a | Three different checks, each run once. |
| overhead measurement | `overhead_measurement_results_72b_full_banking.jsonl`, `overhead_measurement_results_72b_task3.jsonl` | `overhead_measurement_results.jsonl`, `_before_final_answer_fix.jsonl.bak`, `_after_final_answer_fix.jsonl`, `_auto_split_live_check.jsonl`, `_string_methods_live_check.jsonl` | The moved files are `qwen2.5:32b`, n=5 bugfix-verification snapshots, superseded by the n=16 full-banking run (which reversed the n=5 read: prompt-lang 2/5 → 8/16, native 1/5 → 11/16). `_72b_task3.jsonl` is kept visible with a caveat: its "confirmed fixed" result did not reproduce in the full run. |
| defense comparison | `defense_comparison_results.jsonl` | n/a | One run, one question. |
| describe_handle isolation | `describe_handle_isolation_results.jsonl` | n/a | One run, one model. |
| retyping guard | `retyping_guard_live_results.jsonl`, `retyping_guard_live_allow_results.jsonl` | n/a | Two real conditions (guard active vs. approval granted), not reruns of each other. |
| covert channel | (none yet) | n/a | Script exists; no run committed yet. |

Nothing in `checkpoints/` was altered, only moved. Old paths in
`notes/DAILY_SUMMARY.md` still find the right file, one directory deeper.
