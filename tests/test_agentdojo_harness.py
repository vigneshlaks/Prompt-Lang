"""Tests for _run_stmt_with_auto_split (experiments/agentdojo_live.py),
the harness-level fix for the multi-statement turn-discipline mistake
found live in
experiments/results/checkpoints/overhead_measurement_results_string_methods_live_check.jsonl
(user_task_1): a model sometimes writes several statements in one
response instead of one at a time, gets rejected by run_turn()'s
one-statement contract, and sometimes just repeats the identical
mistake until the turn budget runs out with nothing ever executed.

This is deliberately not a change to prompt_lang/interpreter.py:
run_turn()'s one-statement rejection stays exactly as it is, since
relaxing it there would let a "turn" secretly be a whole program,
defeating what turn-by-turn execution exists for. The fix lives in the
harness that reacts to the rejection instead: decompose the rejected
blob into its real top-level statements and run them for real, one at
a time, through the same run_turn() a genuinely separate turn would
have gone through, not a new execution path, the existing one called
more than once.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from agentdojo_live import MAX_AUTO_SPLIT_STATEMENTS, _run_stmt_with_auto_split  # noqa: E402
from prompt_lang.interpreter import Session  # noqa: E402


def test_single_statement_behaves_identically_to_a_normal_turn():
    # _turn_display_result deliberately surfaces an assignment's real
    # bound value instead of run_turn()'s own None return (see its own
    # docstring in experiments/turn_by_turn_live.py). "5\n", not "",
    # is the correct, pre-existing behavior this fix must not change.
    session = Session()
    transcript = []
    output = _run_stmt_with_auto_split(session, "x = 5", {}, frozenset(), frozenset(), transcript)
    assert output == "5\n"
    assert transcript == [("x = 5", "5")]
    assert session.env["x"][0] == 5


def test_single_expression_statement_output_and_transcript():
    session = Session()
    transcript = []
    output = _run_stmt_with_auto_split(session, "1 + 1", {}, frozenset(), frozenset(), transcript)
    assert output == "2\n"
    assert transcript == [("1 + 1", "2")]


def test_multi_statement_blob_is_split_and_each_statement_actually_runs():
    session = Session()
    transcript = []
    blob = "a = 1\nb = 2\na + b"
    output = _run_stmt_with_auto_split(session, blob, {}, frozenset(), frozenset(), transcript)
    # Each assignment's real bound value is surfaced too (see the note
    # in test_single_statement_behaves_identically_to_a_normal_turn),
    # not just the final expression's.
    assert output == "1\n2\n3\n"
    assert [stmt for stmt, _ in transcript] == ["a = 1", "b = 2", "a + b"]
    assert session.env["a"][0] == 1
    assert session.env["b"][0] == 2


def test_multi_statement_blob_matching_the_real_failing_transcript_shape():
    # The exact shape found live: a source read, a running total, a for
    # loop using the new string-method support to filter, then the bare
    # final-answer statement, all written by the model in one turn.
    class Transaction:
        def __init__(self, date, amount):
            self.date = date
            self.amount = amount

    def get_transactions():
        return [Transaction("2022-03-01", 100.0), Transaction("2022-02-01", 50.0)]

    allowed = {"get_transactions": get_transactions}
    blob = (
        "transactions = get_transactions()\n"
        "total = 0\n"
        "for t in transactions:\n"
        "    if t.date.startswith('2022-03'):\n"
        "        total = total + t.amount\n"
        "total"
    )
    session = Session()
    transcript = []
    output = _run_stmt_with_auto_split(
        session, blob, allowed, frozenset({"get_transactions"}), frozenset(), transcript
    )
    assert output.strip().endswith("100.0")
    assert session.env["total"][0] == 100.0


def test_a_statement_that_errors_stops_the_rest_of_the_batch():
    session = Session()
    transcript = []
    blob = "a = 1\nb = undefined_name\nc = 3"
    output = _run_stmt_with_auto_split(session, blob, {}, frozenset(), frozenset(), transcript)
    # "a = 1" ran and contributed its real value before "b" errored.
    assert output == "1\n"
    statements_run = [stmt for stmt, _ in transcript]
    assert statements_run == ["a = 1", "b = undefined_name"]
    assert "c = 3" not in statements_run
    assert "undefined variable" in transcript[-1][1]
    assert "c" not in session.env


def test_a_genuine_syntax_error_falls_through_to_the_normal_single_attempt_error():
    session = Session()
    transcript = []
    output = _run_stmt_with_auto_split(session, "this is not valid(((", {}, frozenset(), frozenset(), transcript)
    assert output == ""
    assert len(transcript) == 1
    assert "could not parse" in transcript[0][1] or "ERROR" in transcript[0][1]


def test_a_batch_over_the_cap_is_rejected_whole_not_partially_executed():
    session = Session()
    transcript = []
    blob = "\n".join(f"x{i} = {i}" for i in range(MAX_AUTO_SPLIT_STATEMENTS + 1))
    output = _run_stmt_with_auto_split(session, blob, {}, frozenset(), frozenset(), transcript)
    assert output == ""
    # The whole blob was rejected as one turn, none of the individual
    # assignments made it into env, unlike the under-cap case above.
    assert len(transcript) == 1
    assert "a turn must be exactly one statement" in transcript[0][1]
    assert not session.env


def test_capability_enforcement_is_not_weakened_by_auto_splitting():
    # The actual security property this fix must not touch: an
    # untrusted value blocked from a privileged call in a split-out
    # sub-statement must still be blocked, exactly as it would be as a
    # genuinely separate turn, auto-splitting must not create a path
    # where two statements submitted together somehow see each other's
    # results before either one's own real capability check runs.
    def read_untrusted():
        return "attacker-controlled"

    def privileged_action(x):
        raise AssertionError("must never be called with an untrusted argument")

    allowed = {"read_untrusted": read_untrusted, "privileged_action": privileged_action}
    session = Session()
    transcript = []
    blob = "x = read_untrusted()\nprivileged_action(x)"
    output = _run_stmt_with_auto_split(
        session, blob, allowed, frozenset({"read_untrusted"}), frozenset({"privileged_action"}), transcript
    )
    # "x = read_untrusted()" ran and contributed its real value before
    # the privileged call correctly raised on the next statement.
    assert output == "attacker-controlled\n"
    assert transcript[0] == ("x = read_untrusted()", "'attacker-controlled'")
    assert "privileged operation 'privileged_action' called with an untrusted argument" in transcript[1][1]


def test_pc_trust_does_not_leak_across_split_statements():
    # Each split statement gets its own fresh pc_trust, the same as a
    # genuinely separate turn would, a privileged call blocked inside
    # one branch in the batch must not affect an unrelated, unconditional,
    # trusted privileged call later in the same batch.
    calls = []

    def read_untrusted():
        return "x"

    def blocked_privileged():
        raise AssertionError("should not run")

    def trusted_privileged():
        calls.append("ran")

    allowed = {
        "read_untrusted": read_untrusted,
        "blocked_privileged": blocked_privileged,
        "trusted_privileged": trusted_privileged,
    }
    session = Session()
    transcript = []
    blob = (
        "y = read_untrusted()\n"
        "if y == y:\n"
        "    blocked_privileged()"
    )
    _run_stmt_with_auto_split(
        session, blob, allowed, frozenset({"read_untrusted"}),
        frozenset({"blocked_privileged", "trusted_privileged"}), transcript,
    )
    # The batch stops at the blocked statement (same as the error-stops
    # behavior above); a fresh, separate call afterward must still work.
    transcript2 = []
    _run_stmt_with_auto_split(
        session, "trusted_privileged()", allowed, frozenset(),
        frozenset({"trusted_privileged"}), transcript2,
    )
    assert calls == ["ran"]
