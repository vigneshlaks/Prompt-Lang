"""Tests for Session/run_turn, incremental, turn-by-turn execution as
an alternative to run()'s write-the-whole-program-blind model. These
don't test eval_node/exec_stmt again; they test that persistent state
(env, the iteration budget) and the security guarantees already proven
for run() carry over correctly to execution split across separate
calls."""

import pytest
from prompt_lang.interpreter import (
    CapabilityError,
    InterpreterError,
    Session,
    run_turn,
)


def test_env_accumulates_across_turns():
    session = Session()
    run_turn(session, "x = 5", {})
    result = run_turn(session, "x + 1", {})
    assert result == 6


def test_a_later_turn_can_be_chosen_from_an_earlier_turns_real_result():
    # The actual point of this mechanism: the driver sees the real
    # return value of one turn before deciding what to submit next:
    # something run() structurally can't offer, since the whole program
    # is written before any of it executes.
    session = Session()

    def get_balance():
        return 150

    real_balance = run_turn(session, "get_balance()", {"get_balance": get_balance})
    assert real_balance == 150

    calls = []
    next_stmt = "approve()" if real_balance >= 100 else "deny()"
    run_turn(
        session,
        next_stmt,
        {"approve": lambda: calls.append("approved"), "deny": lambda: calls.append("denied")},
    )
    assert calls == ["approved"]


def test_capability_enforcement_persists_across_turns():
    session = Session()

    def read_secret():
        return "sk-secret"

    def approve(x):
        raise AssertionError("must not be called with an untrusted argument")

    run_turn(session, "x = read_secret()", {"read_secret": read_secret}, sources=frozenset({"read_secret"}))
    with pytest.raises(CapabilityError):
        run_turn(session, "approve(x)", {"approve": approve}, privileged=frozenset({"approve"}))


def test_trusted_value_across_turns_is_not_blocked():
    session = Session()
    calls = []

    def approve(x):
        calls.append(x)

    run_turn(session, "x = 5", {})
    run_turn(session, "approve(x)", {"approve": approve}, privileged=frozenset({"approve"}))
    assert calls == [5]


def test_budget_is_shared_across_turns_not_reset_each_one():
    # The cross-turn analog of the nested-loop-budget finding: many
    # turns, each individually well under the per-loop cap, must still
    # be caught in total by one shared budget, not given a fresh
    # budget every turn, which would let them multiply past it the same
    # way nested loops within one run() call used to.
    session = Session(budget_limit=100)

    def loop_bound():
        return 50

    allowed = {"loop_bound": loop_bound, "get_zero": lambda: 0, "bump": lambda x: x + 1}
    with pytest.raises(InterpreterError):
        for _ in range(10):
            run_turn(session, "n = get_zero()", allowed)
            run_turn(session, "while n < loop_bound():\n    n = bump(n)", allowed)


def test_a_turn_must_be_exactly_one_statement():
    session = Session()
    with pytest.raises(InterpreterError):
        run_turn(session, "x = 1\ny = 2", {})


def test_an_empty_turn_is_rejected():
    session = Session()
    with pytest.raises(InterpreterError):
        run_turn(session, "", {})


def test_malformed_turn_source_raises_interpreter_error():
    session = Session()
    with pytest.raises(InterpreterError):
        run_turn(session, "this is not valid(((", {})


def test_pc_trust_does_not_leak_from_one_turn_into_the_next():
    session = Session()

    def read_secret():
        return "sk-secret"

    def blocked_approve():
        raise AssertionError("should not run")

    with pytest.raises(CapabilityError):
        run_turn(
            session,
            "if read_secret() == 'sk-secret':\n    blocked_approve()",
            {"read_secret": read_secret, "blocked_approve": blocked_approve},
            sources=frozenset({"read_secret"}),
            privileged=frozenset({"blocked_approve"}),
        )

    # A completely separate, unconditional, trusted privileged call in
    # the next turn must be unaffected by turn 1's raised pc_trust.
    calls = []
    run_turn(session, "approve()", {"approve": lambda: calls.append("ok")}, privileged=frozenset({"approve"}))
    assert calls == ["ok"]


def test_two_sessions_are_independent():
    session_a = Session()
    session_b = Session()
    run_turn(session_a, "x = 1", {})
    with pytest.raises(InterpreterError):
        run_turn(session_b, "x", {})


def test_function_defined_in_one_turn_can_be_called_in_a_later_turn():
    # Function definitions persist across turns the same way env and
    # the budget already do, Session.functions mirrors Session.budget.
    session = Session()
    run_turn(session, "def add(a, b):\n    a + b", {})
    result = run_turn(session, "add(2, 3)", {})
    assert result == 5


def test_two_sessions_do_not_share_function_definitions():
    session_a = Session()
    session_b = Session()
    run_turn(session_a, "def add(a, b):\n    a + b", {})
    with pytest.raises(InterpreterError):
        run_turn(session_b, "add(1, 2)", {})
