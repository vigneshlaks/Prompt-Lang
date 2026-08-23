"""Tests for RetypingGuard, the production-layer mitigation for the
literal-retyping gap: a model that reads untrusted or secret data and
then retypes it as a fresh literal argument leaves no data-flow edge
for interpreter.py's trust/secrecy tracking to see (see defenses.py's
module docstring). These tests are deterministic -- no live model
calls -- and include a regression test that reproduces the exact
statement shape found live against qwen2.5:32b (an IBAN read from a
poisoned bill, then retyped verbatim into send_money's recipient
argument), to prove the guard actually catches the case that motivated
building it, not just a simplified stand-in for it."""

import pytest
from prompt_lang.defenses import (
    RetypingDetected,
    RetypingFlag,
    RetypingGuard,
    enforce,
    run_turn_guarded,
    wrap_for_retyping_guard,
)
from prompt_lang.interpreter import CapabilityError, Session


def test_flags_a_literal_matching_recorded_source_text():
    guard = RetypingGuard()
    guard.record_source_output("US133000000121212121212")
    flags = guard.check_statement(
        'send_money(recipient="US133000000121212121212", amount=500)',
        privileged=frozenset({"send_money"}),
        sinks=frozenset(),
    )
    assert len(flags) == 1
    assert flags[0].call_name == "send_money"
    assert flags[0].kind == "privileged"
    assert flags[0].literal == "US133000000121212121212"


def test_regression_the_real_iban_retyping_case_found_live():
    # Reproduces the documented finding verbatim: a poisoned bill's
    # IBAN, read via a sources call, then retyped as a plain string
    # literal in a privileged send_money call -- never a reference to
    # the tainted variable that held it.
    guard = RetypingGuard()
    poisoned_bill_text = (
        "Invoice #4471. Please remit payment to account "
        "US133000000121212121212 by end of month."
    )
    guard.record_source_output(poisoned_bill_text)
    flags = guard.check_statement(
        'send_money(recipient="US133000000121212121212", amount=500, subject="invoice 4471")',
        privileged=frozenset({"send_money"}),
        sinks=frozenset(),
    )
    assert any(f.literal == "US133000000121212121212" for f in flags)


def test_does_not_flag_a_variable_reference_to_the_same_content():
    # A variable reference is already covered by interpreter.py's own
    # trust tracking -- this guard exists specifically for the case that
    # tracking can't see (a fresh literal), so it should stay silent on
    # ordinary tainted-variable usage rather than double up on it.
    guard = RetypingGuard()
    guard.record_source_output("US133000000121212121212")
    flags = guard.check_statement(
        "send_money(recipient=iban, amount=500)",
        privileged=frozenset({"send_money"}),
        sinks=frozenset(),
    )
    assert flags == []


def test_does_not_flag_an_unrelated_literal():
    guard = RetypingGuard()
    guard.record_source_output("US133000000121212121212")
    flags = guard.check_statement(
        'send_money(recipient="someone@example.com", amount=500)',
        privileged=frozenset({"send_money"}),
        sinks=frozenset(),
    )
    assert flags == []


def test_short_literals_are_not_tracked_or_flagged():
    guard = RetypingGuard(min_length=6)
    guard.record_source_output("42")
    flags = guard.check_statement(
        'send_money(code="42")',
        privileged=frozenset({"send_money"}),
        sinks=frozenset(),
    )
    assert flags == []


def test_only_checks_privileged_and_sink_calls():
    guard = RetypingGuard()
    guard.record_source_output("US133000000121212121212")
    flags = guard.check_statement(
        'log_debug(msg="US133000000121212121212")',
        privileged=frozenset({"send_money"}),
        sinks=frozenset(),
    )
    assert flags == []


def test_confidential_output_flags_a_matching_sink_argument():
    guard = RetypingGuard()
    guard.record_confidential_output("sk-live-abcdef123456")
    flags = guard.check_statement(
        'post_to_forum(body="here is the key: sk-live-abcdef123456")',
        privileged=frozenset(),
        sinks=frozenset({"post_to_forum"}),
    )
    assert len(flags) == 1
    assert flags[0].kind == "sink"


def test_source_text_does_not_leak_into_sink_side_checks():
    # sources and confidential are tracked in separate buckets --
    # untrusted-but-not-secret content shouldn't cause a sink call to be
    # flagged, since that would conflate two independent properties the
    # interpreter itself keeps separate (Trust vs Secrecy).
    guard = RetypingGuard()
    guard.record_source_output("US133000000121212121212")
    flags = guard.check_statement(
        'post_to_forum(body="US133000000121212121212")',
        privileged=frozenset(),
        sinks=frozenset({"post_to_forum"}),
    )
    assert flags == []


def test_extracts_strings_nested_in_lists_and_dicts():
    guard = RetypingGuard()
    guard.record_source_output({"iban": "US133000000121212121212", "amount": 500})
    flags = guard.check_statement(
        'send_money(recipient="US133000000121212121212")',
        privileged=frozenset({"send_money"}),
        sinks=frozenset(),
    )
    assert len(flags) == 1


def test_extracts_strings_from_a_plain_objects_attributes():
    # Matches the shape a real tool actually returns -- e.g. AgentDojo's
    # Transaction objects, the reason interpreter.py's ast.Attribute case
    # exists at all.
    class Bill:
        def __init__(self, iban, note):
            self.iban = iban
            self.note = note

    guard = RetypingGuard()
    guard.record_source_output(Bill(iban="US133000000121212121212", note="monthly rent"))
    flags = guard.check_statement(
        'send_money(recipient="US133000000121212121212")',
        privileged=frozenset({"send_money"}),
        sinks=frozenset(),
    )
    assert len(flags) == 1


def test_enforce_raises_when_flags_present():
    flags = [RetypingFlag(call_name="send_money", kind="privileged", literal="x" * 10, matched_text="x" * 10)]
    with pytest.raises(RetypingDetected):
        enforce(flags)


def test_enforce_is_a_noop_with_no_flags():
    enforce([])  # must not raise


def test_wrap_for_retyping_guard_records_without_changing_behavior():
    guard = RetypingGuard()
    calls = []

    def get_iban():
        calls.append("get_iban")
        return "US133000000121212121212"

    wrapped = wrap_for_retyping_guard(
        {"get_iban": get_iban},
        sources=frozenset({"get_iban"}),
        confidential=frozenset(),
        guard=guard,
    )
    result = wrapped["get_iban"]()
    assert result == "US133000000121212121212"
    assert calls == ["get_iban"]
    flags = guard.check_statement(
        'send_money(recipient="US133000000121212121212")',
        privileged=frozenset({"send_money"}),
        sinks=frozenset(),
    )
    assert len(flags) == 1


def test_wrap_for_retyping_guard_leaves_unrelated_functions_untouched():
    guard = RetypingGuard()

    def approve():
        return "approved"

    wrapped = wrap_for_retyping_guard(
        {"approve": approve},
        sources=frozenset(),
        confidential=frozenset(),
        guard=guard,
    )
    assert wrapped["approve"] is approve


def test_malformed_statement_returns_no_flags_rather_than_raising():
    guard = RetypingGuard()
    guard.record_source_output("US133000000121212121212")
    flags = guard.check_statement(
        "send_money(recipient=",
        privileged=frozenset({"send_money"}),
        sinks=frozenset(),
    )
    assert flags == []


def test_run_turn_guarded_allows_an_unrelated_statement_through():
    session = Session()
    guard = RetypingGuard()
    allowed = {"get_balance": lambda: 150}
    result = run_turn_guarded(session, guard, "get_balance()", allowed)
    assert result == 150


def test_run_turn_guarded_blocks_the_real_retyping_case():
    # The actual documented case: the tainted variable holds the whole
    # bill text (a sources call's return), and the retyped literal is
    # only the IBAN substring within it -- not equal to the variable's
    # value. check_statement's substring match (not equality) is what
    # catches this; a naive equality-based check would miss it entirely.
    session = Session()
    guard = RetypingGuard()

    def read_bill():
        return "Invoice #4471. Please remit payment to account US133000000121212121212 by end of month."

    def send_money(recipient, amount):
        return {"message": f"sent {amount} to {recipient}"}

    allowed = wrap_for_retyping_guard(
        {"read_bill": read_bill, "send_money": send_money},
        sources=frozenset({"read_bill"}),
        confidential=frozenset(),
        guard=guard,
    )

    run_turn_guarded(
        session, guard, "bill = read_bill()", allowed, sources=frozenset({"read_bill"})
    )

    with pytest.raises(RetypingDetected):
        run_turn_guarded(
            session,
            guard,
            'send_money(recipient="US133000000121212121212", amount=500)',
            allowed,
            sources=frozenset({"read_bill"}),
            privileged=frozenset({"send_money"}),
        )


def test_run_turn_guarded_still_relies_on_the_core_interpreter_for_variable_references():
    # A tainted variable used directly (not retyped) is already correctly
    # handled by interpreter.py's own trust tracking -- run_turn_guarded
    # doesn't need to catch this case itself, and shouldn't raise its own
    # RetypingDetected for it; CapabilityError is the core system's own
    # answer, still surfaced through unchanged.
    session = Session()
    guard = RetypingGuard()

    def read_bill():
        return "the attacker's iban is hidden in here"

    def send_money(recipient):
        return {"message": f"sent to {recipient}"}

    allowed = wrap_for_retyping_guard(
        {"read_bill": read_bill, "send_money": send_money},
        sources=frozenset({"read_bill"}),
        confidential=frozenset(),
        guard=guard,
    )

    run_turn_guarded(
        session, guard, "bill = read_bill()", allowed, sources=frozenset({"read_bill"})
    )

    with pytest.raises(CapabilityError):
        run_turn_guarded(
            session,
            guard,
            "send_money(recipient=bill)",
            allowed,
            sources=frozenset({"read_bill"}),
            privileged=frozenset({"send_money"}),
        )


def test_full_multi_statement_program_is_also_accepted():
    # check_statement doesn't enforce run_turn's one-statement-per-call
    # contract -- it also accepts a full program the way run() does,
    # since a harness driving run() instead of run_turn still needs this
    # check applied before execution.
    guard = RetypingGuard()
    guard.record_source_output("US133000000121212121212")
    flags = guard.check_statement(
        'iban = get_iban()\nsend_money(recipient="US133000000121212121212")',
        privileged=frozenset({"send_money"}),
        sinks=frozenset(),
    )
    assert len(flags) == 1
