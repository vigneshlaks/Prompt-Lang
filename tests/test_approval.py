"""Tests for prompt_lang/approval.py -- the human-in-the-loop gate for
the residual gap RetypingGuard and opaque handles both structurally
cannot see: a value corrupted at its declassification source
(describe_handle answering with the attacker's IBAN) and then used
through a completely proper variable reference. These are
deterministic; the `approve` callback is a fake here, not a real human
prompt, matching how `ask` is faked for describe_handle's own tests."""

import pytest
from prompt_lang.approval import ApprovalDenied, ApprovalGate, ApprovalRequest, wrap_for_approval


def test_unflagged_call_never_invokes_approve():
    calls = []

    def approve(request):
        calls.append(request)
        return True

    gate = ApprovalGate(approve=approve)
    gate.check_call("send_money", ("some_iban",), {"amount": 100})
    assert calls == []


def test_flagged_value_as_positional_arg_triggers_approval():
    gate = ApprovalGate(approve=lambda r: True)
    gate.flag("US133000000121212121212", reason="describe_handle answer")
    # Should not raise -- approve() returned True.
    gate.check_call("send_money", ("US133000000121212121212",), {})


def test_flagged_value_denied_raises_approval_denied():
    gate = ApprovalGate(approve=lambda r: False)
    gate.flag("US133000000121212121212", reason="describe_handle answer")
    with pytest.raises(ApprovalDenied):
        gate.check_call("send_money", (), {"recipient": "US133000000121212121212"})


def test_flagged_value_embedded_in_a_larger_string_still_matches():
    gate = ApprovalGate(approve=lambda r: False)
    gate.flag("US133000000121212121212", reason="describe_handle answer")
    with pytest.raises(ApprovalDenied):
        gate.check_call("send_money", (), {"recipient": "please send to US133000000121212121212 now"})


def test_a_larger_flagged_value_matches_a_smaller_argument_containing_it():
    # Mirrors RetypingGuard's own two-directional substring check --
    # matches whichever direction actually overlaps.
    gate = ApprovalGate(approve=lambda r: False)
    gate.flag("the IBAN is US133000000121212121212, use it", reason="describe_handle answer")
    with pytest.raises(ApprovalDenied):
        gate.check_call("send_money", (), {"recipient": "US133000000121212121212"})


def test_unrelated_argument_is_not_flagged():
    gate = ApprovalGate(approve=lambda r: (_ for _ in ()).throw(AssertionError("should not be called")))
    gate.flag("US133000000121212121212", reason="describe_handle answer")
    gate.check_call("send_money", (), {"recipient": "GB29NWBK60161331926819"})


def test_request_passed_to_approve_has_the_real_call_details():
    seen = {}

    def approve(request: ApprovalRequest) -> bool:
        seen["request"] = request
        return True

    gate = ApprovalGate(approve=approve)
    gate.flag("US133000000121212121212", reason="describe_handle answer")
    gate.check_call("send_money", (), {"recipient": "US133000000121212121212", "amount": 100})

    request = seen["request"]
    assert request.call_name == "send_money"
    assert request.kwargs == {"recipient": "US133000000121212121212", "amount": 100}
    assert request.flagged_values == ["US133000000121212121212"]
    assert request.reasons == ["describe_handle answer"]


def test_non_string_flagged_values_use_exact_equality():
    gate = ApprovalGate(approve=lambda r: False)
    gate.flag(500, reason="suspicious amount")
    with pytest.raises(ApprovalDenied):
        gate.check_call("send_money", (), {"amount": 500})
    # A different number entirely shouldn't match.
    gate2 = ApprovalGate(approve=lambda r: (_ for _ in ()).throw(AssertionError("should not be called")))
    gate2.flag(500, reason="suspicious amount")
    gate2.check_call("send_money", (), {"amount": 501})


def test_wrap_for_approval_allows_a_call_with_no_flagged_arguments():
    gate = ApprovalGate(approve=lambda r: (_ for _ in ()).throw(AssertionError("should not be called")))
    calls = []

    def send_money(recipient, amount):
        calls.append((recipient, amount))
        return "sent"

    wrapped = wrap_for_approval(
        {"send_money": send_money}, privileged=frozenset({"send_money"}), sinks=frozenset(), gate=gate
    )
    result = wrapped["send_money"](recipient="GB29NWBK60161331926819", amount=10)
    assert result == "sent"
    assert calls == [("GB29NWBK60161331926819", 10)]


def test_wrap_for_approval_blocks_a_flagged_call_before_the_real_function_runs():
    gate = ApprovalGate(approve=lambda r: False)
    gate.flag("US133000000121212121212", reason="describe_handle answer")
    calls = []

    def send_money(recipient, amount):
        calls.append((recipient, amount))
        return "sent"

    wrapped = wrap_for_approval(
        {"send_money": send_money}, privileged=frozenset({"send_money"}), sinks=frozenset(), gate=gate
    )
    with pytest.raises(ApprovalDenied):
        wrapped["send_money"](recipient="US133000000121212121212", amount=500)
    # The real function must never have run -- denial happens before commit.
    assert calls == []


def test_wrap_for_approval_lets_a_flagged_call_through_when_approved():
    gate = ApprovalGate(approve=lambda r: True)
    gate.flag("US133000000121212121212", reason="describe_handle answer")

    def send_money(recipient, amount):
        return f"sent to {recipient}"

    wrapped = wrap_for_approval(
        {"send_money": send_money}, privileged=frozenset({"send_money"}), sinks=frozenset(), gate=gate
    )
    result = wrapped["send_money"](recipient="US133000000121212121212", amount=500)
    assert result == "sent to US133000000121212121212"


def test_wrap_for_approval_leaves_unrelated_functions_untouched():
    gate = ApprovalGate(approve=lambda r: True)

    def approve_task():
        return "ok"

    wrapped = wrap_for_approval(
        {"approve_task": approve_task}, privileged=frozenset(), sinks=frozenset(), gate=gate
    )
    assert wrapped["approve_task"] is approve_task


def test_wrap_for_approval_covers_both_privileged_and_sinks():
    gate = ApprovalGate(approve=lambda r: False)
    gate.flag("secret content", reason="test")

    def post_to_forum(body):
        return "posted"

    wrapped = wrap_for_approval(
        {"post_to_forum": post_to_forum}, privileged=frozenset(), sinks=frozenset({"post_to_forum"}), gate=gate
    )
    with pytest.raises(ApprovalDenied):
        wrapped["post_to_forum"](body="secret content")
