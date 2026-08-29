"""Tests for prompt_lang/audit.py, minimal audit logging for
privileged/sinks calls. These are deterministic, no model or real tool
involved; the wrapped functions are plain fakes."""

import pytest
from prompt_lang.audit import AuditLog, AuditRecord, wrap_for_audit_log
from prompt_lang.interpreter import CapabilityError, run


def test_successful_call_is_logged_as_ran():
    log = AuditLog()
    allowed = wrap_for_audit_log(
        {"send_money": lambda **k: "sent"},
        privileged=frozenset({"send_money"}),
        sinks=frozenset(),
        log=log,
    )
    run("send_money(amount=5)", allowed, privileged=frozenset({"send_money"}))
    assert log.records == [AuditRecord(call_name="send_money", outcome="ran", detail=None)]


def test_a_downstream_exception_is_logged_as_blocked_with_its_message():
    log = AuditLog()

    def rejected(**k):
        raise ValueError("some downstream tool-level rejection")

    allowed = wrap_for_audit_log(
        {"rejected": rejected},
        privileged=frozenset({"rejected"}),
        sinks=frozenset(),
        log=log,
    )
    with pytest.raises(ValueError):
        run("rejected(x=1)", allowed, privileged=frozenset({"rejected"}))
    assert log.records == [
        AuditRecord(call_name="rejected", outcome="blocked", detail="some downstream tool-level rejection")
    ]


def test_logging_never_swallows_the_real_exception():
    # The actual security-relevant property: a wrap that observes an
    # exception must still let it propagate unchanged. A version that
    # accidentally caught and discarded it would silently turn a real
    # rejection into an unblocked call, worse than no logging at all.
    log = AuditLog()

    def rejected(**k):
        raise ValueError("must still propagate")

    allowed = wrap_for_audit_log(
        {"rejected": rejected},
        privileged=frozenset({"rejected"}),
        sinks=frozenset(),
        log=log,
    )
    with pytest.raises(ValueError, match="must still propagate"):
        run("rejected(x=1)", allowed, privileged=frozenset({"rejected"}))


def test_unrelated_functions_are_left_untouched():
    log = AuditLog()
    allowed = wrap_for_audit_log(
        {"send_money": lambda **k: "sent", "get_balance": lambda: 100},
        privileged=frozenset({"send_money"}),
        sinks=frozenset(),
        log=log,
    )
    result = run("get_balance()", allowed)
    assert result == 100
    assert log.records == []


def test_calls_are_recorded_in_order():
    log = AuditLog()
    allowed = wrap_for_audit_log(
        {"a": lambda: "a-ran", "b": lambda: "b-ran"},
        privileged=frozenset({"a", "b"}),
        sinks=frozenset(),
        log=log,
    )
    run("a()", allowed, privileged=frozenset({"a", "b"}))
    run("b()", allowed, privileged=frozenset({"a", "b"}))
    assert [r.call_name for r in log.records] == ["a", "b"]


def test_wrap_does_not_modify_the_original_allowed_dict():
    original = {"send_money": lambda **k: "sent"}
    log = AuditLog()
    wrap_for_audit_log(original, privileged=frozenset({"send_money"}), sinks=frozenset(), log=log)
    run("send_money(amount=1)", original, privileged=frozenset({"send_money"}))
    assert log.records == []


def test_documented_limitation_interpreter_level_blocks_are_not_observed():
    # The honest limitation named in audit.py's own module docstring:
    # a wrapper around a function can only see what happens inside
    # that call. interpreter.py's own CapabilityError, for an
    # untrusted argument reaching a privileged call, is raised before
    # the wrapped function is ever invoked at all, so the audit log
    # has nothing to record for it. This isn't a bug to fix here; it's
    # the actual, named scope boundary of this first pass.
    log = AuditLog()

    def read_untrusted():
        return "attacker text"

    allowed = wrap_for_audit_log(
        {"read_untrusted": read_untrusted, "send_money": lambda **k: "sent"},
        privileged=frozenset({"send_money"}),
        sinks=frozenset(),
        log=log,
    )
    with pytest.raises(CapabilityError):
        run(
            "x = read_untrusted()\nsend_money(amount=x)",
            allowed,
            sources=frozenset({"read_untrusted"}),
            privileged=frozenset({"send_money"}),
        )
    assert log.records == []
