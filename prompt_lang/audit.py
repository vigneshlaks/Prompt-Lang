"""Minimal audit logging for privileged/sinks calls, `notes/
PRODUCTION_ROADMAP.md` item 2's first real piece. `prompt-lang` had no
record of what ran, what got blocked, or why -- so an operator running
a real system on top of it couldn't tell "nothing bad happened" apart
from "something bad happened and nobody knows." This is a first,
honest pass, not the full item: it does not attempt item 9's harder
goal (a per-value provenance chain showing which `sources` call a
blocked value traces back to) -- it only records which named calls ran
or were blocked, and why, at the point a call was attempted.

A real, named limitation, not glossed over: wrap_for_audit_log() can
only observe what happens *inside* a wrapped function call. It cannot
see interpreter.py's own CapabilityError/ConfidentialityError for the
two checks that happen *before* the underlying function is ever
called -- an untrusted argument reaching a privileged call, or a
privileged/sink call made under a tainted pc_trust/pc_secrecy (see
interpreter.py's own eval_node: both checks raise ahead of
`allowed[name](*args, **kwargs)`, so a wrapper around that function
never runs at all in either case). What this module *does* see: every
call that actually executes, and every call blocked by a different
production-layer wrap composed around it (ApprovalDenied from
approval.py, HandleAccessDenied from handles.py, or any real exception
the underlying tool itself raises). Observing interpreter-level blocks
too would need a different mechanism -- catching at the run()/
run_turn() call site, or a future interpreter-level hook -- not
attempted here, named as the honest next step rather than assumed
solved.

Deliberately excludes argument values from the record, not just from
this first version's convenience: logging real call arguments by
default would make the audit log itself a place untrusted or secret
content accumulates, the exact kind of new exposure this project has
spent real effort finding and closing elsewhere (RetypingGuard,
opaque handles). A caller that specifically wants argument values
logged can do so explicitly by inspecting AuditRecord.detail's
already-str exception text or building a richer wrapper on top -- not
the default here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class AuditRecord:
    call_name: str
    outcome: str  # "ran" or "blocked"
    detail: str | None = None  # the exception's own message, if blocked


class AuditLog:
    """Holds the sequence of privileged/sinks call attempts for one
    run/session, in order. One log per run/session, the same scoping
    convention as RetypingGuard/HandleStore/ApprovalGate -- never
    shared across independent runs, since a shared log would mix one
    task's record with another's."""

    def __init__(self):
        self.records: list[AuditRecord] = []

    def record(self, call_name: str, outcome: str, detail: str | None = None) -> None:
        self.records.append(AuditRecord(call_name=call_name, outcome=outcome, detail=detail))


def wrap_for_audit_log(
    allowed: dict[str, Callable],
    privileged: frozenset[str],
    sinks: frozenset[str],
    log: AuditLog,
) -> dict[str, Callable]:
    """Returns a new allowed dict: same functions, except
    privileged/sinks entries are logged immediately around the real
    call. Doesn't modify allowed in place, matching every other
    wrap_for_* in this project. Logging never changes the outcome --
    a blocked call is recorded, then re-raised unchanged, exactly as
    it would have propagated without this wrap. Compose this as the
    outermost wrap if combining with approval.wrap_for_approval or
    handles.wrap_privileged_for_handles, so the log sees the same
    final decision a human reviewing the approval gate would see."""
    wrapped = dict(allowed)
    for name in privileged | sinks:
        if name not in allowed:
            continue
        original = allowed[name]

        def make_wrapper(fn: Callable, fn_name: str) -> Callable:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    result = fn(*args, **kwargs)
                except Exception as exc:
                    log.record(fn_name, "blocked", detail=str(exc))
                    raise
                log.record(fn_name, "ran")
                return result

            return wrapper

        wrapped[name] = make_wrapper(original, name)
    return wrapped
