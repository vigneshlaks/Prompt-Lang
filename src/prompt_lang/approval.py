"""Human-in-the-loop approval gate for privileged/sinks calls whose
argument traces back to a value flagged as needing review.

The gap this closes (see README.md for the full finding): neither
RetypingGuard (prompt_lang/defenses.py) nor opaque handles
(prompt_lang/handles.py) catch a value that was corrupted at its
declassification source (describe_handle answering a legitimate
question with the attacker's IBAN), and then used completely
correctly afterward, through a normal variable reference. Nothing
about *how* that value was used was wrong; the value itself was. No
argument-checking or content-hiding mechanism built so far can see
that, because there's nothing syntactically or structurally unusual
about the call. This module doesn't try to be smarter about judging
the value. It just refuses to let the system decide alone, once a
value has passed through a channel known to carry this risk.

Deliberately coarse, matching the roadmap's own framing for this
proposal. A value gets flagged wherever it's produced by a risky
channel (describe_handle's answer is the motivating and currently
only real case). Any privileged/sinks call later reached with that
value as an argument (checked by substring match, not just exact
equality, matching defenses.RetypingGuard's own precedent for erring
toward more scrutiny rather than less) is routed through an
injected `approve` callback before it's allowed to run at all.

Known, named limitation: the gate is composed as the outermost wrap
around whatever `allowed` it's given (see wrap_for_approval). That
means it sees exactly the arguments the model wrote, before any
Handle resolution happens inside a
handles.wrap_privileged_for_handles-wrapped function. For
describe_handle's answer specifically this doesn't matter: that
value is always a plain string by the time it's bound to a variable,
never a Handle. But a Handle passed directly to a privileged call
(bypassing describe_handle entirely) would not be visible to this
gate's matching in its current form, since the gate never resolves
it. Not fixed here; named so it isn't quietly assumed away.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


class ApprovalDenied(Exception):
    """Raised when a call was routed to the approve() callback and it
    returned False."""


@dataclass(frozen=True)
class ApprovalRequest:
    """What gets handed to the injected approve() callback: enough for
    a human (or a test) to decide, without exposing anything the gate
    itself doesn't already know."""

    call_name: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    flagged_values: list[Any]
    reasons: list[str]


class ApprovalGate:
    """Tracks values that require approval before reaching a
    privileged/sinks call, and holds the callback that actually asks.
    One gate per run/session, same scoping convention as RetypingGuard
    and HandleStore. Never shared across independent runs, a shared
    gate would let one task's flagged values gate another's calls."""

    def __init__(self, approve: Callable[[ApprovalRequest], bool]):
        self._flagged: list[tuple[Any, str]] = []
        self._approve = approve

    def flag(self, value: Any, reason: str) -> None:
        """Marks value as requiring approval before it (or something
        containing it) can reach a privileged/sinks call as an
        argument. Called wherever a value is produced by a channel
        known to carry this risk, describe_handle's answer, most
        directly, but nothing here is specific to that function."""
        self._flagged.append((value, reason))

    def _matches(self, arg: Any) -> list[str]:
        reasons = []
        for flagged_value, reason in self._flagged:
            if isinstance(arg, str) and isinstance(flagged_value, str):
                if flagged_value in arg or arg in flagged_value:
                    reasons.append(reason)
            elif arg == flagged_value:
                reasons.append(reason)
        return reasons

    def check_call(self, call_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        """Raises ApprovalDenied if any argument matches a flagged
        value and the approve() callback refuses it. A no-op if
        nothing matches, no callback invoked at all, so an
        ordinary call with no flagged content is never interrupted."""
        matched_values: list[Any] = []
        matched_reasons: list[str] = []
        for arg in list(args) + list(kwargs.values()):
            reasons = self._matches(arg)
            if reasons:
                matched_values.append(arg)
                matched_reasons.extend(reasons)
        if not matched_values:
            return
        request = ApprovalRequest(
            call_name=call_name,
            args=args,
            kwargs=kwargs,
            flagged_values=matched_values,
            reasons=matched_reasons,
        )
        if not self._approve(request):
            raise ApprovalDenied(
                f"{call_name}() denied: argument(s) {matched_values!r} traced back to "
                f"a flagged value ({'; '.join(matched_reasons)}) and approval was refused"
            )


def wrap_for_approval(
    allowed: dict[str, Callable],
    privileged: frozenset[str],
    sinks: frozenset[str],
    gate: ApprovalGate,
) -> dict[str, Callable]:
    """Returns a new allowed dict: same functions, except
    privileged/sinks entries are routed through gate.check_call()
    immediately before the real (possibly already handle-wrapped)
    function runs. Doesn't modify allowed in place, matching
    defenses.wrap_for_retyping_guard and
    handles.wrap_privileged_for_handles's own convention. Compose this
    as the outermost wrap around a
    handles.wrap_for_opaque_handles-wrapped allowed, if using both, so
    a human reviewing a request sees it, see this module's own
    docstring for what that ordering does and doesn't cover."""
    wrapped = dict(allowed)
    for name in privileged | sinks:
        if name not in allowed:
            continue
        original = allowed[name]

        def make_wrapper(fn: Callable, fn_name: str) -> Callable:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                gate.check_call(fn_name, args, kwargs)
                return fn(*args, **kwargs)

            return wrapper

        wrapped[name] = make_wrapper(original, name)
    return wrapped
