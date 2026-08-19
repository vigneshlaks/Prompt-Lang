"""Opaque-handle confinement for `sources` functions, adapted from
SecureClaw's read-path design (Ma & Schmid, arXiv:2606.09549) -- not an
original mechanism, adopted deliberately rather than reinvented. The
gap this closes, verified live this session (see
notes/PRODUCTION_ROADMAP.md item 12): interpreter.py's trust/secrecy
tags only ever see explicit and implicit *data flow* through tracked
variables. They have nothing to see when a model reads untrusted text
and retypes a value from it as a fresh literal -- there's no data-flow
edge for a literal to leave. RetypingGuard (prompt_lang/defenses.py)
tries to catch that after the fact by pattern-matching literals against
recently-seen text. SecureClaw's own paper argues that category of
defense isn't a real boundary on its own: "If the runtime can still
reach the effectful sink, checking arguments does not create a
non-bypassable commit boundary." This module takes the structurally
different approach their paper actually validates (0.64% attack
success rate on AgentDojo, the same benchmark this project already
uses): don't detect a retyped literal, remove the raw value from the
model's reach so there's nothing to retype in the first place.

A `sources` function wrapped with wrap_sources_for_handles() returns a
Handle instead of its real value -- an opaque token, not the content.
A `privileged`/`sinks` function wrapped with wrap_privileged_for_handles()
transparently resolves a Handle argument to the real value from a
shared HandleStore immediately before calling the real underlying
function, checked against that handle's own allowed-sinks policy (set
at mint time) -- matching SecureClaw's "the set of sinks at which
dereference is allowed" being part of the handle itself, not a
separate lookup.

Deliberately outside interpreter.py, same reasoning as defenses.py:
this is production-layer infrastructure a driving harness opts into,
not a change to what the language itself can express. A Handle is an
ordinary Python object with one public field (`id`) and nothing else
-- interpreter.py's existing ast.Attribute handling (Load-context only,
blocks names starting with `_`, blocks callable results) already
governs what a program can read off it with no changes needed, and
there is nothing on it to read that would leak the real value; the
real value lives only in HandleStore's own dict, never on the Handle
object itself.

What this module does NOT attempt: SecureClaw's HMAC-signed binding
digest, freshness/replay protection, and confirmation tokens on the
write path. Those defend against a compromised, out-of-process
executor being tricked by a replayed or tampered request -- a real
concern for their distributed architecture, not a needed one here,
since prompt-lang's interpreter already prevents the model from ever
calling a real function directly; every call is dispatched by the
interpreter itself from a parsed, freshly-evaluated ast.Call node each
time, so there is no separate request object for the model to mutate
after the fact and no replay surface to exploit. What this module does
borrow is the part that actually closes the retyping gap: the raw
value never enters the model's reach in the first place.

describe_handle() (the "bounded summary" for planning) is a much
weaker approximation of SecureClaw's own version, and that gap is real,
not glossed over: their summary is "a deterministic, schema-aware
operator with explicit item and character caps" -- fixed, structured,
incapable of being talked into revealing more than the schema allows.
This module's version asks an LLM a freeform question about the real
content and truncates the answer to a character cap, which is a
weaker guarantee -- a sufficiently adversarial question, or injected
content the summarizing call itself is exposed to, could still coax
more out of it than intended. It is offered because a handle system
with no way to reason about content at all can't support any task
requiring judgment on real data (the turn-by-turn "does this look
manipulative" case this whole project was partly built around) -- but
it should be treated as a partial mitigation, not the rigorous
declassification channel SecureClaw describes.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Any, Callable


class HandleAccessDenied(Exception):
    """Raised when a call tries to resolve a Handle that doesn't exist
    in the store, or exists but isn't authorized for the sink asking to
    dereference it."""


@dataclass(frozen=True)
class Handle:
    """An opaque, unforgeable reference to a real value held outside the
    model's reach. `id` is a high-entropy token (see HandleStore.mint),
    not a hash of the real value -- it carries no information about
    what it points to. The only field on this object is public and
    harmless to expose; the real value is never an attribute of a
    Handle."""

    id: str


class HandleStore:
    """Holds real values behind opaque tokens, and the per-handle policy
    of which sinks may dereference each one. One store per run/session,
    the same way RetypingGuard and Session are scoped -- never shared
    across independent runs, since a shared store would let one task's
    handles be dereferenced by another's calls."""

    def __init__(self):
        self._values: dict[str, Any] = {}
        self._allowed_sinks: dict[str, frozenset[str] | None] = {}

    def mint(self, value: Any, allowed_sinks: frozenset[str] | None = None) -> Handle:
        """Stores value and returns an opaque Handle for it.
        allowed_sinks is the set of privileged/sinks function names
        permitted to dereference this specific handle -- None means any
        sink may (the default, since a first version has to start
        somewhere; a real deployment would classify this per source,
        the same way `sources`/`privileged`/etc. are classified per
        task today)."""
        handle_id = secrets.token_hex(16)
        self._values[handle_id] = value
        self._allowed_sinks[handle_id] = allowed_sinks
        return Handle(id=handle_id)

    def resolve(self, handle: Handle, sink_name: str) -> Any:
        """Returns the real value behind handle, if sink_name is
        authorized to dereference it. Raises HandleAccessDenied
        otherwise -- an unknown handle (never minted by this store, or
        a forged id) and a handle whose policy excludes this sink are
        both refused the same way, since distinguishing them in the
        error would tell an adversary which handles are real."""
        if handle.id not in self._values:
            raise HandleAccessDenied("unknown handle")
        allowed = self._allowed_sinks[handle.id]
        if allowed is not None and sink_name not in allowed:
            raise HandleAccessDenied(f"handle not authorized for {sink_name!r}")
        return self._values[handle.id]

    def peek(self, handle: Handle) -> Any:
        """Returns the real value with no sink policy check -- for
        describe_handle()'s use only (summarizing isn't dereferencing
        at a sink, it's the declassification interface itself), never
        for a privileged/sinks wrapper. Raises HandleAccessDenied for an
        unknown handle, same as resolve()."""
        if handle.id not in self._values:
            raise HandleAccessDenied("unknown handle")
        return self._values[handle.id]


def wrap_sources_for_handles(
    allowed: dict[str, Callable],
    sources: frozenset[str],
    store: HandleStore,
    allowed_sinks: frozenset[str] | None = None,
) -> dict[str, Callable]:
    """Returns a new allowed dict, same functions except sources
    entries return a Handle instead of their real value. Doesn't modify
    allowed in place (same convention as defenses.wrap_for_retyping_guard).
    A wrapped sources function's return is an ordinary, harmless Python
    object as far as interpreter.py is concerned -- no interpreter
    change is needed for a Handle to flow through env, get passed as an
    argument, or be read with `.id`."""
    wrapped = dict(allowed)
    for name in sources:
        if name not in allowed:
            continue
        original = allowed[name]

        def make_wrapper(fn: Callable) -> Callable:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                return store.mint(fn(*args, **kwargs), allowed_sinks)

            return wrapper

        wrapped[name] = make_wrapper(original)
    return wrapped


def wrap_privileged_for_handles(
    allowed: dict[str, Callable],
    privileged: frozenset[str],
    sinks: frozenset[str],
    store: HandleStore,
) -> dict[str, Callable]:
    """Returns a new allowed dict, same functions except privileged/
    sinks entries transparently resolve any Handle argument to its real
    value (policy-checked against that handle's own allowed_sinks)
    immediately before calling the real underlying function. A plain,
    non-Handle argument passes through unchanged -- this only affects
    calls that actually carry a handle. HandleAccessDenied propagates
    out of the call the same way any other rejection from a wrapped
    function does."""
    wrapped = dict(allowed)
    for name in privileged | sinks:
        if name not in allowed:
            continue
        original = allowed[name]

        def make_wrapper(fn: Callable, fn_name: str) -> Callable:
            def resolve_if_handle(value: Any) -> Any:
                return store.resolve(value, fn_name) if isinstance(value, Handle) else value

            def wrapper(*args: Any, **kwargs: Any) -> Any:
                resolved_args = [resolve_if_handle(a) for a in args]
                resolved_kwargs = {k: resolve_if_handle(v) for k, v in kwargs.items()}
                return fn(*resolved_args, **resolved_kwargs)

            return wrapper

        wrapped[name] = make_wrapper(original, name)
    return wrapped


def wrap_for_opaque_handles(
    allowed: dict[str, Callable],
    sources: frozenset[str],
    privileged: frozenset[str],
    sinks: frozenset[str],
    store: HandleStore,
    allowed_sinks: frozenset[str] | None = None,
) -> dict[str, Callable]:
    """Convenience: applies both wraps in one call, the shape a driving
    harness actually wants -- sources mint handles, privileged/sinks
    transparently resolve them, both against the same store."""
    wrapped = wrap_sources_for_handles(allowed, sources, store, allowed_sinks)
    return wrap_privileged_for_handles(wrapped, privileged, sinks, store)


def make_describe_handle(
    store: HandleStore,
    ask: Callable[[str, str], str],
    max_chars: int = 200,
) -> Callable[[Handle, str], str]:
    """Returns a describe_handle(handle, question) function suitable
    for registering in a program's `allowed` dict, meant to be listed
    under `sources` (its answer is still derived from real content and
    should still be tagged UNTRUSTED and still checked by
    defenses.RetypingGuard downstream -- this is a weaker, approximate
    declassification channel, not a replacement for either). `ask` is
    injected rather than hardcoded to one HTTP client, so this is
    testable without a live model -- pass something built the same way
    prompt_lang.tools.interpret() calls a real endpoint, or a fake for
    tests. The character cap is the one concrete bound this version
    actually enforces; see this module's own docstring for what it
    doesn't enforce that SecureClaw's real version does."""

    def describe_handle(handle: Handle, question: str) -> str:
        real_value = store.peek(handle)
        answer = ask(str(real_value), question)
        return answer[:max_chars]

    return describe_handle
