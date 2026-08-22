"""Production-layer defenses that sit outside interpreter.py, described
in notes/PRODUCTION_ROADMAP.md item 12. The core interpreter's
trust/secrecy tracking only ever sees explicit and implicit data flow:
a tagged value reaching a call as an argument, or controlling which
branch a call sits in. It has nothing to see when a model reads
untrusted (or secret) data and then retypes the same content as a
fresh literal in a later statement. Found live, first against
qwen2.5:32b in the AgentDojo integration: a poisoned bill's IBAN got
typed verbatim into send_money(recipient="...") as a plain string
literal, never a reference to the tainted variable holding the bill's
content. A literal is TRUSTED/PUBLIC by construction in eval_node's
ast.Constant case. There is no data-flow edge for any taint tracker,
this one included, to check.

RetypingGuard is a heuristic detection layer for exactly this gap, not
a proof. It remembers the text returned by `sources`/`confidential`
calls during a run, and flags a literal argument to a `privileged`/
`sinks` call whose text overlaps something the model only ever saw by
reading untrusted or secret data. A legitimate literal a model writes
on its own has no reason to exactly match content it never typed
itself. But this is a runtime string-matching heuristic, with a real
false-positive rate (a model correctly echoing a value a human typed
directly into the task, not read from a source) and a real
false-negative rate (an attacker's value reformatted -- spaces
inserted, case changed -- before being retyped won't match verbatim).
Deliberately kept out of interpreter.py: the core module's own
docstring is explicit that the whitelist boundary is about what's
allowed to run, not about catching every mistake a legal operation
can still make.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Callable

from prompt_lang.interpreter import Session, run_turn


class RetypingDetected(Exception):
    """Raised by enforce() when at least one RetypingFlag is present.
    Not raised by RetypingGuard itself -- check_statement always
    returns flags rather than raising. That lets a caller choose to
    log, route to a human, or block, per
    notes/PRODUCTION_ROADMAP.md item 12's three proposed responses.
    enforce() is the convenience path for the blocking response."""


@dataclass(frozen=True)
class RetypingFlag:
    call_name: str
    kind: str  # "privileged" or "sink"
    literal: str
    matched_text: str


def _extract_strings(value: Any, _seen: set[int] | None = None) -> list[str]:
    """Pulls every string worth tracking out of a raw function return
    value: a bare string, or (recursively) strings nested in a list,
    tuple, set, dict, or a plain object's attributes. That's the same
    shape real tools return -- e.g. AgentDojo's Transaction objects,
    which is exactly why interpreter.py's ast.Attribute case exists.
    Deliberately permissive: missing a nested string only weakens
    detection, it never breaks anything. So this errs toward finding
    more text rather than less. _seen guards against a value that
    references itself -- a real tool's return value should never do
    that, but a hostile or buggy one could."""
    if _seen is None:
        _seen = set()
    if isinstance(value, str):
        return [value]
    if id(value) in _seen:
        return []
    if isinstance(value, (list, tuple, set)):
        _seen.add(id(value))
        out: list[str] = []
        for item in value:
            out.extend(_extract_strings(item, _seen))
        return out
    if isinstance(value, dict):
        _seen.add(id(value))
        out = []
        for v in value.values():
            out.extend(_extract_strings(v, _seen))
        return out
    if hasattr(value, "__dict__"):
        _seen.add(id(value))
        out = []
        for v in vars(value).values():
            out.extend(_extract_strings(v, _seen))
        return out
    return []


class RetypingGuard:
    """Tracks text seen from `sources`/`confidential` calls during one
    run, and checks later statements for a literal argument to a
    `privileged`/`sinks` call that overlaps it. One guard per
    run/session. Like Session's env and budget, a guard's tracked text
    is specific to one task and must never be shared across
    independent runs."""

    def __init__(self, min_length: int = 6):
        """min_length excludes short matches (a single digit, a
        common word) that would otherwise false-positive constantly.
        Chosen as a starting point, not measured against real traffic
        yet -- notes/PRODUCTION_ROADMAP.md item 12 says this needs a
        real false-positive-rate measurement before being trusted for
        anything more than detection."""
        self.min_length = min_length
        self._source_texts: list[str] = []
        self._confidential_texts: list[str] = []

    def record_source_output(self, value: Any) -> None:
        self._source_texts.extend(
            t for t in _extract_strings(value) if len(t) >= self.min_length
        )

    def record_confidential_output(self, value: Any) -> None:
        self._confidential_texts.extend(
            t for t in _extract_strings(value) if len(t) >= self.min_length
        )

    def check_statement(
        self,
        source: str,
        privileged: frozenset[str],
        sinks: frozenset[str],
    ) -> list[RetypingFlag]:
        """Parses source (one statement, matching run_turn's own
        contract -- but also works on a full multi-statement program
        the way run() takes one) and returns a RetypingFlag for every
        string-literal argument to a privileged or sinks call that
        overlaps text this guard has recorded. A malformed statement
        returns no flags rather than raising. The interpreter itself
        is responsible for rejecting bad syntax; this is a detection
        layer for otherwise-valid programs, not a second parser."""
        try:
            tree = ast.parse(source, mode="exec")
        except SyntaxError:
            return []
        flags: list[RetypingFlag] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            name = node.func.id
            is_privileged = name in privileged
            is_sink = name in sinks
            if not is_privileged and not is_sink:
                continue
            bucket = self._source_texts if is_privileged else self._confidential_texts
            for arg_node in list(node.args) + [kw.value for kw in node.keywords]:
                if not isinstance(arg_node, ast.Constant) or not isinstance(arg_node.value, str):
                    continue
                literal = arg_node.value
                if len(literal) < self.min_length:
                    continue
                for tracked in bucket:
                    if literal in tracked or tracked in literal:
                        flags.append(
                            RetypingFlag(
                                call_name=name,
                                kind="privileged" if is_privileged else "sink",
                                literal=literal,
                                matched_text=tracked,
                            )
                        )
                        break
        return flags


def enforce(flags: list[RetypingFlag]) -> None:
    """Raises RetypingDetected if flags is non-empty. This is the
    blocking response from notes/PRODUCTION_ROADMAP.md item 12's
    three proposals. A caller that instead wants the "coarser
    human-in-the-loop" or "provenance trace annotation" response
    should inspect flags directly, rather than calling this."""
    if flags:
        raise RetypingDetected(
            "; ".join(
                f"{f.call_name}() called with literal {f.literal!r} matching "
                f"text seen from a {'source' if f.kind == 'privileged' else 'confidential'} "
                f"call earlier this run"
                for f in flags
            )
        )


def run_turn_guarded(
    session: Session,
    guard: RetypingGuard,
    source: str,
    allowed: dict[str, Callable],
    *,
    sources: frozenset[str] = frozenset(),
    privileged: frozenset[str] = frozenset(),
    sanitizers: frozenset[str] = frozenset(),
    confidential: frozenset[str] = frozenset(),
    sinks: frozenset[str] = frozenset(),
    declassifiers: frozenset[str] = frozenset(),
) -> Any:
    """Like interpreter.run_turn(), but checks source against guard
    first. If anything is flagged, it raises RetypingDetected instead
    of executing the statement -- turning RetypingGuard from an
    observe-only layer into an actual block, closing the gap rather
    than just reporting it. `allowed` must already be wrapped with
    wrap_for_retyping_guard() using this same guard, or source/
    confidential text read by later calls won't be recorded, and later
    statements won't be checked against it.

    An alternative was considered first, instead of blocking: an AST
    rewrite that replaces a retyped literal with a reference to the
    variable holding the real value, turning the statement into
    something the trust system would then correctly evaluate on its
    own. It doesn't work for the actual documented case. The tainted
    variable holds the whole source text (e.g. a full bill), and the
    retyped literal is only a substring of it (the IBAN inside the
    bill) -- not equal to the variable's value, so there's no
    whole-variable reference to substitute in. Building the correct
    replacement would mean constructing a substring/slice expression,
    and this grammar has no string indexing or slicing to write one in
    (see notes/ROADMAP.md item 1's grammar remainders). Blocking,
    using the substring match check_statement() already does
    correctly, is the mechanism that actually closes the case that was
    found. The rewrite idea is recorded here rather than silently
    dropped, since it's the kind of thing worth not re-discovering the
    hard way twice.
    """
    flags = guard.check_statement(source, privileged, sinks)
    enforce(flags)
    return run_turn(
        session,
        source,
        allowed,
        sources=sources,
        privileged=privileged,
        sanitizers=sanitizers,
        confidential=confidential,
        sinks=sinks,
        declassifiers=declassifiers,
    )


def wrap_for_retyping_guard(
    allowed: dict[str, Callable],
    sources: frozenset[str],
    confidential: frozenset[str],
    guard: RetypingGuard,
) -> dict[str, Callable]:
    """Returns a new allowed dict: same functions, except
    sources/confidential entries are wrapped to also record their
    return value with guard. Doesn't modify allowed in place --
    matching the rest of this codebase's preference for not mutating
    a caller's dict out from under it. A name in both sources and
    confidential gets recorded to both buckets. interpreter.py has a
    tie-break rule for when a name is listed in two roles (sources
    wins over sanitizers, confidential wins over declassifiers), but
    this doesn't need one at all -- recording to both buckets is
    harmless, since the only effect is more text tracked, not a
    conflicting decision."""
    wrapped = dict(allowed)
    for name in sources | confidential:
        if name not in allowed:
            continue
        original = allowed[name]

        def make_wrapper(fn: Callable, fn_name: str) -> Callable:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                result = fn(*args, **kwargs)
                if fn_name in sources:
                    guard.record_source_output(result)
                if fn_name in confidential:
                    guard.record_confidential_output(result)
                return result

            wrapper.__name__ = getattr(fn, "__name__", fn_name)
            return wrapper

        wrapped[name] = make_wrapper(original, name)
    return wrapped
