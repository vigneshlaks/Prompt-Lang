"""Property-based fuzzing for the capability/trust/secrecy system --
item 7 in notes/ROADMAP.md. Every prior adversarial finding (container
laundering, the nested-loop budget bug, both implicit-flow gaps, the
quote-collision bug) came from one person hand-writing a case that
happened to hit a real gap. This generalizes that into a search: build
many random small programs, each with a *ground truth* about whether a
privileged/sink call should be blocked -- known because the generator
built the data-flow chain itself, not derived from the interpreter's
own Trust/Secrecy tags, which would make the check tautological -- and
assert the interpreter's actual behavior matches.

This is structural fuzzing, not full symbolic data-flow fuzzing: the
generator randomizes chain length, which hop types are used (ordinary
passthrough, sanitizer, declassifier, container wrap/unwrap, mixing in
a second untrusted/secret source), and whether the final privileged/sink
call receives the value directly or is only reachable behind a branch
gated on it (including via the in/not-in operator) -- the same shapes
the hand-written adversarial tests sampled a few points from, searched
here at volume instead.

Dicts were entirely absent from this generator until dict iteration
was added to the interpreter (each dict entry now stores a 5-tuple --
value, value_trust, value_secrecy, key_trust, key_secrecy -- instead of
the old 3-tuple, so a key can carry its own precise tag independent of
its value's). Three dict-specific hops were added to close that gap,
not just the basic value-read case: a value-side wrap/unwrap mirroring
the existing listwrap hop, a key-side round-trip that puts the tracked
variable through a dict *key* and reads it back via iteration (the new
capability itself, not just the old value-reading path), and a
cross-contamination check -- the exact adversarial case verified by
hand when this was built: an unrelated entry's freshly-read untrusted
value must not leak onto a different, individually-clean key's own tag
when both live in the same dict.

User-defined functions got the same treatment once function
definitions were added to the interpreter. Every generated program now
opens with three fixed `def` statements (a passthrough identity
function, and two zero-argument wrappers that each call the raw
privileged/sink action with no arguments of their own). A "func_wrap"
hop threads the tracked variable through the identity function
mid-chain, the same regression shape as the existing "ordinary" hop
but through a user-defined call+return instead of a whitelisted one.
More importantly, a new "func_branch" gate calls the zero-argument
wrapper from inside the same branch shape the existing "eq_branch"
gate already uses -- since the wrapper itself takes no arguments, the
*only* way it could ever be blocked is via pc_trust inherited from the
branch condition into the function body, exactly the property that was
hand-verified and mutation-tested when function definitions were
built. This searches that property at random volume across many chain
shapes instead of the handful of hand-written cases.

Fixed seed for reproducibility: a failure here should be a real,
replayable finding, not a flake.
"""

import random

import pytest
from prompt_lang.interpreter import CapabilityError, ConfidentialityError, run

SEED = 20260818
N_CASES = 3000


def _ordinary(x=None):
    return x


def _sanitize(x=None):
    return x


def _declassify(x=None):
    return x


def _mix(a=None, b=None):
    return a


ALLOWED = {
    "read_untrusted": lambda: "untrusted-data",
    "read_secret": lambda: "secret-data",
    "trusted_val": lambda: 1,
    "ordinary": _ordinary,
    "sanitize": _sanitize,
    "declassify": _declassify,
    "mix": _mix,
    "privileged_action": lambda *a, **k: "ok",
    "sink_action": lambda *a, **k: "ok",
}
CAPS = dict(
    sources=frozenset({"read_untrusted"}),
    privileged=frozenset({"privileged_action"}),
    sanitizers=frozenset({"sanitize"}),
    confidential=frozenset({"read_secret"}),
    sinks=frozenset({"sink_action"}),
    declassifiers=frozenset({"declassify"}),
)

# Prepended to every generated program. identity_func is a plain
# passthrough for the "func_wrap" hop. call_privileged/call_sink take
# no arguments of their own -- the only way a branch that calls one of
# them could ever be blocked is via pc_trust inherited from the
# branch's own condition into the function body, which is exactly the
# property the "func_branch" gate below is checking at random volume.
FUNC_DEFS = (
    "def identity_func(x):\n    x\n"
    "def call_privileged():\n    privileged_action()\n"
    "def call_sink():\n    sink_action()"
)


def _generate_case(rng: random.Random) -> tuple[str, bool, bool]:
    """Returns (program, expect_capability_error, expect_confidentiality_error).
    Ground truth (untrusted / secret) is tracked in Python variables as
    the generator builds each statement, independently of anything the
    interpreter computes."""
    stmts = [FUNC_DEFS]
    var = "x"
    untrusted = rng.choice([True, False])
    secret = rng.choice([True, False])

    if untrusted and secret:
        stmts.append(f"{var} = mix(read_untrusted(), read_secret())")
    elif untrusted:
        stmts.append(f"{var} = read_untrusted()")
    elif secret:
        stmts.append(f"{var} = read_secret()")
    else:
        stmts.append(f"{var} = trusted_val()")

    for _ in range(rng.randint(0, 3)):
        hop = rng.choice([
            "ordinary", "sanitize", "declassify", "listwrap",
            "mix_untrusted", "mix_secret",
            "dictwrap", "dict_key_roundtrip", "dict_key_no_cross_contamination",
            "func_wrap",
        ])
        if hop == "ordinary":
            stmts.append(f"{var} = ordinary({var})")
        elif hop == "sanitize":
            stmts.append(f"{var} = sanitize({var})")
            untrusted = False
        elif hop == "declassify":
            stmts.append(f"{var} = declassify({var})")
            secret = False
        elif hop == "listwrap":
            stmts.append(f"{var} = [{var}, trusted_val()][0]")
        elif hop == "mix_untrusted":
            stmts.append(f"{var} = mix({var}, read_untrusted())")
            untrusted = True
        elif hop == "mix_secret":
            stmts.append(f"{var} = mix({var}, read_secret())")
            secret = True
        elif hop == "dictwrap":
            # Value-side wrap/unwrap through a dict -- mirrors listwrap,
            # regression coverage for the read path that already worked
            # before dict iteration existed. Ground truth unchanged: a
            # dict value keeps its own precise tag through a subscript
            # read, same as a list element does.
            stmts.append(f'{var} = {{"k": {var}, "other": trusted_val()}}["k"]')
        elif hop == "dict_key_roundtrip":
            # var becomes a dict *key* (mapped to an unrelated trusted
            # value) and gets read back out via iteration -- the actual
            # new capability, not just the old value-reading path.
            # Single-entry dict, so the loop body runs exactly once;
            # ground truth unchanged if the key's own precise tag
            # survived the round trip correctly.
            stmts.append(f"for _dk in {{{var}: trusted_val()}}:\n    {var} = _dk")
        elif hop == "dict_key_no_cross_contamination":
            # The exact adversarial case verified by hand when dict
            # iteration was built: a second, unrelated entry with a
            # freshly-read untrusted value must not leak onto var's own
            # key. Isolate var's own entry with an explicit equality
            # check inside the loop (dict iteration order alone can't
            # be relied on to land on the right one without it) rather
            # than assuming which key comes first.
            stmts.append(
                f'_poison = {{{var}: trusted_val(), "__poison_marker__": mix(trusted_val(), read_untrusted())}}\n'
                f"for _pk in _poison:\n    if _pk == {var}:\n        {var} = _pk"
            )
        elif hop == "func_wrap":
            # Passthrough hop shaped like "ordinary", but through a
            # user-defined call+return instead of a whitelisted
            # function -- ground truth unchanged if trust/secrecy
            # survive a normal function call round trip.
            stmts.append(f"{var} = identity_func({var})")

    action = rng.choice(["privileged", "sink"])
    gate = rng.choice(["direct", "eq_branch", "in_branch", "func_branch"])

    if action == "privileged":
        if gate == "direct":
            stmts.append(f"privileged_action({var})")
        elif gate == "eq_branch":
            stmts.append(f"if {var} == {var}:\n    privileged_action()")
        elif gate == "in_branch":
            stmts.append(f"if {var} in [{var}]:\n    privileged_action()")
        else:
            stmts.append(f"if {var} == {var}:\n    call_privileged()")
        expect_cap_error = untrusted
        expect_conf_error = False
    else:
        if gate == "direct":
            stmts.append(f"sink_action({var})")
        elif gate == "eq_branch":
            stmts.append(f"if {var} == {var}:\n    sink_action()")
        elif gate == "in_branch":
            stmts.append(f"if {var} in [{var}]:\n    sink_action()")
        else:
            stmts.append(f"if {var} == {var}:\n    call_sink()")
        expect_cap_error = False
        expect_conf_error = secret

    return "\n".join(stmts), expect_cap_error, expect_conf_error


def _run_cases(n: int, seed: int) -> None:
    rng = random.Random(seed)
    for i in range(n):
        program, expect_cap_error, expect_conf_error = _generate_case(rng)
        try:
            run(program, ALLOWED, **CAPS)
            raised_cap_error = False
            raised_conf_error = False
        except CapabilityError:
            raised_cap_error = True
            raised_conf_error = False
        except ConfidentialityError:
            raised_cap_error = False
            raised_conf_error = True

        assert raised_cap_error == expect_cap_error, (
            f"case {i} (seed {seed}): expected CapabilityError="
            f"{expect_cap_error}, got {raised_cap_error}\nprogram:\n{program}"
        )
        assert raised_conf_error == expect_conf_error, (
            f"case {i} (seed {seed}): expected ConfidentialityError="
            f"{expect_conf_error}, got {raised_conf_error}\nprogram:\n{program}"
        )


def test_capability_system_fuzz():
    _run_cases(N_CASES, SEED)


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_capability_system_fuzz_additional_seeds(seed):
    # A handful of extra seeds beyond the primary one, at lower volume,
    # so a gap that the primary seed's random walk happens not to hit
    # still has more than one chance to surface.
    _run_cases(500, seed)
