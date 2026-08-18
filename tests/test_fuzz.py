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


def _generate_case(rng: random.Random) -> tuple[str, bool, bool]:
    """Returns (program, expect_capability_error, expect_confidentiality_error).
    Ground truth (untrusted / secret) is tracked in Python variables as
    the generator builds each statement, independently of anything the
    interpreter computes."""
    stmts = []
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

    action = rng.choice(["privileged", "sink"])
    gate = rng.choice(["direct", "eq_branch", "in_branch"])

    if action == "privileged":
        if gate == "direct":
            stmts.append(f"privileged_action({var})")
        elif gate == "eq_branch":
            stmts.append(f"if {var} == {var}:\n    privileged_action()")
        else:
            stmts.append(f"if {var} in [{var}]:\n    privileged_action()")
        expect_cap_error = untrusted
        expect_conf_error = False
    else:
        if gate == "direct":
            stmts.append(f"sink_action({var})")
        elif gate == "eq_branch":
            stmts.append(f"if {var} == {var}:\n    sink_action()")
        else:
            stmts.append(f"if {var} in [{var}]:\n    sink_action()")
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
