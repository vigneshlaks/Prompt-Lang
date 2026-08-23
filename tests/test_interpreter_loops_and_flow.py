"""Deterministic tests for while loops, the shared iteration budget,
the implicit-flow pc_trust/pc_secrecy mitigation (a privileged/sink
call gated by a branch condition instead of a direct argument), and
the multi-agent shared-store semantics. Split out of the original
test_interpreter.py; see test_interpreter_core.py's own docstring for
why.
"""

import pytest
from prompt_lang.interpreter import (
    MAX_WHILE_ITERATIONS,
    CapabilityError,
    ConfidentialityError,
    InterpreterError,
    Secrecy,
    Trust,
    run,
)


def _make_shared_store():
    store = {}

    def write_shared(key, value):
        store[key] = value

    def read_shared(key):
        return store[key]

    return store, write_shared, read_shared


def test_while_loop_executes_body_while_condition_true():
    calls = []

    def get_count():
        return len(calls)

    def tick():
        calls.append(1)

    run("while get_count() < 3:\n    tick()", {"get_count": get_count, "tick": tick})
    assert len(calls) == 3


def test_while_loop_false_condition_never_executes_body():
    calls = []

    def mark():
        calls.append(1)

    run("while 1 == 2:\n    mark()", {"mark": mark})
    assert calls == []


def test_while_else_is_rejected():
    with pytest.raises(InterpreterError):
        run("while 1 == 2:\n    x = 1\nelse:\n    y = 2", {})


def test_while_loop_exceeding_iteration_cap_raises():
    def always_true():
        return True

    def noop():
        return None

    with pytest.raises(InterpreterError):
        run("while always_true():\n    noop()", {"always_true": always_true, "noop": noop})


def test_while_loop_iteration_cap_is_exported():
    assert MAX_WHILE_ITERATIONS > 0


def test_untrusted_value_still_blocked_inside_a_while_loop_body():
    def read_secret():
        return "sk-secret"

    def approve(x):
        raise AssertionError("must not be called with an untrusted argument")

    def should_continue():
        return True

    with pytest.raises(CapabilityError):
        run(
            "while should_continue():\n    x = read_secret()\n    approve(x)",
            {"should_continue": should_continue, "read_secret": read_secret, "approve": approve},
            sources=frozenset({"read_secret"}),
            privileged=frozenset({"approve"}),
        )


def test_trusted_value_allowed_into_privileged_op():
    calls = []

    def approve(x):
        calls.append(x)

    run("approve(5)", {"approve": approve}, privileged=frozenset({"approve"}))
    assert calls == [5]


def test_untrusted_value_blocked_from_privileged_op():
    def read_secret():
        return "sk-secret"

    def approve(x):
        raise AssertionError("must not be called with an untrusted argument")

    with pytest.raises(CapabilityError):
        run(
            "approve(read_secret())",
            {"read_secret": read_secret, "approve": approve},
            sources=frozenset({"read_secret"}),
            privileged=frozenset({"approve"}),
        )


def test_untrusted_value_still_blocked_after_passing_through_a_variable():
    def read_secret():
        return "sk-secret"

    def approve(x):
        raise AssertionError("must not be called with an untrusted argument")

    with pytest.raises(CapabilityError):
        run(
            "x = read_secret()\napprove(x)",
            {"read_secret": read_secret, "approve": approve},
            sources=frozenset({"read_secret"}),
            privileged=frozenset({"approve"}),
        )


def test_untrusted_value_allowed_into_a_non_privileged_function():
    calls = []

    def read_secret():
        return "sk-secret"

    def log(x):
        calls.append(x)

    run(
        "log(read_secret())",
        {"read_secret": read_secret, "log": log},
        sources=frozenset({"read_secret"}),
    )
    assert calls == ["sk-secret"]


def _make_shared_store():
    store = {}

    def write_shared(key, value):
        store[key] = value

    def read_shared(key):
        return store[key]

    return store, write_shared, read_shared


def test_planted_value_in_shared_store_blocks_privileged_op():
    store, _, read_shared = _make_shared_store()
    store["inbox"] = "attacker-controlled payload"

    def approve(x):
        raise AssertionError("must not be called with an untrusted argument")

    with pytest.raises(CapabilityError):
        run(
            "x = read_shared('inbox')\napprove(x)",
            {"read_shared": read_shared, "approve": approve},
            sources=frozenset({"read_shared"}),
            privileged=frozenset({"approve"}),
        )


def test_shared_store_value_allowed_into_non_privileged_function():
    store, _, read_shared = _make_shared_store()
    store["inbox"] = "some message"
    calls = []

    def log(x):
        calls.append(x)

    run(
        "log(read_shared('inbox'))",
        {"read_shared": read_shared, "log": log},
        sources=frozenset({"read_shared"}),
    )
    assert calls == ["some message"]


def test_write_then_read_own_shared_value_still_blocked_from_privileged_op():
    store, write_shared, read_shared = _make_shared_store()

    def approve(x):
        raise AssertionError("must not be called with an untrusted argument")

    with pytest.raises(CapabilityError):
        run(
            "write_shared('note', 'hello')\nx = read_shared('note')\napprove(x)",
            {"write_shared": write_shared, "read_shared": read_shared, "approve": approve},
            sources=frozenset({"read_shared"}),
            privileged=frozenset({"approve"}),
        )


# Multi-agent semantics: does the tag survive through a second agent's
# own separate run() call reading, reprocessing, and rewriting shared
# data -- not just one run() call reading its own write, which is all
# the tests above cover.


def test_tag_survives_a_second_agents_separate_run_reading_the_first_agents_write():
    store, write_shared, read_shared = _make_shared_store()

    def approve(x):
        raise AssertionError("must not be called with an untrusted argument")

    # Agent A's turn: writes to the shared store.
    run(
        "write_shared('inbox', 'agent A wrote this')",
        {"write_shared": write_shared},
    )

    # Agent B's turn: a completely separate run() call, its own fresh
    # env, reading what A wrote.
    with pytest.raises(CapabilityError):
        run(
            "x = read_shared('inbox')\napprove(x)",
            {"read_shared": read_shared, "approve": approve},
            sources=frozenset({"read_shared"}),
            privileged=frozenset({"approve"}),
        )


def test_tag_does_not_survive_through_a_sanitizer_across_the_write_read_boundary():
    # A store only holds a bare Python value -- write_shared receives it
    # with its tag already stripped, so a genuinely sanitized value looks
    # identical, once written, to an unsanitized one. The next agent's
    # read is untrusted or not based entirely on that agent's own source
    # declaration, never on what the previous agent did to it.
    store, write_shared, read_shared = _make_shared_store()
    store["inbox"] = "agent A's raw data"

    def identity_sanitizer(x):
        return x

    def approve(x):
        raise AssertionError("must not be called with an untrusted argument")

    # Agent B reads A's data, "sanitizes" it in its own execution, and
    # writes the sanitized result back under a new key.
    run(
        "raw = read_shared('inbox')\n"
        "clean = identity_sanitizer(raw)\n"
        "write_shared('cleaned', clean)",
        {
            "read_shared": read_shared,
            "write_shared": write_shared,
            "identity_sanitizer": identity_sanitizer,
        },
        sources=frozenset({"read_shared"}),
        sanitizers=frozenset({"identity_sanitizer"}),
    )

    # Agent C reads the "cleaned" key -- still untrusted, because C's own
    # read is what determines the tag, not B's history.
    with pytest.raises(CapabilityError):
        run(
            "y = read_shared('cleaned')\napprove(y)",
            {"read_shared": read_shared, "approve": approve},
            sources=frozenset({"read_shared"}),
            privileged=frozenset({"approve"}),
        )


def test_reprocessing_through_an_ordinary_function_still_propagates_across_the_boundary():
    store, write_shared, read_shared = _make_shared_store()
    store["inbox"] = "agent A's raw data"

    def wrap(x):
        return f"[{x}]"

    def approve(x):
        raise AssertionError("must not be called with an untrusted argument")

    run(
        "raw = read_shared('inbox')\n"
        "wrapped = wrap(raw)\n"
        "write_shared('processed', wrapped)",
        {"read_shared": read_shared, "write_shared": write_shared, "wrap": wrap},
        sources=frozenset({"read_shared"}),
    )

    with pytest.raises(CapabilityError):
        run(
            "z = read_shared('processed')\napprove(z)",
            {"read_shared": read_shared, "approve": approve},
            sources=frozenset({"read_shared"}),
            privileged=frozenset({"approve"}),
        )


def test_a_misconfigured_agent_does_not_compromise_a_correctly_configured_one():
    # Agent B's own run() call doesn't declare read_shared as a source --
    # a misconfiguration -- so within B's own execution the value looks
    # trusted to B. That mistake is scoped to B's own run() call; it
    # doesn't weaken Agent C's independent, correctly configured read of
    # the same store afterward.
    store, write_shared, read_shared = _make_shared_store()
    store["inbox"] = "attacker-controlled payload"

    calls = []

    def approve_in_b(x):
        calls.append(x)

    run(
        "x = read_shared('inbox')\napprove_in_b(x)",
        {"read_shared": read_shared, "approve_in_b": approve_in_b},
        privileged=frozenset({"approve_in_b"}),
    )
    assert calls == ["attacker-controlled payload"]  # B's own misconfiguration, not the interpreter's

    def approve_in_c(x):
        raise AssertionError("must not be called with an untrusted argument")

    with pytest.raises(CapabilityError):
        run(
            "y = read_shared('inbox')\napprove_in_c(y)",
            {"read_shared": read_shared, "approve_in_c": approve_in_c},
            sources=frozenset({"read_shared"}),
            privileged=frozenset({"approve_in_c"}),
        )


def test_no_interpreter_state_leaks_between_separate_run_calls():
    calls = []

    def approve(x):
        calls.append(x)

    def loop_bound():
        return 5

    run(
        "n = 0\nwhile n < loop_bound():\n    n = bump(n)",
        {"loop_bound": loop_bound, "bump": lambda x: x + 1},
    )

    run(
        "approve(5)",
        {"approve": approve},
        privileged=frozenset({"approve"}),
    )
    assert calls == [5]


def test_combining_shared_store_data_with_locally_trusted_data_is_untrusted():
    # "Merging trust levels from different sources" isn't a special
    # multi-agent concept -- it's the same join/propagation rule already
    # used for any function taking multiple arguments of different trust.
    store, _, read_shared = _make_shared_store()
    store["inbox"] = "attacker text"

    def combine(a, b):
        return f"{a}-{b}"

    def approve(x):
        raise AssertionError("must not be called with an untrusted argument")

    with pytest.raises(CapabilityError):
        run(
            "approve(combine('trusted-local-value', read_shared('inbox')))",
            {"read_shared": read_shared, "combine": combine, "approve": approve},
            sources=frozenset({"read_shared"}),
            privileged=frozenset({"approve"}),
        )


def test_untrusted_value_still_blocked_inside_a_conditional_branch():
    def read_secret():
        return "sk-secret"

    def approve(x):
        raise AssertionError("must not be called with an untrusted argument")

    with pytest.raises(CapabilityError):
        run(
            "if 1 == 1:\n    x = read_secret()\n    approve(x)",
            {"read_secret": read_secret, "approve": approve},
            sources=frozenset({"read_secret"}),
            privileged=frozenset({"approve"}),
        )


def test_untrusted_value_blocked_as_a_keyword_argument():
    def read_secret():
        return "sk-secret"

    def approve(value):
        raise AssertionError("must not be called with an untrusted argument")

    with pytest.raises(CapabilityError):
        run(
            "approve(value=read_secret())",
            {"read_secret": read_secret, "approve": approve},
            sources=frozenset({"read_secret"}),
            privileged=frozenset({"approve"}),
        )


def test_result_of_a_non_source_function_is_allowed_into_privileged_op():
    calls = []

    def get_config():
        return "default-mode"

    def approve(x):
        calls.append(x)

    run(
        "approve(get_config())",
        {"get_config": get_config, "approve": approve},
        privileged=frozenset({"approve"}),
    )
    assert calls == ["default-mode"]


def test_untrusted_value_that_never_reaches_privileged_op_is_not_blocked():
    calls = []

    def read_secret():
        return "sk-secret"

    def log(x):
        calls.append(x)

    def approve(x):
        raise AssertionError("must not be called at all in this scenario")

    run(
        "x = read_secret()\nlog(x)",
        {"read_secret": read_secret, "log": log, "approve": approve},
        sources=frozenset({"read_secret"}),
        privileged=frozenset({"approve"}),
    )
    assert calls == ["sk-secret"]


def test_untrusted_value_propagates_through_a_chain_of_ordinary_functions():
    def read_secret():
        return "sk-secret"

    def wrap(x):
        return f"[{x}]"

    def approve(x):
        raise AssertionError("must not be called with an untrusted argument")

    with pytest.raises(CapabilityError):
        run(
            "approve(wrap(wrap(read_secret())))",
            {"read_secret": read_secret, "wrap": wrap, "approve": approve},
            sources=frozenset({"read_secret"}),
            privileged=frozenset({"approve"}),
        )


def test_trusted_value_through_ordinary_function_chain_stays_trusted():
    calls = []

    def wrap(x):
        return f"[{x}]"

    def approve(x):
        calls.append(x)

    run(
        "approve(wrap(wrap(5)))",
        {"wrap": wrap, "approve": approve},
        privileged=frozenset({"approve"}),
    )
    assert calls == ["[[5]]"]


def test_ordinary_function_result_is_untrusted_if_any_argument_is_untrusted():
    def read_secret():
        return "sk-secret"

    def combine(a, b):
        return f"{a}-{b}"

    def approve(x):
        raise AssertionError("must not be called with an untrusted argument")

    with pytest.raises(CapabilityError):
        run(
            "approve(combine('safe', read_secret()))",
            {"read_secret": read_secret, "combine": combine, "approve": approve},
            sources=frozenset({"read_secret"}),
            privileged=frozenset({"approve"}),
        )


def test_untrusted_value_used_only_in_comparison_is_not_blocked():
    calls = []

    def read_secret():
        return "sk-secret"

    def notify():
        calls.append("notified")

    def approve(x):
        raise AssertionError("must not be called at all in this scenario")

    run(
        "if read_secret() == 'sk-secret':\n    notify()",
        {"read_secret": read_secret, "notify": notify, "approve": approve},
        sources=frozenset({"read_secret"}),
        privileged=frozenset({"approve"}),
    )
    assert calls == ["notified"]


