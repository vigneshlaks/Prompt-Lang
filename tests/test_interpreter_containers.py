"""Deterministic tests for lists and dicts: literals, subscripting,
auto-wrap of plain values returned from external functions, for-loop
iteration, and the adversarial laundering tests specific to
containers. Split out of the original test_interpreter.py; see
test_interpreter_core.py's own docstring for why.
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


def test_nested_loops_cannot_multiply_past_the_total_iteration_budget():
    def outer_cond():
        outer_cond.n = getattr(outer_cond, "n", 0) + 1
        return outer_cond.n <= 300

    def get_zero():
        return 0

    def bump(x):
        return x + 1

    def do_work():
        pass

    with pytest.raises(InterpreterError):
        run(
            "while outer_cond():\n"
            "    inner_x = get_zero()\n"
            "    while inner_x < 300:\n"
            "        inner_x = bump(inner_x)\n"
            "        do_work()",
            {
                "outer_cond": outer_cond,
                "get_zero": get_zero,
                "bump": bump,
                "do_work": do_work,
            },
        )


def test_list_literal_returned_from_run_is_unwrapped_to_plain_values():
    # eval_node's own internal representation of a list is still a list
    # of (value, Trust, Secrecy) triples, unchanged, and still what
    # list-of-tests below exercise indirectly. What changed: run() is a
    # real external boundary, documented ("the Trust and Secrecy tags
    # are unwrapped here, not exposed to callers") to hand back plain
    # values, the same as it always did for a bare scalar, this was
    # never actually true for a list/dict result until now. Found live
    # while wiring real external functions (AgentDojo's real tools) that
    # take list arguments: a caller receiving prompt-lang's own internal
    # tags instead of real values is exactly the same class of leak as
    # passing them to a whitelisted function's own arguments (see
    # test_ordinary_function_receives_real_unwrapped_list_values below).
    assert run("[1, 2, 3]", {}) == [1, 2, 3]


def test_dict_literal_returned_from_run_is_unwrapped_to_plain_values():
    assert run('{"a": 1, "b": 2}', {}) == {"a": 1, "b": 2}


def test_ordinary_function_receives_real_unwrapped_list_values():
    # A real external function (this stub stands in for something like
    # a real AgentDojo tool taking `restaurant_names: list[str]`) must
    # receive actual Python values, not this interpreter's own internal
    # (value, Trust, Secrecy) triples, discovered live wiring up real
    # AgentDojo tools that take list arguments, where a pydantic
    # ValidationError on every element was the first sign something was
    # leaking internal bookkeeping across the call boundary.
    received = []

    def inspect(items):
        received.append(items)
        return items

    run('inspect(["a", "b", "c"])', {"inspect": inspect})
    assert received == [["a", "b", "c"]]


def test_dict_subscript_reads_a_value_by_key():
    result = run('x = {"a": 1, "b": 2}\nx["b"]', {})
    assert result == 2


def test_dict_subscript_missing_key_raises():
    with pytest.raises(InterpreterError):
        run('x = {"a": 1}\nx["missing"]', {})


def test_dict_unpacking_is_rejected():
    with pytest.raises(InterpreterError):
        run('x = {"a": 1}\n{**x, "b": 2}', {})


def test_subscripting_a_plain_value_raises():
    with pytest.raises(InterpreterError):
        run("x = 5\nx[0]", {})


def test_dict_element_keeps_its_own_trust_when_subscripted():
    def read_secret():
        return "sk-secret"

    def approve(x):
        raise AssertionError("must not be called with an untrusted argument")

    with pytest.raises(CapabilityError):
        run(
            'd = {"safe": 1, "leak": read_secret()}\napprove(d["leak"])',
            {"read_secret": read_secret, "approve": approve},
            sources=frozenset({"read_secret"}),
            privileged=frozenset({"approve"}),
        )


def test_trusted_dict_value_is_not_blocked_even_if_sibling_is_untrusted():
    calls = []

    def read_secret():
        return "sk-secret"

    def approve(x):
        calls.append(x)

    run(
        'd = {"safe": 1, "leak": read_secret()}\napprove(d["safe"])',
        {"read_secret": read_secret, "approve": approve},
        sources=frozenset({"read_secret"}),
        privileged=frozenset({"approve"}),
    )
    assert calls == [1]


def test_dict_with_untrusted_value_is_untrusted_as_a_whole():
    def read_secret():
        return "sk-secret"

    def approve(x):
        raise AssertionError("must not be called with an untrusted argument")

    with pytest.raises(CapabilityError):
        run(
            'd = {"safe": 1, "leak": read_secret()}\napprove(d)',
            {"read_secret": read_secret, "approve": approve},
            sources=frozenset({"read_secret"}),
            privileged=frozenset({"approve"}),
        )


def test_dict_with_untrusted_key_is_untrusted_as_a_whole():
    def read_secret():
        return "sk-secret"

    def approve(x):
        raise AssertionError("must not be called with an untrusted argument")

    with pytest.raises(CapabilityError):
        run(
            "d = {read_secret(): 1}\napprove(d)",
            {"read_secret": read_secret, "approve": approve},
            sources=frozenset({"read_secret"}),
            privileged=frozenset({"approve"}),
        )


def test_subscripting_a_plain_dict_from_an_outside_function_is_auto_wrapped():
    def get_config():
        return {"mode": "default"}

    result = run('x = get_config()\nx["mode"]', {"get_config": get_config})
    assert result == "default"


def test_auto_wrapped_dict_values_share_the_calls_own_untrusted_status():
    def read_config():
        return {"key": "sk-secret"}

    def approve(x):
        raise AssertionError("must not be called with an untrusted argument")

    with pytest.raises(CapabilityError):
        run(
            'd = read_config()\napprove(d["key"])',
            {"read_config": read_config, "approve": approve},
            sources=frozenset({"read_config"}),
            privileged=frozenset({"approve"}),
        )


# Adversarial: the same laundering shape already found and fixed for
# lists, checked against dicts up front rather than waiting to
# rediscover it as a second bug later.


def test_sanitizing_a_dict_does_not_launder_an_untrusted_value_inside_it():
    def read_secret():
        return "sk-secret"

    def identity_sanitizer(x):
        return x

    def approve(x):
        raise AssertionError("must not be called with an untrusted argument")

    with pytest.raises(CapabilityError):
        run(
            'd = {"leak": read_secret()}\ny = identity_sanitizer(d)\napprove(y)',
            {
                "read_secret": read_secret,
                "identity_sanitizer": identity_sanitizer,
                "approve": approve,
            },
            sources=frozenset({"read_secret"}),
            sanitizers=frozenset({"identity_sanitizer"}),
            privileged=frozenset({"approve"}),
        )


def test_subscript_reads_an_element_by_index():
    result = run("x = [10, 20, 30]\nx[1]", {})
    assert result == 20


def test_subscript_out_of_range_raises():
    with pytest.raises(InterpreterError):
        run("x = [1, 2]\nx[5]", {})


def test_subscript_with_non_integer_index_raises():
    with pytest.raises(InterpreterError):
        run("x = [1, 2]\nx['a']", {})


def test_for_loop_iterates_over_a_list_literal():
    calls = []

    def visit(x):
        calls.append(x)

    run("for x in [1, 2, 3]:\n    visit(x)", {"visit": visit})
    assert calls == [1, 2, 3]


def test_for_else_is_rejected():
    with pytest.raises(InterpreterError):
        run("for x in [1]:\n    y = x\nelse:\n    y = 0", {})


def test_for_loop_target_must_be_a_plain_name():
    with pytest.raises(InterpreterError):
        run("for x[0] in [1]:\n    pass", {})


def test_subscripting_a_plain_list_from_an_outside_function_is_auto_wrapped():
    def get_raw_list():
        return [1, 2, 3]

    result = run("x = get_raw_list()\nx[1]", {"get_raw_list": get_raw_list})
    assert result == 2


def test_for_loop_over_a_plain_list_from_an_outside_function_is_auto_wrapped():
    def get_raw_list():
        return [1, 2, 3]

    calls = []

    def visit(x):
        calls.append(x)

    run(
        "for x in get_raw_list():\n    visit(x)",
        {"get_raw_list": get_raw_list, "visit": visit},
    )
    assert calls == [1, 2, 3]


def test_auto_wrapped_elements_share_the_calls_own_untrusted_status():
    def read_secrets():
        return ["sk-one", "sk-two"]

    def approve(x):
        raise AssertionError("must not be called with an untrusted argument")

    with pytest.raises(CapabilityError):
        run(
            "for x in read_secrets():\n    approve(x)",
            {"read_secrets": read_secrets, "approve": approve},
            sources=frozenset({"read_secrets"}),
            privileged=frozenset({"approve"}),
        )


def test_auto_wrapped_elements_share_the_calls_own_trusted_status():
    calls = []

    def get_items():
        return [1, 2]

    def approve(x):
        calls.append(x)

    run(
        "for x in get_items():\n    approve(x)",
        {"get_items": get_items, "approve": approve},
        privileged=frozenset({"approve"}),
    )
    assert calls == [1, 2]


def test_function_returning_already_tagged_pairs_opts_out_of_auto_wrap():
    def read_mixed_trust_items():
        return [("sk-secret", Trust.UNTRUSTED, Secrecy.PUBLIC), (1, Trust.TRUSTED, Secrecy.PUBLIC)]

    def approve(x):
        raise AssertionError("must not be called with an untrusted argument")

    with pytest.raises(CapabilityError):
        run(
            "for x in read_mixed_trust_items():\n    approve(x)",
            {"read_mixed_trust_items": read_mixed_trust_items, "approve": approve},
            privileged=frozenset({"approve"}),
        )


def test_list_element_keeps_its_own_trust_when_indexed():
    def read_secret():
        return "sk-secret"

    def approve(x):
        raise AssertionError("must not be called with an untrusted argument")

    with pytest.raises(CapabilityError):
        run(
            "items = [1, read_secret()]\napprove(items[1])",
            {"read_secret": read_secret, "approve": approve},
            sources=frozenset({"read_secret"}),
            privileged=frozenset({"approve"}),
        )


def test_trusted_list_element_is_not_blocked_even_if_sibling_is_untrusted():
    calls = []

    def read_secret():
        return "sk-secret"

    def approve(x):
        calls.append(x)

    run(
        "items = [1, read_secret()]\napprove(items[0])",
        {"read_secret": read_secret, "approve": approve},
        sources=frozenset({"read_secret"}),
        privileged=frozenset({"approve"}),
    )
    assert calls == [1]


def test_for_loop_preserves_per_element_trust():
    def read_secret():
        return "sk-secret"

    def approve(x):
        raise AssertionError("must not be called with an untrusted argument")

    with pytest.raises(CapabilityError):
        run(
            "for item in [read_secret(), 1]:\n    approve(item)",
            {"read_secret": read_secret, "approve": approve},
            sources=frozenset({"read_secret"}),
            privileged=frozenset({"approve"}),
        )


def test_list_containing_an_untrusted_element_is_untrusted_as_a_whole():
    def read_secret():
        return "sk-secret"

    def approve(x):
        raise AssertionError("must not be called with an untrusted argument")

    with pytest.raises(CapabilityError):
        run(
            "items = [1, read_secret()]\napprove(items)",
            {"read_secret": read_secret, "approve": approve},
            sources=frozenset({"read_secret"}),
            privileged=frozenset({"approve"}),
        )


def test_for_loop_iterates_over_a_dict_literals_keys():
    calls = []

    def visit(x):
        calls.append(x)

    run('for k in {"a": 1, "b": 2}:\n    visit(k)', {"visit": visit})
    assert calls == ["a", "b"]


def test_dict_key_keeps_its_own_trust_independent_of_a_sibling_entrys_value():
    # The whole reason dict entries were changed to a 5-tuple instead of
    # the aggregate-tag shortcut: a key from an entry whose own value is
    # untrusted must not drag an unrelated, individually-fine key down
    # with it. This is exactly the case an aggregate dict-level tag
    # would have gotten wrong.
    def read_secret():
        return "untrusted content"

    def approve(x):
        return x

    result = run(
        'd = {"safe_key": read_secret()}\nfor k in d:\n    y = k\napprove(y)',
        {"read_secret": read_secret, "approve": approve},
        sources=frozenset({"read_secret"}),
        privileged=frozenset({"approve"}),
    )
    assert result == "safe_key"


def test_privileged_call_blocked_using_an_untrusted_dict_key_from_iteration():
    def read_untrusted_key():
        return "attacker-controlled-key"

    def approve(x):
        raise AssertionError("must not run: key itself was untrusted")

    with pytest.raises(CapabilityError):
        run(
            'k = read_untrusted_key()\nd = {k: 1}\nfor key in d:\n    approve(key)',
            {"read_untrusted_key": read_untrusted_key, "approve": approve},
            sources=frozenset({"read_untrusted_key"}),
            privileged=frozenset({"approve"}),
        )


def test_dict_with_an_untrusted_key_is_caught_when_passed_wholesale_too():
    # Not just iteration, passing the whole dict directly to a
    # privileged call must also see the untrusted key, the same
    # container-laundering check already applied to values.
    def read_untrusted_key():
        return "attacker-controlled-key"

    def approve(x):
        raise AssertionError("must not run: dict contains an untrusted key")

    with pytest.raises(CapabilityError):
        run(
            'k = read_untrusted_key()\nd = {k: 1}\napprove(d)',
            {"read_untrusted_key": read_untrusted_key, "approve": approve},
            sources=frozenset({"read_untrusted_key"}),
            privileged=frozenset({"approve"}),
        )


def test_sink_call_blocked_using_a_secret_dict_key_from_iteration():
    def get_secret_key():
        return "sk-secret-key"

    def post(x):
        raise AssertionError("must not run: key itself was secret")

    with pytest.raises(ConfidentialityError):
        run(
            'k = get_secret_key()\nd = {k: 1}\nfor key in d:\n    post(key)',
            {"get_secret_key": get_secret_key, "post": post},
            confidential=frozenset({"get_secret_key"}),
            sinks=frozenset({"post"}),
        )


def test_for_loop_over_a_dict_from_an_untrusted_source_taints_pc_trust():
    # Iterable-level implicit-flow protection, mirroring the existing
    # list case: the dict itself coming from an untrusted source must
    # raise pc_trust for the whole loop body, independent of any
    # individual key's own tag.
    def get_dict():
        return {"harmless_key": 1}

    def approve(x):
        raise AssertionError("must not run: reached only via an untrusted iterable")

    with pytest.raises(CapabilityError):
        run(
            "for k in get_dict():\n    approve(k)",
            {"get_dict": get_dict, "approve": approve},
            sources=frozenset({"get_dict"}),
            privileged=frozenset({"approve"}),
        )


def test_dict_indexing_still_returns_the_right_value_after_the_5_tuple_change():
    result = run('d = {"a": 5}\nd["a"]', {})
    assert result == 5


def test_auto_wrapped_dict_from_a_source_taints_both_keys_and_values():
    def read_dict_source():
        return {"x": "y"}

    def approve(x):
        raise AssertionError("must not run: auto-wrapped key was untrusted")

    with pytest.raises(CapabilityError):
        run(
            "d = read_dict_source()\nfor k in d:\n    approve(k)",
            {"read_dict_source": read_dict_source, "approve": approve},
            sources=frozenset({"read_dict_source"}),
            privileged=frozenset({"approve"}),
        )


def test_for_loop_over_a_non_container_raises_a_clear_error():
    with pytest.raises(InterpreterError):
        run("for x in 5:\n    y = x", {})


