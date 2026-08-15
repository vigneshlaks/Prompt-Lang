"""Deterministic tests for the safe interpreter core, ported and adapted
from provenance-ac's tests/test_agent_loop.py, which covered the identical
parsing and dispatch pattern for a single tool call.
"""

import pytest
from interpreter import MAX_WHILE_ITERATIONS, CapabilityError, InterpreterError, Trust, run


def test_simple_call_dispatches():
    def add_one(x):
        return x + 1

    result = run("add_one(5)", {"add_one": add_one})
    assert result == 6


def test_nested_calls_dispatch_inner_first():
    def double(x):
        return x * 2

    def add_one(x):
        return x + 1

    result = run("add_one(double(5))", {"double": double, "add_one": add_one})
    assert result == 11


def test_keyword_arguments_work():
    def greet(name, greeting="hello"):
        return f"{greeting}, {name}"

    result = run('greet(name="world", greeting="hi")', {"greet": greet})
    assert result == "hi, world"


def test_unknown_name_is_rejected():
    with pytest.raises(InterpreterError):
        run("not_allowed(1)", {})


def test_malformed_source_raises_interpreter_error():
    with pytest.raises(InterpreterError):
        run("this is not valid python(((", {})


def test_arbitrary_code_is_not_executed():
    """Only ast.parse and whitelist dispatch are used, never eval(). A
    call to something not in the whitelist, even a builtin, must be
    rejected."""
    with pytest.raises(InterpreterError):
        run('__import__("os").system("echo pwned")', {})


def test_bare_name_without_call_is_rejected():
    def noop():
        return None

    with pytest.raises(InterpreterError):
        run("noop", {"noop": noop})


def test_assignment_binds_a_readable_variable():
    def add_one(x):
        return x + 1

    result = run("x = 5\nadd_one(x)", {"add_one": add_one})
    assert result == 6


def test_assignment_to_non_name_target_is_rejected():
    with pytest.raises(InterpreterError):
        run("x[0] = 1", {})


def test_conditional_true_branch_executes():
    source = "if 1 == 1:\n    y = 1\nelse:\n    y = 0\ny"
    assert run(source, {}) == 1


def test_conditional_false_branch_executes():
    source = "if 1 == 2:\n    y = 1\nelse:\n    y = 0\ny"
    assert run(source, {}) == 0


def test_conditional_with_undefined_test_variable_is_rejected():
    with pytest.raises(InterpreterError):
        run("if x == 1:\n    y = 1", {})


def test_unsupported_statement_inside_branch_is_rejected():
    with pytest.raises(InterpreterError):
        run("if 1 == 1:\n    import os", {})


def test_sanitizer_result_is_trusted_regardless_of_argument_trust():
    calls = []

    def read_secret():
        return "sk-secret"

    def sanitize(x):
        return "cleaned"

    def approve(x):
        calls.append(x)

    run(
        "approve(sanitize(read_secret()))",
        {"read_secret": read_secret, "sanitize": sanitize, "approve": approve},
        sources=frozenset({"read_secret"}),
        privileged=frozenset({"approve"}),
        sanitizers=frozenset({"sanitize"}),
    )
    assert calls == ["cleaned"]


def test_untrusted_value_still_blocked_when_sanitizer_is_not_called():
    def read_secret():
        return "sk-secret"

    def sanitize(x):
        return "cleaned"

    def approve(x):
        raise AssertionError("must not be called with an untrusted argument")

    with pytest.raises(CapabilityError):
        run(
            "approve(read_secret())",
            {"read_secret": read_secret, "sanitize": sanitize, "approve": approve},
            sources=frozenset({"read_secret"}),
            privileged=frozenset({"approve"}),
            sanitizers=frozenset({"sanitize"}),
        )


def test_function_not_named_as_sanitizer_does_not_clear_trust():
    def read_secret():
        return "sk-secret"

    def wrap(x):
        return f"[{x}]"

    def approve(x):
        raise AssertionError("must not be called with an untrusted argument")

    with pytest.raises(CapabilityError):
        run(
            "approve(wrap(read_secret()))",
            {"read_secret": read_secret, "wrap": wrap, "approve": approve},
            sources=frozenset({"read_secret"}),
            privileged=frozenset({"approve"}),
            sanitizers=frozenset(),
        )


def test_source_wins_when_a_name_is_both_source_and_sanitizer():
    def confused():
        return "value"

    def approve(x):
        raise AssertionError("must not be called with an untrusted argument")

    with pytest.raises(CapabilityError):
        run(
            "approve(confused())",
            {"confused": confused, "approve": approve},
            sources=frozenset({"confused"}),
            privileged=frozenset({"approve"}),
            sanitizers=frozenset({"confused"}),
        )


def test_list_literal_evaluates_to_a_list_of_tagged_elements():
    result = run("[1, 2, 3]", {})
    assert result == [(1, Trust.TRUSTED), (2, Trust.TRUSTED), (3, Trust.TRUSTED)]


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


def test_subscripting_a_plain_list_from_an_outside_function_raises():
    def get_raw_list():
        return [1, 2, 3]

    with pytest.raises(InterpreterError):
        run("x = get_raw_list()\nx[0]", {"get_raw_list": get_raw_list})


def test_for_loop_over_a_plain_list_from_an_outside_function_raises():
    def get_raw_list():
        return [1, 2, 3]

    def visit(x):
        pass

    with pytest.raises(InterpreterError):
        run(
            "for x in get_raw_list():\n    visit(x)",
            {"get_raw_list": get_raw_list, "visit": visit},
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
