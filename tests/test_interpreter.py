"""Deterministic tests for the safe interpreter core, ported and adapted
from provenance-ac's tests/test_agent_loop.py, which covered the identical
parsing and dispatch pattern for a single tool call.
"""

import pytest
from interpreter import CapabilityError, InterpreterError, run


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


def test_trusted_value_allowed_into_privileged_op():
    calls = []

    def approve(x):
        calls.append(x)

    run("approve(5)", {"approve": approve}, privileged=frozenset({"approve"}))
    assert calls == [5]


def test_tainted_value_blocked_from_privileged_op():
    def read_secret():
        return "sk-secret"

    def approve(x):
        raise AssertionError("must not be called with a tainted argument")

    with pytest.raises(CapabilityError):
        run(
            "approve(read_secret())",
            {"read_secret": read_secret, "approve": approve},
            sources=frozenset({"read_secret"}),
            privileged=frozenset({"approve"}),
        )


def test_tainted_value_still_blocked_after_passing_through_a_variable():
    def read_secret():
        return "sk-secret"

    def approve(x):
        raise AssertionError("must not be called with a tainted argument")

    with pytest.raises(CapabilityError):
        run(
            "x = read_secret()\napprove(x)",
            {"read_secret": read_secret, "approve": approve},
            sources=frozenset({"read_secret"}),
            privileged=frozenset({"approve"}),
        )


def test_tainted_value_allowed_into_a_non_privileged_function():
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
        raise AssertionError("must not be called with a tainted argument")

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
        raise AssertionError("must not be called with a tainted argument")

    with pytest.raises(CapabilityError):
        run(
            "write_shared('note', 'hello')\nx = read_shared('note')\napprove(x)",
            {"write_shared": write_shared, "read_shared": read_shared, "approve": approve},
            sources=frozenset({"read_shared"}),
            privileged=frozenset({"approve"}),
        )
