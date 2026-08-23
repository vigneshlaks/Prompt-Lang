"""Deterministic tests for the safe interpreter core: basic call
dispatch, assignment, conditionals, and integrity/trust propagation
(sources, privileged, sanitizers). Split out of the original
test_interpreter.py, which grew to 3045 lines as every new grammar
feature added its own block -- split by area for navigability, not
by rewriting any test.
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


# Confidentiality: the mirror image of the integrity tests above. A
# `confidential` function's output is always SECRET; a `sinks` function
# refuses to run if any argument is SECRET, raising ConfidentialityError;
# `declassifiers` deliberately clear the SECRET tag. Same shape as
# sources/privileged/sanitizers, checked in the opposite direction.


def test_secret_value_blocked_from_sink():
    def read_api_key():
        return "sk-live-real-key"

    def send_to_webhook(x):
        raise AssertionError("must not be called with a secret argument")

    with pytest.raises(ConfidentialityError):
        run(
            "send_to_webhook(read_api_key())",
            {"read_api_key": read_api_key, "send_to_webhook": send_to_webhook},
            confidential=frozenset({"read_api_key"}),
            sinks=frozenset({"send_to_webhook"}),
        )


def test_secret_value_still_blocked_after_passing_through_a_variable():
    def read_api_key():
        return "sk-live-real-key"

    def send_to_webhook(x):
        raise AssertionError("must not be called with a secret argument")

    with pytest.raises(ConfidentialityError):
        run(
            "x = read_api_key()\nsend_to_webhook(x)",
            {"read_api_key": read_api_key, "send_to_webhook": send_to_webhook},
            confidential=frozenset({"read_api_key"}),
            sinks=frozenset({"send_to_webhook"}),
        )


def test_secret_value_allowed_into_a_non_sink_function():
    calls = []

    def read_api_key():
        return "sk-live-real-key"

    def log_internally(x):
        calls.append(x)

    run(
        "log_internally(read_api_key())",
        {"read_api_key": read_api_key, "log_internally": log_internally},
        confidential=frozenset({"read_api_key"}),
    )
    assert calls == ["sk-live-real-key"]


def test_public_value_allowed_into_sink():
    calls = []

    def send_to_webhook(x):
        calls.append(x)

    run(
        "send_to_webhook(5)",
        {"send_to_webhook": send_to_webhook},
        sinks=frozenset({"send_to_webhook"}),
    )
    assert calls == [5]


def test_declassifier_result_is_public_regardless_of_argument_secrecy():
    calls = []

    def read_api_key():
        return "sk-live-real-key"

    def redact(x):
        return "***"

    def send_to_webhook(x):
        calls.append(x)

    run(
        "send_to_webhook(redact(read_api_key()))",
        {"read_api_key": read_api_key, "redact": redact, "send_to_webhook": send_to_webhook},
        confidential=frozenset({"read_api_key"}),
        declassifiers=frozenset({"redact"}),
        sinks=frozenset({"send_to_webhook"}),
    )
    assert calls == ["***"]


def test_secret_value_still_blocked_when_declassifier_is_not_called():
    def read_api_key():
        return "sk-live-real-key"

    def redact(x):
        return "***"

    def send_to_webhook(x):
        raise AssertionError("must not be called with a secret argument")

    with pytest.raises(ConfidentialityError):
        run(
            "send_to_webhook(read_api_key())",
            {
                "read_api_key": read_api_key,
                "redact": redact,
                "send_to_webhook": send_to_webhook,
            },
            confidential=frozenset({"read_api_key"}),
            declassifiers=frozenset({"redact"}),
            sinks=frozenset({"send_to_webhook"}),
        )


def test_function_not_named_as_declassifier_does_not_clear_secrecy():
    def read_api_key():
        return "sk-live-real-key"

    def wrap(x):
        return f"[{x}]"

    def send_to_webhook(x):
        raise AssertionError("must not be called with a secret argument")

    with pytest.raises(ConfidentialityError):
        run(
            "send_to_webhook(wrap(read_api_key()))",
            {"read_api_key": read_api_key, "wrap": wrap, "send_to_webhook": send_to_webhook},
            confidential=frozenset({"read_api_key"}),
            declassifiers=frozenset(),
            sinks=frozenset({"send_to_webhook"}),
        )


def test_confidential_wins_when_a_name_is_both_confidential_and_declassifier():
    def confused():
        return "value"

    def send_to_webhook(x):
        raise AssertionError("must not be called with a secret argument")

    with pytest.raises(ConfidentialityError):
        run(
            "send_to_webhook(confused())",
            {"confused": confused, "send_to_webhook": send_to_webhook},
            confidential=frozenset({"confused"}),
            declassifiers=frozenset({"confused"}),
            sinks=frozenset({"send_to_webhook"}),
        )


def test_secrecy_propagates_through_a_chain_of_ordinary_functions():
    def read_api_key():
        return "sk-live-real-key"

    def wrap(x):
        return f"[{x}]"

    def send_to_webhook(x):
        raise AssertionError("must not be called with a secret argument")

    with pytest.raises(ConfidentialityError):
        run(
            "send_to_webhook(wrap(wrap(read_api_key())))",
            {"read_api_key": read_api_key, "wrap": wrap, "send_to_webhook": send_to_webhook},
            confidential=frozenset({"read_api_key"}),
            sinks=frozenset({"send_to_webhook"}),
        )


def test_comparison_result_is_secret_if_either_operand_is_secret():
    def read_api_key():
        return "sk-live-real-key"

    def send_to_webhook(x):
        raise AssertionError("must not be called with a secret argument")

    with pytest.raises(ConfidentialityError):
        run(
            "send_to_webhook(read_api_key() == 'guess')",
            {"read_api_key": read_api_key, "send_to_webhook": send_to_webhook},
            confidential=frozenset({"read_api_key"}),
            sinks=frozenset({"send_to_webhook"}),
        )


def test_list_element_keeps_its_own_secrecy_when_indexed():
    def read_api_key():
        return "sk-live-real-key"

    def send_to_webhook(x):
        raise AssertionError("must not be called with a secret argument")

    with pytest.raises(ConfidentialityError):
        run(
            "items = [1, read_api_key()]\nsend_to_webhook(items[1])",
            {"read_api_key": read_api_key, "send_to_webhook": send_to_webhook},
            confidential=frozenset({"read_api_key"}),
            sinks=frozenset({"send_to_webhook"}),
        )


def test_public_list_element_is_not_blocked_even_if_sibling_is_secret():
    calls = []

    def read_api_key():
        return "sk-live-real-key"

    def send_to_webhook(x):
        calls.append(x)

    run(
        "items = [1, read_api_key()]\nsend_to_webhook(items[0])",
        {"read_api_key": read_api_key, "send_to_webhook": send_to_webhook},
        confidential=frozenset({"read_api_key"}),
        sinks=frozenset({"send_to_webhook"}),
    )
    assert calls == [1]


def test_dict_value_keeps_its_own_secrecy_when_subscripted():
    def read_api_key():
        return "sk-live-real-key"

    def send_to_webhook(x):
        raise AssertionError("must not be called with a secret argument")

    with pytest.raises(ConfidentialityError):
        run(
            'd = {"safe": 1, "leak": read_api_key()}\nsend_to_webhook(d["leak"])',
            {"read_api_key": read_api_key, "send_to_webhook": send_to_webhook},
            confidential=frozenset({"read_api_key"}),
            sinks=frozenset({"send_to_webhook"}),
        )


def test_auto_wrapped_list_elements_share_the_calls_own_secret_status():
    def read_secrets():
        return ["sk-one", "sk-two"]

    def send_to_webhook(x):
        raise AssertionError("must not be called with a secret argument")

    with pytest.raises(ConfidentialityError):
        run(
            "for x in read_secrets():\n    send_to_webhook(x)",
            {"read_secrets": read_secrets, "send_to_webhook": send_to_webhook},
            confidential=frozenset({"read_secrets"}),
            sinks=frozenset({"send_to_webhook"}),
        )


