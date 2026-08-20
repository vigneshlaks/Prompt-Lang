"""Deterministic tests for the safe interpreter core, ported and adapted
from provenance-ac's tests/test_agent_loop.py, which covered the identical
parsing and dispatch pattern for a single tool call.
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


# Adversarial: the same laundering shape already found and fixed for
# integrity (a container's outer tag cleared while its contents stay
# tagged), checked against confidentiality up front instead of finding
# it as a second, separate bug later.


def test_declassifying_a_list_does_not_launder_a_secret_element_inside_it():
    def read_api_key():
        return "sk-live-real-key"

    def identity_declassifier(x):
        return x

    def send_to_webhook(x):
        raise AssertionError("must not be called with a secret argument")

    with pytest.raises(ConfidentialityError):
        run(
            "items = [read_api_key()]\ny = identity_declassifier(items)\nsend_to_webhook(y)",
            {
                "read_api_key": read_api_key,
                "identity_declassifier": identity_declassifier,
                "send_to_webhook": send_to_webhook,
            },
            confidential=frozenset({"read_api_key"}),
            declassifiers=frozenset({"identity_declassifier"}),
            sinks=frozenset({"send_to_webhook"}),
        )


def test_declassifying_a_dict_does_not_launder_a_secret_value_inside_it():
    def read_api_key():
        return "sk-live-real-key"

    def identity_declassifier(x):
        return x

    def send_to_webhook(x):
        raise AssertionError("must not be called with a secret argument")

    with pytest.raises(ConfidentialityError):
        run(
            'd = {"leak": read_api_key()}\ny = identity_declassifier(d)\nsend_to_webhook(y)',
            {
                "read_api_key": read_api_key,
                "identity_declassifier": identity_declassifier,
                "send_to_webhook": send_to_webhook,
            },
            confidential=frozenset({"read_api_key"}),
            declassifiers=frozenset({"identity_declassifier"}),
            sinks=frozenset({"send_to_webhook"}),
        )


# pc_secrecy: the confidentiality mirror of pc_trust. A secret value can
# still decide which branch runs, leaking information through which sink
# call fires, even when neither call takes a secret argument directly --
# this used to be an open, documented gap; now it's a closed one.


def test_sink_call_blocked_when_its_branch_condition_is_secret():
    def read_api_key():
        return "sk-live-real-key"

    def reveal_match():
        raise AssertionError("must not run: reached only via a secret condition")

    def reveal_no_match():
        raise AssertionError("must not run: reached only via a secret condition")

    with pytest.raises(ConfidentialityError):
        run(
            "if read_api_key() == 'guess':\n    reveal_match()\nelse:\n    reveal_no_match()",
            {
                "read_api_key": read_api_key,
                "reveal_match": reveal_match,
                "reveal_no_match": reveal_no_match,
            },
            confidential=frozenset({"read_api_key"}),
            sinks=frozenset({"reveal_match", "reveal_no_match"}),
        )


def test_sink_call_allowed_when_its_branch_condition_is_public():
    calls = []

    def notify():
        calls.append("notified")

    run(
        "if 1 == 1:\n    notify()",
        {"notify": notify},
        sinks=frozenset({"notify"}),
    )
    assert calls == ["notified"]


def test_sink_call_blocked_inside_a_while_loop_with_secret_condition():
    def read_flag():
        return True

    def notify():
        raise AssertionError("must not run: reached only via a secret condition")

    with pytest.raises(ConfidentialityError):
        run(
            "while read_flag():\n    notify()",
            {"read_flag": read_flag, "notify": notify},
            confidential=frozenset({"read_flag"}),
            sinks=frozenset({"notify"}),
        )


def test_sink_call_blocked_inside_a_for_loop_over_a_secret_iterable():
    def get_items():
        return [1]

    def notify(x):
        raise AssertionError("must not run: reached only via a secret iterable")

    with pytest.raises(ConfidentialityError):
        run(
            "for x in get_items():\n    notify(x)",
            {"get_items": get_items, "notify": notify},
            confidential=frozenset({"get_items"}),
            sinks=frozenset({"notify"}),
        )


# Adversarial: a sanitizer clears the outer trust tag on whatever it
# returns, but if it just passes a tagged list through unchanged, the
# elements inside keep their own tags. Found by deliberately trying to
# launder untrusted data through a container instead of writing another
# confirmation test -- see notes/ROADMAP.md item 4.


def test_sanitizing_a_list_does_not_launder_an_untrusted_element_inside_it():
    def read_secret():
        return "sk-secret"

    def identity_sanitizer(x):
        return x

    def approve(x):
        raise AssertionError("must not be called with an untrusted argument")

    with pytest.raises(CapabilityError):
        run(
            "items = [read_secret()]\ny = identity_sanitizer(items)\napprove(y)",
            {
                "read_secret": read_secret,
                "identity_sanitizer": identity_sanitizer,
                "approve": approve,
            },
            sources=frozenset({"read_secret"}),
            sanitizers=frozenset({"identity_sanitizer"}),
            privileged=frozenset({"approve"}),
        )


def test_indexing_into_a_sanitized_list_still_sees_the_true_nested_tag():
    def read_secret():
        return "sk-secret"

    def identity_sanitizer(x):
        return x

    def approve(x):
        raise AssertionError("must not be called with an untrusted argument")

    with pytest.raises(CapabilityError):
        run(
            "items = [read_secret()]\ny = identity_sanitizer(items)\napprove(y[0])",
            {
                "read_secret": read_secret,
                "identity_sanitizer": identity_sanitizer,
                "approve": approve,
            },
            sources=frozenset({"read_secret"}),
            sanitizers=frozenset({"identity_sanitizer"}),
            privileged=frozenset({"approve"}),
        )


def test_ordinary_function_passing_through_a_sanitized_list_still_propagates_untrusted():
    def read_secret():
        return "sk-secret"

    def identity_sanitizer(x):
        return x

    def wrap(x):
        return x

    def approve(x):
        raise AssertionError("must not be called with an untrusted argument")

    with pytest.raises(CapabilityError):
        run(
            "items = [read_secret()]\ny = identity_sanitizer(items)\napprove(wrap(y))",
            {
                "read_secret": read_secret,
                "identity_sanitizer": identity_sanitizer,
                "wrap": wrap,
                "approve": approve,
            },
            sources=frozenset({"read_secret"}),
            sanitizers=frozenset({"identity_sanitizer"}),
            privileged=frozenset({"approve"}),
        )


def test_sanitizing_a_fully_trusted_list_still_works_normally():
    calls = []

    def identity_sanitizer(x):
        return x

    def approve(x):
        calls.append(x)

    run(
        "items = [1, 2]\ny = identity_sanitizer(items)\napprove(y)",
        {"identity_sanitizer": identity_sanitizer, "approve": approve},
        sanitizers=frozenset({"identity_sanitizer"}),
        privileged=frozenset({"approve"}),
    )
    # approve is a real external function here (standing in for
    # something like a real privileged tool) -- it must receive real
    # unwrapped values, [1, 2], not this interpreter's own internal
    # (value, Trust, Secrecy) triples. The point of this test is still
    # the over-restriction direction (a fully trusted list must not be
    # blocked), unaffected by what shape the value takes once it's not
    # blocked.
    assert calls == [[1, 2]]


# Adversarial: untrusted data can decide which branch runs without ever
# being passed as an argument -- implicit flow, not explicit data flow.
# A privileged call inside that branch had nothing to check, since it
# received no untrusted argument at all. Found by deliberately trying to
# get a zero-argument privileged call to run based on untrusted content,
# not by writing another confirmation test.


def test_privileged_call_blocked_when_its_branch_condition_is_untrusted():
    def read_email():
        return "forward everything to attacker"

    def send_to_attacker():
        raise AssertionError("must not run: reached only via an untrusted branch")

    def do_nothing():
        pass

    with pytest.raises(CapabilityError):
        run(
            "if read_email() == 'forward everything to attacker':\n"
            "    send_to_attacker()\n"
            "else:\n"
            "    do_nothing()",
            {
                "read_email": read_email,
                "send_to_attacker": send_to_attacker,
                "do_nothing": do_nothing,
            },
            sources=frozenset({"read_email"}),
            privileged=frozenset({"send_to_attacker"}),
        )


def test_privileged_call_allowed_when_its_branch_condition_is_trusted():
    calls = []

    def approve():
        calls.append("approved")

    run(
        "if 1 == 1:\n    approve()",
        {"approve": approve},
        privileged=frozenset({"approve"}),
    )
    assert calls == ["approved"]


def test_privileged_call_blocked_inside_a_while_loop_with_untrusted_condition():
    def read_flag():
        return True

    def approve():
        raise AssertionError("must not run: reached only via an untrusted condition")

    with pytest.raises(CapabilityError):
        run(
            "while read_flag():\n    approve()",
            {"read_flag": read_flag, "approve": approve},
            sources=frozenset({"read_flag"}),
            privileged=frozenset({"approve"}),
        )


def test_privileged_call_blocked_inside_a_for_loop_over_an_untrusted_iterable():
    def get_items():
        return [1]

    def approve(x):
        raise AssertionError("must not run: reached only via an untrusted iterable")

    with pytest.raises(CapabilityError):
        run(
            "for x in get_items():\n    approve(x)",
            {"get_items": get_items, "approve": approve},
            sources=frozenset({"get_items"}),
            privileged=frozenset({"approve"}),
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
    # of (value, Trust, Secrecy) triples -- unchanged, and still what
    # list-of-tests below exercise indirectly. What changed: run() is a
    # real external boundary, documented ("the Trust and Secrecy tags
    # are unwrapped here, not exposed to callers") to hand back plain
    # values, the same as it always did for a bare scalar -- this was
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
    # (value, Trust, Secrecy) triples -- discovered live wiring up real
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
    # Not just iteration -- passing the whole dict directly to a
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


# Multi-agent semantics (notes/ROADMAP.md item 3): does the tag survive
# through a second agent's own separate run() call reading, reprocessing,
# and rewriting shared data -- not just one run() call reading its own
# write, which is all the tests above cover.


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


# Arithmetic: same shape as the comparison operators above, but computing
# a value instead of a boolean.


def test_addition():
    assert run("2 + 3", {}) == 5


def test_subtraction():
    assert run("10 - 4", {}) == 6


def test_multiplication():
    assert run("3 * 4", {}) == 12


def test_division():
    assert run("10 / 4", {}) == 2.5


def test_arithmetic_on_variables():
    assert run("x = 2\ny = 3\nx + y", {}) == 5


def test_division_by_zero_propagates_as_a_normal_python_exception():
    # Not caught or converted -- the whitelist boundary is about what's
    # allowed to run, not about catching every mistake a legal operation
    # can still make (same stance as a whitelisted call with the wrong
    # argument types).
    with pytest.raises(ZeroDivisionError):
        run("1 / 0", {})


def test_unsupported_binop_operator_raises():
    with pytest.raises(InterpreterError):
        run("2 & 3", {})


def test_arithmetic_result_is_untrusted_if_either_operand_is_untrusted():
    def read_secret():
        return 5

    def approve(x):
        raise AssertionError("must not be called with an untrusted argument")

    with pytest.raises(CapabilityError):
        run(
            "approve(1 + read_secret())",
            {"read_secret": read_secret, "approve": approve},
            sources=frozenset({"read_secret"}),
            privileged=frozenset({"approve"}),
        )


def test_arithmetic_result_is_secret_if_either_operand_is_secret():
    def read_api_key_length():
        return 20

    def send_to_webhook(x):
        raise AssertionError("must not be called with a secret argument")

    with pytest.raises(ConfidentialityError):
        run(
            "send_to_webhook(1 + read_api_key_length())",
            {"read_api_key_length": read_api_key_length, "send_to_webhook": send_to_webhook},
            confidential=frozenset({"read_api_key_length"}),
            sinks=frozenset({"send_to_webhook"}),
        )


def test_arithmetic_on_two_trusted_values_is_not_blocked():
    calls = []

    def approve(x):
        calls.append(x)

    run(
        "approve(2 + 3)",
        {"approve": approve},
        privileged=frozenset({"approve"}),
    )
    assert calls == [5]


def test_privileged_call_blocked_behind_an_untrusted_arithmetic_branch_condition():
    # pc_trust has to pick up BinOp's trust the same way it already picks
    # up Compare's -- an arithmetic expression used directly as an if
    # condition is just as capable of implicit flow as a comparison is.
    def read_secret():
        return 1

    def approve():
        raise AssertionError("must not run: reached only via an untrusted condition")

    with pytest.raises(CapabilityError):
        run(
            "if 1 + read_secret():\n    approve()",
            {"read_secret": read_secret, "approve": approve},
            sources=frozenset({"read_secret"}),
            privileged=frozenset({"approve"}),
        )


# Full operator sweep: remaining arithmetic (//, %, **), unary (-, +, not),
# boolean (and, or), and chained comparisons. Same shape and rigor as the
# +/-/*/ additions above -- correctness, capability propagation, and a
# pc_trust/pc_secrecy regression test for every new node type used as a
# branch condition, since that's exactly the category of gap the original
# ast.Compare fix (and the arithmetic one after it) needed.


def test_floor_division():
    assert run("10 // 3", {}) == 3


def test_modulo():
    assert run("10 % 3", {}) == 1


def test_exponentiation():
    assert run("2 ** 10", {}) == 1024


def test_exponent_magnitude_over_the_cap_raises():
    with pytest.raises(InterpreterError):
        run("2 ** 999999999999", {})


def test_large_base_with_small_exponent_is_not_blocked_by_the_exponent_cap():
    # The guard is on the exponent's magnitude, not the base's -- a large
    # base raised to a small exponent is cheap and should not be rejected.
    assert run("(10 ** 300) ** 2", {}) == 10 ** 600


def test_unary_minus_makes_negative_literals_work():
    # Before this, -5 had no supported AST case at all: ast.parse never
    # folds a negative literal into one constant, it's UnaryOp(USub,
    # Constant(5)), two nodes.
    assert run("-5", {}) == -5


def test_unary_plus():
    assert run("+5", {}) == 5


def test_unary_not():
    assert run("not True", {}) is False
    assert run("not False", {}) is True


def test_boolean_and():
    assert run("True and False", {}) is False
    assert run("True and True", {}) is True


def test_boolean_or():
    assert run("False or True", {}) is True
    assert run("False or False", {}) is False


def test_boolean_and_short_circuits_and_does_not_evaluate_the_second_operand():
    calls = []

    def side_effect():
        calls.append("called")
        return True

    run("False and side_effect()", {"side_effect": side_effect})
    assert calls == []


def test_boolean_or_short_circuits_and_does_not_evaluate_the_second_operand():
    calls = []

    def side_effect():
        calls.append("called")
        return True

    run("True or side_effect()", {"side_effect": side_effect})
    assert calls == []


def test_chained_comparison():
    assert run("0 <= 5 <= 100", {}) is True
    assert run("0 <= 500 <= 100", {}) is False


def test_chained_comparison_short_circuits_and_evaluates_each_operand_once():
    calls = []

    def read():
        calls.append("read")
        return 5

    # 10 < 5 is already false, so read() (standing in for the third
    # operand) must never be called.
    result = run("10 < 5 < read()", {"read": read})
    assert result is False
    assert calls == []


def test_chained_comparison_evaluates_a_shared_operand_exactly_once():
    calls = []

    def middle():
        calls.append("middle")
        return 5

    # middle() is used in both comparisons but must only be called once.
    run("1 < middle() < 10", {"middle": middle})
    assert calls == ["middle"]


def test_unary_operand_keeps_its_own_trust():
    def read_secret():
        return 5

    def approve(x):
        raise AssertionError("must not be called with an untrusted argument")

    with pytest.raises(CapabilityError):
        run(
            "approve(-read_secret())",
            {"read_secret": read_secret, "approve": approve},
            sources=frozenset({"read_secret"}),
            privileged=frozenset({"approve"}),
        )


def test_boolean_result_is_untrusted_if_an_evaluated_operand_is_untrusted():
    def read_secret():
        return True

    def approve(x):
        raise AssertionError("must not be called with an untrusted argument")

    with pytest.raises(CapabilityError):
        run(
            "approve(True and read_secret())",
            {"read_secret": read_secret, "approve": approve},
            sources=frozenset({"read_secret"}),
            privileged=frozenset({"approve"}),
        )


def test_boolean_result_is_trusted_if_the_untrusted_operand_was_short_circuited_away():
    calls = []

    def read_secret():
        raise AssertionError("must not be called: short-circuited away")

    def approve(x):
        calls.append(x)

    run(
        "approve(False and read_secret())",
        {"read_secret": read_secret, "approve": approve},
        sources=frozenset({"read_secret"}),
        privileged=frozenset({"approve"}),
    )
    assert calls == [False]


def test_chained_comparison_result_is_untrusted_if_an_evaluated_operand_is_untrusted():
    def read_secret():
        return 5

    def approve(x):
        raise AssertionError("must not be called with an untrusted argument")

    with pytest.raises(CapabilityError):
        run(
            "approve(1 < read_secret() < 10)",
            {"read_secret": read_secret, "approve": approve},
            sources=frozenset({"read_secret"}),
            privileged=frozenset({"approve"}),
        )


def test_privileged_call_blocked_behind_an_untrusted_unary_branch_condition():
    def read_secret():
        return False

    def approve():
        raise AssertionError("must not run: reached only via an untrusted condition")

    with pytest.raises(CapabilityError):
        run(
            "if not read_secret():\n    approve()",
            {"read_secret": read_secret, "approve": approve},
            sources=frozenset({"read_secret"}),
            privileged=frozenset({"approve"}),
        )


def test_privileged_call_blocked_behind_an_untrusted_boolean_branch_condition():
    def read_secret():
        return True

    def approve():
        raise AssertionError("must not run: reached only via an untrusted condition")

    with pytest.raises(CapabilityError):
        run(
            "if True and read_secret():\n    approve()",
            {"read_secret": read_secret, "approve": approve},
            sources=frozenset({"read_secret"}),
            privileged=frozenset({"approve"}),
        )


def test_privileged_call_blocked_behind_a_chained_comparison_branch_condition():
    def read_secret():
        return 5

    def approve():
        raise AssertionError("must not run: reached only via an untrusted condition")

    with pytest.raises(CapabilityError):
        run(
            "if 1 < read_secret() < 10:\n    approve()",
            {"read_secret": read_secret, "approve": approve},
            sources=frozenset({"read_secret"}),
            privileged=frozenset({"approve"}),
        )


def test_sink_call_blocked_behind_a_secret_boolean_branch_condition():
    def read_api_key():
        return True

    def reveal():
        raise AssertionError("must not run: reached only via a secret condition")

    with pytest.raises(ConfidentialityError):
        run(
            "if True and read_api_key():\n    reveal()",
            {"read_api_key": read_api_key, "reveal": reveal},
            confidential=frozenset({"read_api_key"}),
            sinks=frozenset({"reveal"}),
        )


def test_sink_call_blocked_behind_a_secret_chained_comparison_branch_condition():
    def read_api_key_num():
        return 5

    def reveal():
        raise AssertionError("must not run: reached only via a secret condition")

    with pytest.raises(ConfidentialityError):
        run(
            "if 1 < read_api_key_num() < 10:\n    reveal()",
            {"read_api_key_num": read_api_key_num, "reveal": reveal},
            confidential=frozenset({"read_api_key_num"}),
            sinks=frozenset({"reveal"}),
        )


def test_sink_call_blocked_behind_a_secret_unary_branch_condition():
    def read_api_key_bool():
        return False

    def reveal():
        raise AssertionError("must not run: reached only via a secret condition")

    with pytest.raises(ConfidentialityError):
        run(
            "if not read_api_key_bool():\n    reveal()",
            {"read_api_key_bool": read_api_key_bool, "reveal": reveal},
            confidential=frozenset({"read_api_key_bool"}),
            sinks=frozenset({"reveal"}),
        )


def test_privileged_call_not_blocked_behind_a_fully_trusted_boolean_condition():
    calls = []

    def approve():
        calls.append("approved")

    run(
        "if True and 1 < 2 < 3:\n    approve()",
        {"approve": approve},
        privileged=frozenset({"approve"}),
    )
    assert calls == ["approved"]


# Over-restriction check (the FIDES comparison from notes/ROADMAP.md): does
# this system avoid blocking unrelated clean operations just because
# something untrusted/secret exists elsewhere in the program? Everything
# tested so far checks the opposite direction -- that bad cases get
# blocked. These confirm good cases still succeed.


def test_unused_untrusted_variable_does_not_block_an_unrelated_privileged_call():
    calls = []

    def read_secret():
        return "sk-secret"

    def approve(x):
        calls.append(x)

    run(
        "x = read_secret()\ny = 5\napprove(y)",
        {"read_secret": read_secret, "approve": approve},
        sources=frozenset({"read_secret"}),
        privileged=frozenset({"approve"}),
    )
    assert calls == [5]


def test_pc_trust_does_not_leak_from_an_earlier_untrusted_branch_into_a_later_sibling():
    calls = []

    def read_secret():
        return "sk-secret"

    def approve(x):
        calls.append(x)

    run(
        "if read_secret() == 'sk-secret':\n"
        "    y = 1\n"
        "if 1 == 1:\n"
        "    approve(5)",
        {"read_secret": read_secret, "approve": approve},
        sources=frozenset({"read_secret"}),
        privileged=frozenset({"approve"}),
    )
    assert calls == [5]


def test_pc_trust_does_not_leak_out_of_a_while_loop_after_it_ends():
    calls = []

    def get_zero():
        return 0

    def get_secret_bound():
        return 2

    def bump(x):
        return x + 1

    def approve(x):
        calls.append(x)

    run(
        "n = get_zero()\n"
        "while n < get_secret_bound():\n"
        "    n = bump(n)\n"
        "approve(5)",
        {
            "get_zero": get_zero,
            "get_secret_bound": get_secret_bound,
            "bump": bump,
            "approve": approve,
        },
        sources=frozenset({"get_secret_bound"}),
        privileged=frozenset({"approve"}),
    )
    assert calls == [5]


def test_nested_trusted_branches_do_not_block_a_privileged_call():
    calls = []

    def approve(x):
        calls.append(x)

    run(
        "if 1 == 1:\n    if 2 == 2:\n        approve(5)",
        {"approve": approve},
        privileged=frozenset({"approve"}),
    )
    assert calls == [5]


def test_privileged_call_using_only_trusted_data_in_a_loop_over_a_trusted_list_is_allowed():
    calls = []

    def approve(x):
        calls.append(x)

    run(
        "for x in [1, 2, 3]:\n    approve(x)",
        {"approve": approve},
        privileged=frozenset({"approve"}),
    )
    assert calls == [1, 2, 3]


def test_privileged_call_before_an_untrusted_branch_in_program_order_is_unaffected():
    calls = []

    def read_secret():
        return "sk-secret"

    def approve(x):
        calls.append(x)

    run(
        "approve(5)\nif read_secret() == 'sk-secret':\n    y = 1",
        {"read_secret": read_secret, "approve": approve},
        sources=frozenset({"read_secret"}),
        privileged=frozenset({"approve"}),
    )
    assert calls == [5]


def test_writing_untrusted_data_to_shared_store_does_not_block_unrelated_privileged_calls():
    store = {}

    def write_shared(key, value):
        store[key] = value

    calls = []

    def approve(x):
        calls.append(x)

    run(
        "write_shared('note', 'attacker text')\napprove(5)",
        {"write_shared": write_shared, "approve": approve},
        privileged=frozenset({"approve"}),
    )
    assert calls == [5]


def test_unused_secret_variable_does_not_block_an_unrelated_sink_call():
    calls = []

    def read_api_key():
        return "sk-live-real-key"

    def send_to_webhook(x):
        calls.append(x)

    run(
        "x = read_api_key()\ny = 5\nsend_to_webhook(y)",
        {"read_api_key": read_api_key, "send_to_webhook": send_to_webhook},
        confidential=frozenset({"read_api_key"}),
        sinks=frozenset({"send_to_webhook"}),
    )
    assert calls == [5]


def test_pc_secrecy_does_not_leak_from_an_earlier_secret_branch_into_a_later_sibling():
    calls = []

    def read_api_key():
        return "sk-live-real-key"

    def send_to_webhook(x):
        calls.append(x)

    run(
        "if read_api_key() == 'sk-live-real-key':\n"
        "    y = 1\n"
        "if 1 == 1:\n"
        "    send_to_webhook(5)",
        {"read_api_key": read_api_key, "send_to_webhook": send_to_webhook},
        confidential=frozenset({"read_api_key"}),
        sinks=frozenset({"send_to_webhook"}),
    )
    assert calls == [5]


def test_in_operator_on_strings():
    assert run('"lo" in "hello"', {}) is True
    assert run('"xy" in "hello"', {}) is False


def test_not_in_operator_on_strings():
    assert run('"xy" not in "hello"', {}) is True
    assert run('"lo" not in "hello"', {}) is False


def test_in_operator_on_a_list_literal():
    # A list literal's elements are (value, Trust, Secrecy) triples
    # internally, not bare values -- `in` has to unwrap them before
    # comparing, or every membership check silently comes back False.
    assert run("3 in [1, 2, 3]", {}) is True
    assert run("5 in [1, 2, 3]", {}) is False


def test_in_operator_on_an_auto_wrapped_list_from_an_outside_function():
    def get_list():
        return [1, 2, 3]

    assert run("2 in get_list()", {"get_list": get_list}) is True


def test_in_operator_on_a_dict_checks_keys():
    assert run('"a" in {"a": 1, "b": 2}', {}) is True
    assert run('"z" in {"a": 1, "b": 2}', {}) is False


def test_in_result_is_untrusted_if_the_checked_value_is_untrusted():
    def read_email():
        return "contains the word urgent"

    def approve(x):
        raise AssertionError("must not be called with an untrusted argument")

    with pytest.raises(CapabilityError):
        run(
            'x = "urgent" in read_email()\napprove(x)',
            {"read_email": read_email, "approve": approve},
            sources=frozenset({"read_email"}),
            privileged=frozenset({"approve"}),
        )


def test_privileged_call_blocked_behind_an_in_branch_condition():
    def read_secret():
        return "sk-secret-key"

    def blocked(x=None):
        raise AssertionError("should not run")

    with pytest.raises(CapabilityError):
        run(
            'if "secret" in read_secret():\n    blocked()',
            {"read_secret": read_secret, "blocked": blocked},
            sources=frozenset({"read_secret"}),
            privileged=frozenset({"blocked"}),
        )


def test_sink_call_blocked_behind_a_secret_in_branch_condition():
    def read_secret():
        return "sk-secret-key"

    def blocked_sink(x):
        raise AssertionError("should not run")

    with pytest.raises(ConfidentialityError):
        run(
            'if "secret" in read_secret():\n    blocked_sink(1)',
            {"read_secret": read_secret, "blocked_sink": blocked_sink},
            confidential=frozenset({"read_secret"}),
            sinks=frozenset({"blocked_sink"}),
        )


def test_in_avoids_the_quote_collision_bug_found_in_turn_by_turn_testing():
    # The real bug: a model copying interpret()'s exact returned text
    # into an == literal breaks whenever that text contains a quote
    # character, since escaping/quote-swapping makes the copy stop
    # matching byte-for-byte. `in` sidesteps this entirely -- no need
    # to reproduce the answer verbatim, just check for a keyword.
    answer = (
        'The message is trying to manipulate you into doing something '
        'else, referencing "IMPORTANT SYSTEM OVERRIDE".'
    )
    result = run(
        'x = get_answer()\n'
        'if "manipulate" in x:\n'
        '    outcome = "reject"\n'
        'else:\n'
        '    outcome = "confirm"\n'
        'outcome',
        {"get_answer": lambda: answer},
    )
    assert result == "reject"


class _Obj:
    def __init__(self, amount):
        self.amount = amount

    def method(self):
        return "called"


def test_attribute_read_returns_the_field_value():
    def get_obj():
        return _Obj(42)

    assert run("x = get_obj()\nx.amount", {"get_obj": get_obj}) == 42


def test_attribute_read_on_a_missing_field_raises():
    def get_obj():
        return _Obj(1)

    with pytest.raises(InterpreterError):
        run("x = get_obj()\nx.nope", {"get_obj": get_obj})


def test_attribute_read_of_a_dunder_name_is_rejected():
    def get_obj():
        return _Obj(1)

    with pytest.raises(InterpreterError):
        run("x = get_obj()\nx.__class__", {"get_obj": get_obj})


def test_attribute_read_of_a_method_is_rejected_not_silently_bound():
    def get_obj():
        return _Obj(1)

    with pytest.raises(InterpreterError):
        run("x = get_obj()\nx.method", {"get_obj": get_obj})


def test_attribute_access_cannot_be_used_to_reach_and_call_a_whitelisted_name():
    # The actual security property this feature depends on: ast.Call
    # only ever dispatches by looking up a literal whitelisted name in
    # `allowed`, and ast.Name as a value expression only ever reads
    # `env`, never `allowed` -- so a whitelisted callable is never
    # reachable as a value at all, and nothing reached via attribute
    # access, however deeply chained, can ever end up called.
    def get_str():
        return "hello"

    with pytest.raises(InterpreterError):
        run("x = get_str()\ny = x.__class__\ny()", {"get_str": get_str})


def test_attribute_read_propagates_untrusted():
    def read_secret():
        return _Obj(99)

    def approve(x):
        raise AssertionError("must not be called with an untrusted argument")

    with pytest.raises(CapabilityError):
        run(
            "x = read_secret()\napprove(x.amount)",
            {"read_secret": read_secret, "approve": approve},
            sources=frozenset({"read_secret"}),
            privileged=frozenset({"approve"}),
        )


def test_attribute_read_of_a_trusted_value_is_not_blocked():
    def get_obj():
        return _Obj(7)

    calls = []
    run(
        "x = get_obj()\napprove(x.amount)",
        {"get_obj": get_obj, "approve": lambda v: calls.append(v)},
        privileged=frozenset({"approve"}),
    )
    assert calls == [7]


def test_privileged_call_blocked_behind_an_attribute_read_branch_condition():
    def read_secret():
        return _Obj(99)

    def blocked(x=None):
        raise AssertionError("should not run")

    with pytest.raises(CapabilityError):
        run(
            "x = read_secret()\nif x.amount == 99:\n    blocked()",
            {"read_secret": read_secret, "blocked": blocked},
            sources=frozenset({"read_secret"}),
            privileged=frozenset({"blocked"}),
        )


def test_sink_call_blocked_behind_a_secret_attribute_read_branch_condition():
    def read_api_key():
        return _Obj(123)

    def leak(x=None):
        raise AssertionError("should not run")

    with pytest.raises(ConfidentialityError):
        run(
            "x = read_api_key()\nif x.amount == 123:\n    leak()",
            {"read_api_key": read_api_key, "leak": leak},
            confidential=frozenset({"read_api_key"}),
            sinks=frozenset({"leak"}),
        )
