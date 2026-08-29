"""Deterministic tests for the whitelisted string-method calls
(_STRING_METHODS in prompt_lang/interpreter.py), startswith,
endswith, strip, lower/upper, replace, split, find, count. Split out
of the original test_interpreter.py; see test_interpreter_core.py's
own docstring for why.
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


def test_string_method_on_a_variable_not_just_a_literal():
    assert run('d = "2022-03-15"\nd.startswith("2022-03")', {}) is True


def test_string_method_result_is_usable_in_further_expressions():
    assert run('"  HELLO  ".strip().lower()', {}) == "hello"


def test_string_method_on_an_attribute_read_from_a_real_object():
    class Record:
        def __init__(self, date):
            self.date = date

    def get_record():
        return Record("2022-03-07")

    assert run(
        "get_record().date.startswith('2022-03')",
        {"get_record": get_record},
    ) is True


def test_method_call_on_a_non_string_value_is_rejected():
    # The actual security boundary: string methods must never become a
    # general "call any method on any object" escape hatch. A tagged
    # list is a plain Python list internally, so without this check
    # `.append()` could mutate it in place and corrupt the tagged-triple
    # invariant directly.
    with pytest.raises(InterpreterError):
        run("x = [1, 2, 3]\nx.append(4)", {})


def test_method_call_on_a_non_string_value_is_rejected_even_when_the_name_is_whitelisted():
    # A weaker version of the isinstance(str) check, one that only
    # filters by method *name*, would silently pass here: "count" is a
    # real method on both str and list, so a name-only whitelist can't
    # tell these apart. Confirmed live (before this test existed) that
    # removing the isinstance check specifically lets this one through
    # silently, returning a wrong answer (0) instead of raising, since a
    # tagged list is really a list of (value, Trust, Secrecy) triples
    # internally, `.count(2)` searches for a bare 2 that can never
    # match. This is the actual case that exercises the isinstance
    # check, not just an unlisted method name.
    with pytest.raises(InterpreterError):
        run("x = [1, 2, 2, 3]\nx.count(2)", {})


def test_method_call_on_an_int_is_rejected():
    with pytest.raises(InterpreterError):
        run("(5).bit_length()", {})


def test_unsupported_string_method_is_rejected():
    with pytest.raises(InterpreterError):
        run('"abc".format(1)', {})


def test_dunder_style_name_is_rejected_as_an_unsupported_method():
    with pytest.raises(InterpreterError):
        run('"abc".__class__', {})


def test_string_method_with_keyword_argument_is_rejected():
    with pytest.raises(InterpreterError):
        run('"abc".replace(old="a", new="b")', {})


def test_string_method_wrong_arg_count_raises_interpreter_error_not_a_raw_python_error():
    with pytest.raises(InterpreterError):
        run('"abc".replace("a")', {})


def test_string_method_trust_propagates_from_the_receiver():
    def read_untrusted():
        return "attacker payload"

    def privileged_action(*args, **kwargs):
        return "ok"

    with pytest.raises(CapabilityError):
        run(
            "x = read_untrusted()\ny = x.upper()\nprivileged_action(y)",
            {"read_untrusted": read_untrusted, "privileged_action": privileged_action},
            sources=frozenset({"read_untrusted"}),
            privileged=frozenset({"privileged_action"}),
        )


def test_string_method_trust_propagates_from_an_argument_too():
    # The receiver itself is trusted; the *argument* to replace() is the
    # untrusted one, trust must combine both operands, not just the
    # receiver, the same rule ast.BinOp/ast.Compare already follow.
    def read_untrusted():
        return "x"

    def privileged_action(*args, **kwargs):
        return "ok"

    with pytest.raises(CapabilityError):
        run(
            "needle = read_untrusted()\nresult = 'trusted text'.replace(needle, 'y')\nprivileged_action(result)",
            {"read_untrusted": read_untrusted, "privileged_action": privileged_action},
            sources=frozenset({"read_untrusted"}),
            privileged=frozenset({"privileged_action"}),
        )


def test_string_method_secrecy_propagates_from_the_receiver():
    def read_secret():
        return "sk-secret"

    def sink_action(*args, **kwargs):
        return "ok"

    with pytest.raises(ConfidentialityError):
        run(
            "s = read_secret()\nu = s.lower()\nsink_action(u)",
            {"read_secret": read_secret, "sink_action": sink_action},
            confidential=frozenset({"read_secret"}),
            sinks=frozenset({"sink_action"}),
        )


def test_string_method_on_a_trusted_value_is_never_blocked():
    def privileged_action(*args, **kwargs):
        return "ok"

    calls = []
    run(
        "privileged_action('HELLO'.lower())",
        {"privileged_action": lambda x: calls.append(x)},
        privileged=frozenset({"privileged_action"}),
    )
    assert calls == ["hello"]
