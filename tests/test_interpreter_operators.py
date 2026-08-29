"""Deterministic tests for the arithmetic, comparison, boolean, and
unary operators, chained comparisons, short-circuit evaluation,
the MAX_EXPONENT bound, and unary minus. Split out of the original
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
    # Not caught or converted, the whitelist boundary is about what's
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
    # up Compare's, an arithmetic expression used directly as an if
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
# +/-/*/ additions above: correctness, capability propagation, and a
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
    # The guard is on the exponent's magnitude, not the base's, a large
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


# Over-restriction check (the same comparison FIDES-style systems get
# judged on): does this system avoid blocking unrelated clean
# operations just because something untrusted/secret exists elsewhere
# in the program? Everything tested so far checks the opposite
# direction, that bad cases get blocked. These confirm good cases
# still succeed.


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
    # internally, not bare values, `in` has to unwrap them before
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
    # matching byte-for-byte. `in` sidesteps this entirely, no need
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
    # `env`, never `allowed`, so a whitelisted callable is never
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


