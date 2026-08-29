"""Deterministic tests for ternary expressions (x if y else z) and
user-defined function definitions (def name(params): body), including
both constructs' own pc_trust/pc_secrecy implicit-flow protection and
function definitions' recursion guard. Split out of the original
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


def test_ternary_picks_the_true_branch():
    result = run("5 if True else 10", {})
    assert result == 5


def test_ternary_picks_the_false_branch():
    result = run("5 if False else 10", {})
    assert result == 10


def test_ternary_is_lazy_the_unchosen_branch_never_executes():
    def blow_up():
        raise AssertionError("unchosen branch must not be evaluated")

    result = run("1 if True else blow_up()", {"blow_up": blow_up})
    assert result == 1


def test_ternary_result_from_the_chosen_branch_keeps_its_own_trust():
    def read_secret():
        return "untrusted"

    def blocked(x=None):
        raise AssertionError("should not run")

    with pytest.raises(CapabilityError):
        run(
            "x = read_secret() if True else 1\nblocked(x)",
            {"read_secret": read_secret, "blocked": blocked},
            sources=frozenset({"read_secret"}),
            privileged=frozenset({"blocked"}),
        )


def test_privileged_call_blocked_behind_an_untrusted_ternary_test():
    # The actual security question this addition raises: a privileged
    # call hiding inside a ternary branch, gated by an untrusted test,
    # must be caught the same way ast.If's own pc_trust already catches
    # it for the statement form, ast.IfExp is a different node and
    # was never covered by that just because ast.If exists.
    def read_secret():
        return "untrusted"

    def blocked(x=None):
        raise AssertionError("should not run")

    with pytest.raises(CapabilityError):
        run(
            'x = read_secret()\nblocked() if x == "untrusted" else None',
            {"read_secret": read_secret, "blocked": blocked},
            sources=frozenset({"read_secret"}),
            privileged=frozenset({"blocked"}),
        )


def test_sink_call_blocked_behind_a_secret_ternary_test():
    def read_api_key():
        return "sk-secret"

    def leak(x=None):
        raise AssertionError("should not run")

    with pytest.raises(ConfidentialityError):
        run(
            'x = read_api_key()\nleak() if x == "sk-secret" else None',
            {"read_api_key": read_api_key, "leak": leak},
            confidential=frozenset({"read_api_key"}),
            sinks=frozenset({"leak"}),
        )


def test_privileged_call_in_the_unchosen_ternary_branch_is_never_reached_so_never_blocked():
    # Mirrors the lazy-evaluation guarantee: a privileged call sitting
    # in the branch that ISN'T taken must not raise at all, since it
    # never actually runs, confirms blocking is about reachability,
    # not just syntactic presence in the source. Uses a trusted literal
    # test so this is cleanly about laziness alone, not entangled with
    # pc_trust gating a branch that actually gets reached (covered
    # separately above).
    calls = []

    def approve(x=None):
        calls.append("ran")
        return "should not run"

    result = run(
        "approve() if False else 2",
        {"approve": approve},
        privileged=frozenset({"approve"}),
    )
    assert result == 2
    assert calls == []


def test_ternary_does_not_retroactively_taint_a_value_chosen_via_an_untrusted_test():
    # Consistency with ast.If's own documented choice: picking between
    # two already-safe values based on an untrusted condition doesn't
    # itself taint the result, matches ast.If not retroactively
    # tainting a plain assignment made inside a branch.
    def read_secret():
        return "untrusted"

    def approve(x=None):
        return "ok"

    result = run(
        'x = read_secret()\ny = "safe1" if x == "untrusted" else "safe2"\napprove(y)',
        {"read_secret": read_secret, "approve": approve},
        sources=frozenset({"read_secret"}),
        privileged=frozenset({"approve"}),
    )
    assert result == "ok"


def test_ternary_composes_as_a_call_argument():
    def report(x):
        return x

    result = run('report(1 if True else 2)', {"report": report})
    assert result == 1


def test_ternary_test_condition_evaluates_only_once():
    calls = []

    def check():
        calls.append("checked")
        return True

    run("1 if check() else 2", {"check": check})
    assert calls == ["checked"]


def test_function_definition_and_call_basic():
    result = run("def add(a, b):\n    a + b\nadd(2, 3)", {})
    assert result == 5


def test_function_with_no_parameters():
    result = run("def get_five():\n    5\nget_five()", {})
    assert result == 5


def test_function_body_returns_the_last_statements_value():
    result = run(
        "def compute(x):\n    y = x + 1\n    y + 1\ncompute(1)", {}
    )
    assert result == 3


def test_function_whose_last_statement_is_an_assignment_returns_none():
    result = run("def noop():\n    x = 1\nnoop()", {})
    assert result is None


def test_function_result_keeps_the_trust_of_what_it_actually_returns():
    def read_secret():
        return "untrusted"

    def blocked(x=None):
        raise AssertionError("should not run")

    with pytest.raises(CapabilityError):
        run(
            "def get_it():\n    read_secret()\nx = get_it()\nblocked(x)",
            {"read_secret": read_secret, "blocked": blocked},
            sources=frozenset({"read_secret"}),
            privileged=frozenset({"blocked"}),
        )


def test_untrusted_argument_stays_tagged_inside_the_function_body():
    def read_secret():
        return "untrusted"

    def blocked(x=None):
        raise AssertionError("should not run")

    with pytest.raises(CapabilityError):
        run(
            "def forward(x):\n    blocked(x)\ny = read_secret()\nforward(y)",
            {"read_secret": read_secret, "blocked": blocked},
            sources=frozenset({"read_secret"}),
            privileged=frozenset({"blocked"}),
        )


def test_privileged_call_inside_a_function_reached_via_an_untrusted_branch_is_blocked():
    # The actual security question this addition raises: a function
    # call is a continuation of the caller's own control-flow context,
    # not a fresh entry point, if the function body started fresh at
    # pc_trust=TRUSTED regardless of the untrusted branch that reached
    # it, an unconditional privileged call inside would run unblocked,
    # reopening the exact bug pc_trust exists to close, through a new
    # door.
    def read_secret():
        return "untrusted"

    def blocked():
        raise AssertionError("should not run")

    with pytest.raises(CapabilityError):
        run(
            'def my_func():\n    blocked()\nx = read_secret()\nif x == "untrusted":\n    my_func()',
            {"read_secret": read_secret, "blocked": blocked},
            sources=frozenset({"read_secret"}),
            privileged=frozenset({"blocked"}),
        )


def test_sink_call_inside_a_function_reached_via_a_secret_branch_is_blocked():
    def read_api_key():
        return "sk-secret"

    def leak():
        raise AssertionError("should not run")

    with pytest.raises(ConfidentialityError):
        run(
            'def my_func():\n    leak()\nx = read_api_key()\nif x == "sk-secret":\n    my_func()',
            {"read_api_key": read_api_key, "leak": leak},
            confidential=frozenset({"read_api_key"}),
            sinks=frozenset({"leak"}),
        )


def test_direct_recursion_is_rejected():
    with pytest.raises(InterpreterError):
        run("def f(n):\n    f(n)\nf(1)", {})


def test_mutual_recursion_is_rejected():
    with pytest.raises(InterpreterError):
        run("def f(n):\n    g(n)\ndef g(n):\n    f(n)\nf(1)", {})


def test_sequential_non_nested_calls_to_the_same_function_are_not_flagged_as_recursion():
    result = run("def double(x):\n    x + x\ndouble(1)\ndouble(2)", {})
    assert result == 4


def test_a_function_can_call_another_already_defined_function():
    result = run(
        "def inc(x):\n    x + 1\ndef inc_twice(x):\n    inc(inc(x))\ninc_twice(1)", {}
    )
    assert result == 3


def test_function_body_cannot_see_the_callers_outer_variables():
    with pytest.raises(InterpreterError):
        run("x = 5\ndef f():\n    x\nf()", {})


def test_function_name_colliding_with_a_whitelisted_name_is_rejected():
    def send_money():
        return "real"

    with pytest.raises(InterpreterError):
        run(
            "def send_money():\n    1\nsend_money()",
            {"send_money": send_money},
            privileged=frozenset({"send_money"}),
        )


def test_too_many_positional_arguments_is_rejected():
    with pytest.raises(InterpreterError):
        run("def f(a):\n    a\nf(1, 2)", {})


def test_missing_required_argument_is_rejected():
    with pytest.raises(InterpreterError):
        run("def f(a, b):\n    a\nf(1)", {})


def test_unexpected_keyword_argument_is_rejected():
    with pytest.raises(InterpreterError):
        run("def f(a):\n    a\nf(a=1, b=2)", {})


def test_duplicate_argument_via_positional_and_keyword_is_rejected():
    with pytest.raises(InterpreterError):
        run("def f(a):\n    a\nf(1, a=2)", {})


def test_function_call_by_keyword_argument_works():
    result = run("def f(a, b):\n    a - b\nf(b=1, a=5)", {})
    assert result == 4


def test_function_with_default_arguments_is_rejected():
    with pytest.raises(InterpreterError):
        run("def f(a=1):\n    a\nf()", {})


def test_function_with_star_args_is_rejected():
    with pytest.raises(InterpreterError):
        run("def f(*a):\n    1\nf()", {})


def test_function_with_star_kwargs_is_rejected():
    with pytest.raises(InterpreterError):
        run("def f(**a):\n    1\nf()", {})


def test_function_with_decorator_is_rejected():
    with pytest.raises(InterpreterError):
        run("@staticmethod\ndef f():\n    1\nf()", {})


def test_calling_an_undefined_name_still_raises_the_original_error():
    with pytest.raises(InterpreterError):
        run("undefined_thing()", {})


def test_loop_inside_a_function_body_is_still_bounded_by_the_per_loop_cap():
    # Confirms the shared iteration budget and MAX_WHILE_ITERATIONS
    # threading actually reach inside a function body, if they
    # didn't, this loop would run to 10000 instead of being capped.
    with pytest.raises(InterpreterError):
        run(
            "def spin():\n    i = 0\n    while i < 10000:\n        i = i + 1\nspin()",
            {},
        )


def test_string_startswith_true_case():
    assert run('"2022-03-15".startswith("2022-03")', {}) is True


def test_string_startswith_false_case():
    assert run('"2022-02-15".startswith("2022-03")', {}) is False


def test_string_endswith():
    assert run('"report.txt".endswith(".txt")', {}) is True


def test_string_strip_variants():
    assert run('"  hi  ".strip()', {}) == "hi"
    assert run('"  hi  ".lstrip()', {}) == "hi  "
    assert run('"  hi  ".rstrip()', {}) == "  hi"


def test_string_lower_and_upper():
    assert run('"HeLLo".lower()', {}) == "hello"
    assert run('"HeLLo".upper()', {}) == "HELLO"


def test_string_replace():
    assert run('"2022-03-01".replace("-", "/")', {}) == "2022/03/01"


def test_string_find_and_count():
    assert run('"banana".find("na")', {}) == 2
    assert run('"banana".count("na")', {}) == 2


def test_string_split_returns_a_usable_list():
    # split()'s result must come back as a real usable list, not just
    # the right value, but actually indexable/iterable the way any other
    # list literal is, confirming the auto-wrap path applied correctly.
    assert run('"a,b,c".split(",")[1]', {}) == "b"


