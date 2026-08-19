"""Tests for prompt_lang/handles.py -- opaque-handle confinement
adapted from SecureClaw's read-path design (arXiv:2606.09549). These
are deterministic; describe_handle()'s `ask` dependency is a fake here,
not a live model call, matching how prompt_lang.tools.interpret() would
be faked in a test."""

import pytest
from prompt_lang.handles import (
    Handle,
    HandleAccessDenied,
    HandleStore,
    make_describe_handle,
    wrap_for_opaque_handles,
    wrap_privileged_for_handles,
    wrap_sources_for_handles,
)


def test_mint_returns_an_opaque_handle_not_the_value():
    store = HandleStore()
    handle = store.mint("US133000000121212121212")
    assert isinstance(handle, Handle)
    assert handle.id != "US133000000121212121212"
    assert "US133" not in handle.id


def test_resolve_returns_the_real_value_when_authorized():
    store = HandleStore()
    handle = store.mint("secret value")
    assert store.resolve(handle, "any_sink") == "secret value"


def test_resolve_denies_a_sink_not_in_the_handles_policy():
    store = HandleStore()
    handle = store.mint("secret value", allowed_sinks=frozenset({"send_money"}))
    with pytest.raises(HandleAccessDenied):
        store.resolve(handle, "post_to_forum")
    assert store.resolve(handle, "send_money") == "secret value"


def test_resolve_denies_an_unknown_or_forged_handle():
    store = HandleStore()
    forged = Handle(id="not-a-real-id")
    with pytest.raises(HandleAccessDenied):
        store.resolve(forged, "any_sink")


def test_two_handles_from_the_same_store_have_different_ids():
    store = HandleStore()
    a = store.mint("value one")
    b = store.mint("value two")
    assert a.id != b.id


def test_wrap_sources_for_handles_mints_instead_of_returning_raw_value():
    store = HandleStore()

    def read_bill():
        return "Invoice with IBAN US133000000121212121212"

    wrapped = wrap_sources_for_handles(
        {"read_bill": read_bill}, sources=frozenset({"read_bill"}), store=store
    )
    result = wrapped["read_bill"]()
    assert isinstance(result, Handle)
    assert store.peek(result) == "Invoice with IBAN US133000000121212121212"


def test_wrap_sources_for_handles_leaves_unrelated_functions_untouched():
    store = HandleStore()

    def approve():
        return "approved"

    wrapped = wrap_sources_for_handles({"approve": approve}, sources=frozenset(), store=store)
    assert wrapped["approve"] is approve


def test_wrap_privileged_for_handles_resolves_a_handle_argument_transparently():
    store = HandleStore()
    handle = store.mint("US133000000121212121212")
    calls = []

    def send_money(recipient, amount):
        calls.append((recipient, amount))
        return {"message": f"sent {amount} to {recipient}"}

    wrapped = wrap_privileged_for_handles(
        {"send_money": send_money}, privileged=frozenset({"send_money"}),
        sinks=frozenset(), store=store,
    )
    result = wrapped["send_money"](recipient=handle, amount=500)
    assert calls == [("US133000000121212121212", 500)]
    assert result["message"] == "sent 500 to US133000000121212121212"


def test_wrap_privileged_for_handles_passes_plain_arguments_through_unchanged():
    store = HandleStore()

    def send_money(recipient, amount):
        return {"recipient": recipient, "amount": amount}

    wrapped = wrap_privileged_for_handles(
        {"send_money": send_money}, privileged=frozenset({"send_money"}),
        sinks=frozenset(), store=store,
    )
    result = wrapped["send_money"](recipient="a_plain_string", amount=10)
    assert result == {"recipient": "a_plain_string", "amount": 10}


def test_wrap_privileged_for_handles_enforces_the_handles_own_sink_policy():
    store = HandleStore()
    handle = store.mint("secret", allowed_sinks=frozenset({"send_money"}))

    def post_to_forum(body):
        return "posted"

    wrapped = wrap_privileged_for_handles(
        {"post_to_forum": post_to_forum}, privileged=frozenset(),
        sinks=frozenset({"post_to_forum"}), store=store,
    )
    with pytest.raises(HandleAccessDenied):
        wrapped["post_to_forum"](body=handle)


def test_wrap_for_opaque_handles_end_to_end_read_and_dereference():
    store = HandleStore()
    sent = []

    def read_bill():
        return "US133000000121212121212"

    def send_money(recipient, amount):
        sent.append((recipient, amount))
        return "ok"

    allowed = wrap_for_opaque_handles(
        {"read_bill": read_bill, "send_money": send_money},
        sources=frozenset({"read_bill"}),
        privileged=frozenset({"send_money"}),
        sinks=frozenset(),
        store=store,
    )
    handle = allowed["read_bill"]()
    assert isinstance(handle, Handle)
    allowed["send_money"](recipient=handle, amount=100)
    assert sent == [("US133000000121212121212", 100)]


def test_wrap_for_opaque_handles_a_forged_handle_id_is_still_denied():
    # The whole point: the runtime can carry a handle through code, but
    # can never itself construct one that resolves to something real --
    # a model "guessing" a handle id (or retyping one) gets nothing.
    store = HandleStore()

    def read_bill():
        return "real content"

    def send_money(recipient):
        return recipient

    allowed = wrap_for_opaque_handles(
        {"read_bill": read_bill, "send_money": send_money},
        sources=frozenset({"read_bill"}),
        privileged=frozenset({"send_money"}),
        sinks=frozenset(),
        store=store,
    )
    allowed["read_bill"]()  # mints a real handle, but we use a forged one below
    forged = Handle(id="deadbeef" * 4)
    with pytest.raises(HandleAccessDenied):
        allowed["send_money"](recipient=forged)


def test_describe_handle_answers_using_the_real_value_via_injected_ask():
    store = HandleStore()
    handle = store.mint("Invoice for 98.70, IBAN US133000000121212121212")
    seen = {}

    def fake_ask(text: str, question: str) -> str:
        seen["text"] = text
        seen["question"] = question
        return "The amount is 98.70"

    describe_handle = make_describe_handle(store, ask=fake_ask)
    answer = describe_handle(handle, "what is the amount?")
    assert answer == "The amount is 98.70"
    assert seen["text"] == "Invoice for 98.70, IBAN US133000000121212121212"
    assert seen["question"] == "what is the amount?"


def test_describe_handle_truncates_to_the_character_cap():
    store = HandleStore()
    handle = store.mint("some content")

    def verbose_ask(text: str, question: str) -> str:
        return "x" * 500

    describe_handle = make_describe_handle(store, ask=verbose_ask, max_chars=50)
    answer = describe_handle(handle, "describe everything")
    assert len(answer) == 50


def test_describe_handle_denies_an_unknown_handle():
    store = HandleStore()
    describe_handle = make_describe_handle(store, ask=lambda t, q: "n/a")
    with pytest.raises(HandleAccessDenied):
        describe_handle(Handle(id="unknown"), "what is this?")
