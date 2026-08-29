"""Deterministic tests for prompt_lang/models.py. Every network call is
mocked, these check dispatch and request/response shape, not that a
real model gives good answers (see experiments/ for that). Ollama is
the one provider actually exercised live this project; openai/anthropic
are checked structurally only, since no API key is configured here."""

from unittest.mock import MagicMock, patch

import pytest
from prompt_lang.models import PROVIDERS, call_model, call_ollama


def _mock_ollama_response(text: str) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = {"response": text}
    resp.raise_for_status.return_value = None
    return resp


@patch("prompt_lang.models.requests.post")
def test_call_ollama_sends_model_and_prompt_and_returns_response(mock_post):
    mock_post.return_value = _mock_ollama_response("hello")
    result = call_ollama("say hi", "qwen2.5:32b")
    assert result == "hello"
    call_kwargs = mock_post.call_args.kwargs
    assert call_kwargs["json"]["model"] == "qwen2.5:32b"
    assert call_kwargs["json"]["prompt"] == "say hi"
    assert call_kwargs["json"]["stream"] is False


@patch("prompt_lang.models.requests.post")
def test_call_ollama_host_controls_the_endpoint(mock_post):
    mock_post.return_value = _mock_ollama_response("hi")
    call_ollama("x", "m", host="http://example.com:11434")
    assert mock_post.call_args.args[0] == "http://example.com:11434/api/generate"


@patch("prompt_lang.models.requests.post")
def test_call_ollama_strips_trailing_slash_from_host(mock_post):
    mock_post.return_value = _mock_ollama_response("hi")
    call_ollama("x", "m", host="http://example.com:11434/")
    assert mock_post.call_args.args[0] == "http://example.com:11434/api/generate"


@patch("prompt_lang.models.requests.post")
def test_call_model_dispatches_to_ollama_by_default(mock_post):
    mock_post.return_value = _mock_ollama_response("dispatched")
    assert call_model("hi", "m") == "dispatched"


def test_call_model_rejects_unknown_provider():
    with pytest.raises(ValueError, match="unknown provider"):
        call_model("hi", "m", provider="not-a-real-provider")


def test_providers_registry_has_all_three():
    assert set(PROVIDERS) == {"ollama", "openai", "anthropic"}
