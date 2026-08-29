"""Provider-agnostic model calling. Factored out of experiments/, where
every harness (feasibility_live.py, turn_by_turn_live.py, agentdojo_live.py,
overhead_measurement.py) had grown its own copy of the same
requests.post-to-Ollama pattern. Pure plumbing: nothing here touches
the interpreter, the grammar, or the capability system, and none of
that changes when this does.

Every provider function takes the same shape: one flat prompt string in,
one completion string out. That matches how every harness in this
project already treats a "turn": a single prompt, not a structured,
multi-message conversation, so adding a provider means matching that
shape, not changing how callers use it.
"""

from __future__ import annotations

import requests


def call_ollama(prompt: str, model: str, host: str = "http://localhost:11434", timeout: float = 120.0) -> str:
    """Ollama's own /api/generate endpoint. The one provider actually
    exercised against a live model in this project, every recorded
    turn-by-turn, AgentDojo, and overhead-measurement result went
    through this logic."""
    resp = requests.post(
        host.rstrip("/") + "/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def call_openai(prompt: str, model: str, timeout: float = 120.0) -> str:
    """OpenAI's chat completions API, given the flat prompt as a single
    user message. Built to the real SDK's documented interface, but not
    live-verified against a real endpoint in this project, no
    OPENAI_API_KEY is configured in this environment. Requires
    `pip install openai` and that key set before it will actually run.
    """
    import openai

    client = openai.OpenAI(timeout=timeout)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content or ""


def call_anthropic(prompt: str, model: str, timeout: float = 120.0, max_tokens: int = 4096) -> str:
    """Anthropic's messages API, same single-prompt-as-one-user-message
    shape as call_openai. Same caveat: built to the real SDK's
    documented interface, not live-verified here, no ANTHROPIC_API_KEY
    is configured in this environment. Requires `pip install anthropic`
    and that key set before it will actually run.
    """
    import anthropic

    client = anthropic.Anthropic(timeout=timeout)
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in resp.content if block.type == "text")


PROVIDERS = {
    "ollama": call_ollama,
    "openai": call_openai,
    "anthropic": call_anthropic,
}


def call_model(prompt: str, model: str, provider: str = "ollama", **kwargs) -> str:
    """Single entrypoint harnesses should call instead of hand-rolling
    their own request. `provider` must be one of PROVIDERS' keys.
    Anything else raises immediately, rather than silently defaulting
    to Ollama and producing a confusing downstream connection
    error."""
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}, expected one of {sorted(PROVIDERS)}")
    return PROVIDERS[provider](prompt, model, **kwargs)
