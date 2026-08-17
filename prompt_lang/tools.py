"""Real tool functions meant to be wired into a program's `allowed` dict.
Unlike interpreter.py, which never depends on anything outside the small
language it defines, these are ordinary Python functions that do real
I/O -- the kind of thing an actual deployment wires in, not a stub.
"""

from __future__ import annotations

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_INTERPRET_MODEL = "llama3.2:3b"


def interpret(text: str, question: str, model: str = DEFAULT_INTERPRET_MODEL) -> str:
    """Asks a local model to answer `question` using only `text`. Meant
    to be registered as an ordinary (non-sanitizer) entry in a program's
    `allowed` dict -- deliberately not a sanitizer, and that's a
    security-relevant choice, not an oversight. A model reading
    untrusted text to extract or summarize something is reading the
    same content an injected instruction could be hiding inside; it is
    not a trust boundary just because a model did the reading. Left as
    an ordinary function, the interpreter's existing propagation rule
    already keeps the result tagged the same as its input -- untrusted
    text in, untrusted answer out -- with no interpreter change needed
    to make that true.
    """
    prompt = (
        "Answer the question using only the text below, as briefly as "
        "possible. If the answer isn't in the text, say so.\n\n"
        f"Text:\n{text}\n\nQuestion: {question}\nAnswer:"
    )
    response = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()["response"].strip()
