# prompt-lang

A constrained language and interpreter for LLM agents, aimed at open-weight
models. The actual architecture isn't decided yet — this is early. See
the sibling project `../provenance-ac/` for the prior work this grew out
of.

## Where this starts

`interpreter.py` is the seed of the actual interpreter: a safe core that
parses a model's output with `ast.parse` and dispatches only through an
explicit whitelist, never `eval()`. It's ported from `provenance-ac`'s
`agent_demo/agent_loop.py`, which used the identical pattern for a single
tool call. It currently supports a toy grammar of function calls,
assignment, and if/else conditionals (see the grammar in
`interpreter.py`'s module docstring); growing it further means adding
more statement types to `eval_node`/`exec_stmt` (loops, more expression
forms), not relaxing the whitelist check.

## What didn't come over from provenance-ac, on purpose

The object-identity provenance mechanism, `ProvenanceStr` and the side
table, is specific to retrofitting tracking onto unrestricted Python
execution. A custom interpreter doesn't need it: capability metadata can
be attached directly to this interpreter's own values, since every
operation the language supports is one this interpreter's author defined
in the first place.

## Status

Parsing, dispatch, assignment, and if/else conditionals exist, each with
a passing and a rejected-case test. No capability/taint system and no
shared store yet. The immediate next step is testing whether an
open-weight model can reliably generate valid syntax for this grammar,
before building it out further.

```bash
pip install pytest
pytest tests/ -v
```
