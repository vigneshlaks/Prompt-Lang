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

`provenance-ac` enforces sinks and sources by wrapping specific dangerous
callables directly (`requests.get`/`post`, `subprocess.run`/`Popen`,
`open`) and flags untrusted values by Python object identity (`ProvenanceStr`, the
side table). Both are enumerative: safety depends on every dangerous
entry point having been identified and wrapped, and tracking survives
only as long as an untrusted value's object identity survives. Neither
holds in general — the project's own AgentDojo integration hit this
directly. Tool output there gets serialized to a string and re-parsed by
`json.loads()` into a new object with no identity link to the original,
so `id()`-based matching can't see it; `benchmarks/agentdojo_adapter.py`
had to fall back to content matching instead.

A closed interpreter doesn't have that class of gap. Nothing can execute
except what `eval_node`/`exec_stmt` explicitly implement, so there's no
"did we remember to wrap this sink" question — an unhandled path to a
privileged operation isn't missed, it doesn't exist, by construction, not
by coverage effort. And since the interpreter owns its own variable
bindings (`env`) rather than relying on Python's object identity,
untrusted status can be tracked through the interpreter's own bookkeeping instead of
`id()`, sidestepping the exact fragility that broke object-identity
tracking against AgentDojo. The tradeoff is real: this only works because
the language is small enough to fully enumerate every operation it can
perform, a far smaller surface than unrestricted Python.

## Status

Parsing, dispatch, assignment, and if/else conditionals exist, each with
a passing and a rejected-case test. The feasibility question — can an
open-weight model reliably generate valid syntax for this grammar — has
a first real answer across two 3B models (`llama3.2:3b` 72% correct,
`qwen2.5:3b` 60%, 5 tasks x 5 reps each, `feasibility_test.py`), with
failures cleanly split between invalid syntax, valid Python outside this
grammar (the model reaching for constructs like the walrus operator or
ternary expressions), and legal-but-wrong-behavior programs.

A minimal capability system now exists: every value carries a Trust tag
(`TRUSTED`/`UNTRUSTED`) threaded through `eval_node`/`exec_stmt`, not
attached via object identity. `sources` names whitelisted functions whose
return value is always untrusted; `privileged` names whitelisted functions
that refuse to run at all if any argument is untrusted, raising
`CapabilityError` before the underlying call happens. This is single-hop
only — an untrusted value passed through an ordinary function comes back
trusted again, since propagation through arbitrary computation isn't tracked
yet.

A single-hop shared-store test confirms the mechanism against a stateful
store, not just a stateless stub: a planted value blocks a downstream
privileged call, an unrelated non-privileged call using shared-store data
is unaffected, and — the case worth calling out — a program is still
blocked from a privileged call using data it wrote to the store itself
earlier in the same run, since every read from a shared store is treated
as untrusted regardless of who wrote it.

```bash
pip install pytest
pytest tests/ -v
```
