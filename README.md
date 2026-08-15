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

Parsing, dispatch, assignment, if/else conditionals, and bounded while
loops exist, each with a passing and a rejected-case test. Loops are
capped at a fixed number of iterations (`MAX_WHILE_ITERATIONS` in
`interpreter.py`); a loop whose condition never goes false raises
`InterpreterError` instead of running forever, since a language a model
writes programs in shouldn't be able to hang the process running it.

The feasibility question — can an open-weight model reliably generate
valid syntax for this grammar — has a real answer across three models,
5 tasks x 5 reps each (`feasibility_test.py`): `llama3.2:3b` 80% correct,
`qwen2.5:3b` 56%, and `qwen2.5:32b` 100%, run on a rented A40 GPU since
the 32B model doesn't fit on this machine's 8GB of RAM. Failures split
cleanly between invalid syntax, valid Python outside this grammar (the
model reaching for constructs like the walrus operator or ternary
expressions), and legal-but-wrong-behavior programs. The gap between 3B
and 32B is the first real evidence that the earlier 3B failures are a
capability ceiling, not a fixable prompting issue.

A targeted check on the loop task specifically (`loop_count`, 5 reps,
`llama3.2:3b` and `qwen2.5:3b`, both run locally) showed a sharper split
than the five tasks above: `qwen2.5:3b` got it right on every attempt,
`llama3.2:3b` failed on all but one, consistently by trying to reassign
the result of a function call directly
(`get_start() = increment(get_start())`) instead of tracking a local
variable across iterations. Whether this holds past this one task is
untested.

A minimal capability system now exists: every value carries a Trust tag
(`TRUSTED`/`UNTRUSTED`) threaded through `eval_node`/`exec_stmt`, not
attached via object identity. `sources` names whitelisted functions whose
return value is always untrusted; `privileged` names whitelisted functions
that refuse to run at all if any argument is untrusted, raising
`CapabilityError` before the underlying call happens. Trust propagates
through ordinary functions too — an untrusted value passed through any
chain of non-source, non-privileged calls stays untrusted, since each call
combines its own source-tag with the trust of its arguments rather than
resetting to trusted. `sanitizers` names whitelisted functions whose
return value is always trusted, regardless of its arguments — the one
deliberate, explicitly declared way for a program to turn untrusted data
back into trusted, kept intentionally narrow: a name has this effect only
if it's listed, never because of what the function happens to do.
Confidentiality, the reverse direction of stopping sensitive data from
reaching an untrusted sink, still isn't handled.

A shared-store test confirms the mechanism against a stateful store, not
just a stateless stub: a planted value blocks a downstream privileged
call, an unrelated non-privileged call using shared-store data is
unaffected, and — the case worth calling out — a program is still blocked
from a privileged call using data it wrote to the store itself earlier in
the same run, since every read from a shared store is treated as
untrusted regardless of who wrote it.

```bash
pip install pytest
pytest tests/ -v
```
