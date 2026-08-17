# prompt-lang

A constrained language and interpreter for LLM agents, aimed at open-weight
models. The actual architecture isn't decided yet — this is early. See
the sibling project `../provenance-ac/` for the prior work this grew out
of.

## Where this starts

`prompt_lang/interpreter.py` is the seed of the actual interpreter: a safe core that
parses a model's output with `ast.parse` and dispatches only through an
explicit whitelist, never `eval()`. It's ported from `provenance-ac`'s
`agent_demo/agent_loop.py`, which used the identical pattern for a single
tool call. It currently supports a toy grammar of function calls,
assignment, if/else conditionals, while and for loops, and lists (see the
grammar in `prompt_lang/interpreter.py`'s module docstring); growing it further means
adding more statement types to `eval_node`/`exec_stmt`, not relaxing the
whitelist check.

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
`prompt_lang/interpreter.py`); a loop whose condition never goes false raises
`InterpreterError` instead of running forever, since a language a model
writes programs in shouldn't be able to hang the process running it.
That cap bounds a single loop, though, not total program execution —
nested loops each reset their own counter independently, so two loops
nested inside each other can each individually stay under 1000 while the
program as a whole does far more work (found by deliberately trying it:
300 outer times 300 inner ran 90,000 operations with neither loop ever
exceeding the cap). Fixed with a shared `_IterationBudget`, created once
per `run()` call and consumed by every loop pass across the whole
program, so nesting can't multiply past one fixed total.

Lists now exist too, built with a list literal, plus indexing and for
loops. Trust is tracked per element, not per collection — a list literal
evaluates each element normally and keeps every element's own
(value, Trust) pair, so a list mixing trusted and untrusted items doesn't
collapse to one tag for the whole thing; indexing and iterating both
read that per-element structure directly. A plain Python list handed
back by a function in `allowed` has no per-element tags of its own, since
the interpreter never watched it get built, but instead of rejecting it,
`eval_node` auto-wraps it: every element gets the same trust the call as
a whole already earned. That's uniform, not precise — a function that
needs real per-element distinctions can opt out by returning already
`(value, Trust)`-tagged pairs itself, which auto-wrap detects and leaves
untouched.

Dicts now exist too, following the exact same rules as lists: a dict
literal keeps each key's and value's own trust, its outer trust is the
join of all of them, subscripting reads a value's own tag directly, and
a plain dict from an outside function gets auto-wrapped the same way a
plain list does. The sanitizer-laundering finding described further down
(a sanitizer clearing a container's outer tag while its contents stay
untrusted) was checked against dicts before shipping them, not after —
`_has_untrusted` covers both container shapes from the start, so the
same class of gap didn't need to be found a second time. Dicts don't
support iteration yet (`for k in some_dict`), only literal construction
and lookup.

The feasibility question — can an open-weight model reliably generate
valid syntax for this grammar — has a real answer across three models,
5 tasks x 5 reps each (`experiments/feasibility_test.py`): `llama3.2:3b` 80% correct,
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
Confidentiality now exists too, as a fully independent second tag —
`Secrecy` (`PUBLIC`/`SECRET`) — tracked alongside `Trust` on every value,
not folded into it, since a value can be trustworthy but secret (a real
API key) or untrustworthy but not remotely sensitive (an attacker's
email body) at the same time. `confidential` names whitelisted functions
whose return value is always secret; `sinks` names whitelisted functions
that refuse to run at all if any argument is secret, raising
`ConfidentialityError`; `declassifiers` names whitelisted functions
allowed to deliberately clear the secret tag, the confidentiality
counterpart to `sanitizers`. Every rule already established for the
integrity side — propagation through chains of calls, per-element
tracking in lists and dicts, auto-wrap, `sources` beating `sanitizers` on
a contradictory name — has a mirrored version here, and the
sanitizer-laundering gap already found and fixed for integrity was
checked against confidentiality before shipping, not after.

Neither side tracks more than explicit data flow by default — does a
tagged value reach a call as an argument. Adversarial testing found that
this misses implicit flow entirely on the integrity side first:
untrusted data can decide which branch of an if/while runs without ever
appearing as an argument, so a privileged call that takes no untrusted
argument (or no argument at all) had nothing for the check to see.
`send_to_attacker()`, called from inside
`if read_email() == "...": send_to_attacker()`, ran with no error at
all, purely because the email's content picked that branch. The same
gap existed on the confidentiality side, confirmed directly:
`if read_api_key() == guess: reveal_match() else: reveal_no_match()`
also ran with no error, leaking which branch fired even though neither
call took a secret argument.

Both are now closed by a scoped, partial mitigation: every call threads
a `pc_trust` value and a `pc_secrecy` value — the trust and secrecy of
the control flow that led there — raised on entering any branch or loop
body whose condition was tagged. A privileged call made under an
untrusted `pc_trust` is refused regardless of its own arguments; a sink
call made under a secret `pc_secrecy` is refused the same way. This is
deliberately narrow on both sides — it gates privileged/sink calls only,
not a complete implicit-flow system (an assignment made under a raised
pc isn't itself retroactively tainted, for instance). Real implicit-flow
tracking done properly tends to make a system very restrictive, since
almost anything computed inside a branch on tagged data ends up needing
the same treatment; most practical taint trackers, Perl's taint mode
included, deliberately don't attempt it for that reason. This is a
bounded compromise scoped to the concrete cases adversarial testing
actually found on each side, not a claim of completeness.

A shared-store test confirms the mechanism against a stateful store, not
just a stateless stub: a planted value blocks a downstream privileged
call, an unrelated non-privileged call using shared-store data is
unaffected, and — the case worth calling out — a program is still blocked
from a privileged call using data it wrote to the store itself earlier in
the same run, since every read from a shared store is treated as
untrusted regardless of who wrote it.

That test used one `run()` call reading its own write. A further pass
tested genuinely separate agent turns — one `run()` call per agent,
sharing only the store object — and confirmed the same guarantee holds
across the boundary: agent A writes, a completely separate agent B call
reads it, reprocesses it (through an ordinary function, and separately
through a declared sanitizer), writes the result back under a new key,
and a third agent's independent read is still correctly untrusted. This
isn't incidental — the store only ever holds a bare value, since
`write_shared` receives it with its trust tag already stripped, so trust
is never carried through the data itself. Every agent's tag comes
entirely from *that agent's own* `sources` declaration at the moment it
reads, never from what an earlier agent did. One consequence, also
confirmed: an agent that forgets to declare the shared read as untrusted
only weakens its own safety — it doesn't compromise a separately,
correctly configured agent reading the same data afterward. Concurrency
itself — two `run()` calls actually racing on the same store — is
deliberately outside the interpreter's scope; it holds no state across
calls (confirmed directly), but true concurrent access is the store
implementation's own responsibility.

A first adversarial pass found a real gap, not a hypothetical one: a
list's own outer trust tag can be cleared by any sanitizer, however
trivial, independent of what's nested inside it. `identity_sanitizer(x):
return x` applied to `[read_secret()]` produced a list whose outer tag
read trusted while the element inside was still tagged untrusted, and
the privileged check only ever looked at the outer tag — passing the
whole list to a privileged function slipped through untouched. Indexing
into the list still saw the correct nested tag; only passing the
container as a whole was affected. Fixed by making the privileged check
and trust propagation both look recursively inside a tagged list's
elements (`_has_untrusted` in `prompt_lang/interpreter.py`), not just at a value's
own outer tag, so a sanitizer can't launder a container's contents by
clearing only its own top-level label.

Everything above tested one direction — does the system correctly block
bad cases. A separate pass tested the opposite direction: does it
over-restrict, blocking things that should be allowed just because
something untrusted or secret exists elsewhere in the program (the
failure mode `provenance-ac` checked itself against, named after
FIDES's session-level tainting). Nine deliberately constructed cases —
an unused untrusted/secret variable sitting near an unrelated clean
call, `pc_trust`/`pc_secrecy` not leaking between sibling `if` blocks or
out of a `while` loop after it ends, nested trusted branches, a loop
over a trusted list, program order, an unrelated shared-store write —
all passed on the first attempt. No over-restriction bug found. That's
a real result, not proof there isn't one anywhere: these nine specific
cases are confirmed correct, nothing more.

```bash
pip install pytest
pytest tests/ -v
```
