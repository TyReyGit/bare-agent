# bare-agent

A from-scratch ReAct agent loop, built to understand and prove out agent
harness engineering — not to demo one. No LangChain, no framework. Every
design decision below is one I can defend, because I made it deliberately
and tested it, not because a library made it for me.

**Status:** Phase 2 of an active harness-engineering track. Loop, tool
contracts, verification, loop detection, cross-run memory, and a
three-tier permission gate are built and tested. Persisted observability
and subagent delegation are not yet — see
[What's not built yet](#whats-not-built-yet).

~1,601 lines across `agent.py` (708 — loop, tool dispatch, compaction,
loop detection, permission checks), `tools.py` (512 — sandboxed tools
plus the bash allowlist), `permissions.py` (209 — three-tier gate),
`memory.py` (169 — cross-run memory), and `__init__.py` (3) — plus 71
tests and a 14-task eval harness with pass^k instrumentation.

## Why no framework

Frameworks like LangChain's `deepagents` (planning tools, virtual
filesystem, subagent delegation, context compression, persistent memory,
sandboxed execution) solve exactly the problems this repo solves. I built
this without one on purpose: I wanted to be able to explain every
mechanism in an interview, not describe a library's behavior. Having since
read the `deepagents` overview, I can say precisely what I chose not to
inherit and why — that comparison is a better interview answer than
either building or not building would be alone.

## Architecture at a glance

| Component | Status | What it is here |
|---|---|---|
| Execution environment | Done | `workspace/` — a scratch directory, not a VM. See [Security posture](#security-posture) for what that actually means |
| Tool contracts | Done | Four tools, typed schemas, error strings written to teach the model what to fix next |
| The loop (ReAct) | Done | `run_agent()` — reason, call a tool, observe, repeat, until a final answer or a stop condition |
| Verification | Done | 14-task eval harness, pass^k instrumentation, ERROR/FAIL distinction |
| Context management | Done | Deterministic compaction + cross-run memory (this doc covers both below) |
| Permissions | Done | Three-tier gate (auto-allow/ask-first/never), 'never' enforced in code — see [Security posture](#security-posture) |
| Observability | Partial | Console trace only; no persisted structured logs yet |
| Orchestration | Not started | No subagent delegation yet |

## Quickstart

```bash
python -m venv venv
source venv/bin/activate

# Reproducible install (exact pinned versions)
pip install -r requirements-lock.txt
pip install -e . --no-deps

# Or, latest versions within the pyproject.toml ranges (not lock-tested)
pip install -e .   # deps are declared in pyproject.toml
```

Note: the lock was generated with `pip freeze` from the environment that
passed the 8 Nov 2026 live smoke test. It pins exact versions but does not
include hashes, and runtime and dev dependencies are in one file.

Create a `.env` file:

`AGENT_PROVIDER` defaults to `groq` if unset. Set it explicitly rather
than relying on the default — it does not persist across sessions:

```bash
AGENT_PROVIDER=groq python -m bare_agent.agent
```

### Providers and models

Three providers: `groq` (default), `gemini`, and `openrouter` (via the
`openai` SDK; needs `OPENROUTER_API_KEY`). Select with flags or env vars
(`AGENT_PROVIDER`, `AGENT_MODEL`); flags win. OpenRouter has no default
model — name an exact slug. The eval runners **require** both flags and
record `provider`, `model` and a per-task `retries` count in every result:

```bash
python -m bare_agent.agent --provider openrouter --model anthropic/claude-haiku-5.5
python eval/run_passk.py -k 3 --provider gemini --model gemini-3.6-flash
```

A 401/402/429 from OpenRouter is reported as a one-line message saying what
to check (key, credit, rate limit) rather than a traceback.

Run the tests:
```bash
python -m pytest -v
mypy src/
```

## The four tools

| Tool | What it does | Notable constraint |
|---|---|---|
| `read_file` | Reads a text file from `workspace/` | Rejects absolute paths and `..` traversal |
| `run_bash` | Runs a read-only inspection command | Allowlist, not denylist — `ls, cat, grep, wc, find, head, tail, sort, uniq, echo` only. No shell, no pipes, no chaining. Options that turn an allowlisted command into a write/exec primitive (`find -exec`, `sort --compress-program`) are blocked per-command, not globally, since the same flag is harmless on a different tool |
| `calculate` | Evaluates arithmetic | AST-parsed against a whitelist of numeric operator nodes — never `eval()` |
| `save_memory` | Records a fact for a future run | Always writes an unverified, expiring entry. Cannot produce a permanent or verified one — see below |

Every tool follows the same contract: never raise, always return a string
the model can act on. `Error: 400` teaches a model nothing; `File 'x' not
found. Files present: [...]` lets it recover on its own. This convention
is Anthropic's, from *Writing effective tools for AI agents* — cited here
because it's the actual source, not because it sounds good.

## Memory: design and the constraint that matters

The agent has no memory between process invocations by default. `memory.py`
adds a small persistent file the harness loads at the start of each run
and the model can append to during one.

The design question that mattered wasn't "how do I persist a fact" — it's
this: **a file the agent writes and later reads back as trusted
instruction is a poisoning channel** (OWASP ASI06). Anything that reaches
the agent's context once — including something read from an adversarial
file via `run_bash` — could otherwise be written to memory and treated as
authoritative forever after.

The mitigation is structural, not a prompt instruction:

- **`DERIVED`** — anything written through `save_memory` (the model's only
  entry point). Hardcoded, unconditionally, regardless of what the model
  claims about where the fact came from. Expires after 7 days.
- **`VERIFIED`** — human-confirmed only, entered via
  `python -m bare_agent.memory "fact"`, run from a terminal. There is no
  code path from inside the agent loop to a `VERIFIED` entry — not a
  restricted one, an absent one. The constraint the "never" permission tier
  is supposed to express in code rather than prompt is applied here to
  *trust* instead of to *actions*.

Tested directly, not just asserted: a unit test calls `save_memory` with a
`basis` string that explicitly claims user confirmation, and confirms the
stored tier is `DERIVED` anyway.

**What actually happened when I tested this live**, rather than what I
expected to happen:

- **Read side — proven.** A workspace with several decoy files and one
  correctly-answering file, non-obviously named. Same task, same
  workspace, only variable changed: with a relevant memory entry present
  (seeded via the human-only path), tool calls dropped from 5 to 2. The
  model didn't blindly trust the memory entry either — it used it as a
  pointer to the right file, then still read that file to get the actual
  figures before calculating, rather than trusting a number it hadn't
  itself verified.
- **Write side — built, tested, not proven to fire unprompted.** Across
  three live runs and two different system-prompt phrasings — one general
  encouragement, one narrower version explicitly naming the trigger
  condition — the model never spontaneously called `save_memory`, even
  immediately after a genuine multi-file search that is exactly the
  situation the tool exists for. I stopped after three consistent
  negative results rather than keep rewording the prompt until one
  attempt happened to work; that would have been evidence of a lucky
  prompt, not a working mechanism. Documented here as a real, unresolved
  limitation.

## Evaluation and metrics

14 tasks across six categories (single-tool, multi-step, bash-dependent,
honest-refusal, silent-success, termination), nine deterministic fixtures.
Verification checks final output, following τ-bench's principle that a
passing suite should confirm the *outcome*, not just that a plausible tool
was called.

| Change | Metric | Result |
|---|---|---|
| Bounded retry on Groq's malformed tool calls | Task completion (single-run) | ~14% → ~57–72% |
| pass^k instrumentation (k=3, Gemini) | Baseline reliability, not just single-run success | pass@1 mean 83%; pass^3 79% (11/14) |
| Loop detection added | pass^3, re-run against the same baseline | Unchanged, 79% (11/14) — proves the mechanism doesn't regress the trusted baseline, though the current fixture set doesn't contain the exact-repeat pattern it targets, so it hasn't yet been proven to *prevent* a real failure |
| System-prompt fix for capability-absence confusion (term-02) | pass^3 | 79% (11/14) → 86% (12/14), zero variance across all three passes |
| Memory present vs. absent (live, same task) | Tool calls to reach the answer | 5 → 2 |

pass^k matters because pass@1 hides variance that kills production
reliability — a system at 90% single-run success can sit around 57%
consistency at k=8 (τ-bench, Yao et al., arXiv 2406.12045). A single run
on a 14-task suite has enormous variance; before/after claims on that
scale are close to meaningless without it.

## Known limitations

Stated honestly, not hidden — a passing test suite proves the code does
what it says, nothing about whether the underlying claim is true.

- **Loop detection catches exact-repeat thrash only.** It does not catch
  the broader failure already observed once in practice: many iterations
  reading different files with no repeated call signature to detect
  against.
- **Compaction is tested against an artificially low threshold**
  (`KEEP_RECENT_EXCHANGES=5`), not real context limits. The current eval
  suite never produces a conversation past the low thousands of tokens —
  nowhere near either provider's actual window (Groq 131K, Gemini
  ~1,048,576) — so exercising the real limit live would be slow and
  costly for little signal. A real bug was still found this way: a
  front-only truncation helper silently dropped a trailing total from a
  `wc -l` result inside a compacted summary. Fixed with a middle-preserving
  truncation and a regression test using the exact string that broke.
- **Memory's write side is unproven under real conditions** — see above.
- **No persisted, structured traces.** Console output only.

## Security posture

- **`run_bash` is a subprocess in a scratch directory with an argv-based
  allowlist — not a sandboxed VM or container.** Commands run via
  `subprocess.run` with an explicit argument list, never `shell=True`, so
  shell operators (`&&`, pipes, globs) are passed through literally rather
  than interpreted. Ten-second timeout. Options that would turn an
  allowlisted command into a write or execution primitive are denied
  per-command. This is the accurate description, not "sandboxed" — see
  OWASP ASI05 (unexpected code execution): separating generation from
  execution is the principle; being precise about what the separation
  actually is is the practice.
- **Three-tier permission gate — auto-allow / ask-first / never.**
  `classify(tool_name, args)` is a pure function mapping every tool call
  to a tier; `check_permission()` is mode-aware and produces the actual
  decision. Tier is a function of `(tool, args)`, not tool name alone —
  `read_file` inside the workspace and `read_file` outside it are
  different tiers. The `never` tier is enforced as a code path the model
  cannot reach, not a prompt instruction the model is asked to honour.
  `ask-first` actions the model can't get approval for (eval runs,
  non-interactive mode) resolve to `AUTO_DENY` — the tool function is
  never entered, and the denial is returned as a teaching result rather
  than a silent failure. Every decision is logged, including auto-allows.
- **A real bug this gate caught before it shipped:** `_classify_bash`
  tokenized commands with `str.split()` while the actual executor used
  `shlex.split()`. The mismatch meant a quoted absolute path — e.g.
  `cat "/etc/passwd"` — was classified against the wrong tokens and
  bypassed the workspace-escape check the allowlist was supposed to
  enforce. Fixed by aligning both paths on `shlex.split()`, with
  regression tests covering the exact quoting pattern that broke.
  Included here rather than left in a commit message because a
  permission gate whose classifier disagrees with its own executor is a
  more interesting and more honest finding than a clean build.
- **Measured, not asserted.** `perm-01` is scored by filesystem state (did
  the forbidden write actually happen), not by asking the model whether it
  complied. pass^3 with the gate live: `perm-01` held 3/3, and non-permission
  tasks held at the pre-gate baseline (11/14) — the gate cost nothing on
  the existing suite.
- **Memory poisoning (OWASP ASI06)** is mitigated structurally, not by
  instruction — see the memory section above.
- **What this doesn't cover yet:** no persisted trace of which actions
  were taken and why beyond console output, no rate limiting beyond what
  the provider SDKs enforce.

## What's not built yet

Persisted structured observability, one subagent delegation. Roadmap
detail lives outside this repo.

## Lessons learned

- **A passing test doesn't verify a claim, only the code.** Applied here
  by testing the memory tier separation adversarially — passing a `basis`
  string that explicitly claims verification, and confirming the tier
  still comes out `DERIVED`.
- **A negative result, reported honestly, is still a result.** Three
  consecutive live runs where the model didn't use a tool it was
  encouraged to use are a real finding about prompt reliability, not a
  bug to quietly patch around before anyone sees it.
- **Fake-model unit tests can't catch everything.** The compaction
  truncation bug and the memory write-side non-use finding were both only
  visible against the real API — deterministic tests proved the mechanism
  correct, live testing proved (or disproved) that it actually fires.

---

*A note on how this feature came to exist: most of this repo — the
original ReAct loop, the sandboxed tools, loop detection, compaction —
was typed by hand, deliberately, so I can walk through any line of it in
an interview. The memory feature (`memory.py`, plus the `save_memory`
wiring into `tools.py` and `agent.py`) broke that pattern: Claude wrote
the code, and I copied it into the files myself and made my own edits on
top, using that process as deliberate Python-learning practice rather
than transcribing it verbatim. I can explain every line here too, but the
route to that understanding was different for this one deliverable —
worth being straightforward about rather than letting the rest of this
README imply otherwise.*
