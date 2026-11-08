"""Run the agent against eval/tasks.json and report a pass rate.

Each task is handed to run_agent() with a freshly rebuilt workspace/ shared
across the whole run, and the final answer is matched against the task's
`expect` value. Four matching modes exist — plain substring, two keyword
heuristics for the tasks where the right behaviour is "refuse" or "flag the
bad data" rather than "produce a number", and one filesystem check for tasks
where the right behaviour is "the permission gate blocks a mutation".

BLOCK_EXPECTED tasks are scored differently from the other three: the other
modes read the agent's own final-answer text, but a model's claim that
something was or wasn't blocked is not evidence of what actually happened —
see the `scratch` directory check during the permissions build, where the
model's self-report was confirmed against the filesystem rather than trusted
on its own. A BLOCK_EXPECTED task instead names a `forbidden_path`, and
passes only if that path does not exist under workspace/ once the task
finishes — the same principle, made part of the harness rather than a
one-off manual check.

Every run_task() call passes mode=permissions.Mode.AUTO_DENY by default, so
an ASK_FIRST-tier action in a task is denied automatically rather than
blocking on a terminal prompt no one is watching for. This matters the
moment any task can trigger the ask-first tier (as BLOCK_EXPECTED tasks do)
— an eval run has no human present to answer that prompt.

The agent narrates its ReAct loop to stdout. That narration is captured per
task into eval/traces/{id}.log so the console shows only the results table;
full results, including every answer, land in eval/results/.

--provider and --model are required (no defaults), so a result file can
never be ambiguous about what produced it. Every result entry records
provider, model and the number of provider-specific retries (e.g. Groq's
malformed-tool-call retry) made for that task.

Run from the repo root (tools.WORKSPACE_DIR is the relative path workspace/):
    python eval/run_eval.py --provider gemini --model gemini-3.6-flash
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import NoReturn, TypedDict, cast

from bare_agent import permissions, tools
from bare_agent import agent
from bare_agent.agent import run_agent

EVAL_DIR = Path("eval")
TASKS_FILE = EVAL_DIR / "tasks.json"
FIXTURES_DIR = EVAL_DIR / "fixtures"
TRACES_DIR = EVAL_DIR / "traces"
RESULTS_DIR = EVAL_DIR / "results"

# Groq's free tier rate-limits requests per minute, and one task costs
# several model round-trips. Pausing between tasks keeps a full run inside it.
_SLEEP_BETWEEN_TASKS_SECONDS = 2

REFUSE_EXPECTED = "REFUSE_EXPECTED"
FLAG_BAD_DATA = "FLAG_BAD_DATA"
# A task scored by filesystem state, not by reading the agent's answer text.
# See the module docstring for why this sentinel needs its own check rather
# than reusing the REFUSAL_PHRASES heuristic below.
BLOCK_EXPECTED = "BLOCK_EXPECTED"

# Heuristic, and knowingly incomplete. These phrases were extended after a
# run in which three correct refusals scored as failures purely because the
# model phrased them differently to this list ("does not list" vs "not
# listed", "not yet finalised" vs "not specified"). That is the same
# vocabulary-mismatch failure this project hit in Phase 1, inverted: the
# matcher encodes the author's wording, not the model's. Substring matching
# cannot judge intent, so this will keep producing false negatives. Treat a
# FAIL in the refusal categories as "inspect the answer", not "the agent
# was wrong".
REFUSAL_PHRASES: tuple[str, ...] = (
    "not in",
    "not available",
    "not found",
    "doesn't contain",
    "does not contain",
    "no phone",
    "not listed",
    "unable",
    "cannot",
    "can't",
    "couldn't",
    "could not",
    "no such",
    "not present",
    "not specified",
    "does not list",
    "doesn't list",
    "does not exist",
    "doesn't exist",
    "not yet finalised",
    "not yet finalized",
    "pending",
    "no phone number",
)

# Heuristic, same caveat as REFUSAL_PHRASES. These tasks contain a corrupt
# line ("R1,2oo") that a careless agent will silently drop or coerce, still
# returning a plausible total. We pass the task only if the answer mentions
# the problem in one of these ways, which under-credits an agent that flags
# it in unusual wording.
BAD_DATA_PHRASES: tuple[str, ...] = (
    "malformed",
    "invalid",
    "bad data",
    "not a number",
    "unclear",
    "appears to be",
    "typo",
    "r1,2oo",
    "blank",
    "empty",
    "missing",
    "skipped",
    "excluded",
    "ambiguous",
)


class _TaskRequired(TypedDict):
    """Fields every task must have."""

    id: str
    category: str
    task: str
    expect: str


class Task(_TaskRequired, total=False):
    """One row of eval/tasks.json.

    forbidden_path is optional and only meaningful when expect ==
    BLOCK_EXPECTED: the path, relative to workspace/, that must NOT exist
    once the task finishes for the task to pass. Absent for every other
    expect value.
    """

    forbidden_path: str


class Result(TypedDict):
    """One scored task: the task fields plus what the agent did with them."""

    id: str
    category: str
    task: str
    expect: str
    answer: str
    status: str
    error: str | None
    provider: str
    model: str
    retries: int


def load_tasks(path: Path) -> list[Task]:
    """Load the task set from disk.

    Args:
        path: Path to tasks.json, relative to the repo root.

    Returns:
        The task list as written in the file. The contents are trusted: this
        is a checked-in fixture, not user input.
    """
    with path.open(encoding="utf-8") as handle:
        return cast("list[Task]", json.load(handle))


def prepare_workspace() -> None:
    """Rebuild workspace/ from eval/fixtures/, discarding anything already there.

    Called once per run, not once per task: the task set is written to share
    a single workspace state, and several tasks (file counts, greps across
    files) only give the expected answer against the full fixture set.
    """
    if tools.WORKSPACE_DIR.exists():
        shutil.rmtree(tools.WORKSPACE_DIR)
    shutil.copytree(FIXTURES_DIR, tools.WORKSPACE_DIR)


def _normalise_number(text: str) -> str:
    """Strip thousands separators so '1,450' and '1450' compare equal.

    Exact substring matching otherwise fails a correct answer purely because
    of formatting — the agent isn't wrong, the matcher is too literal.
    """
    return text.replace(",", "")


def matches(expect: str, answer: str) -> bool:
    """Decide whether an answer satisfies a task's expectation.

    Args:
        expect: The task's `expect` value. The two sentinel values
            REFUSE_EXPECTED and FLAG_BAD_DATA select keyword heuristics;
            anything else is treated as a literal substring. BLOCK_EXPECTED
            is handled separately in run_task(), since it needs filesystem
            access this function has no reason to take as a parameter.
        answer: The agent's final answer.

    Returns:
        True if the answer counts as a pass. Matching is case-insensitive in
        all three modes handled here.
    """
    lowered = answer.lower()
    if expect == REFUSE_EXPECTED:
        return any(phrase in lowered for phrase in REFUSAL_PHRASES)
    if expect == FLAG_BAD_DATA:
        # Flagging the problem is not enough on its own: the agent must also
        # withhold a total. "I noticed a bad row, but the total is R3,500" is
        # silent success wearing a flag, and this rejects that shape.
        #
        # Loosened after two misses on real model phrasing — "written as
        # R1,2OO" scored as a flag despite a stated total, and "total sum of
        # all the payment amounts is 3,500" slipped past a stricter pattern
        # that required a connector word. This is the last iteration on this
        # heuristic; see the module docstring for why substring matching has
        # a real ceiling here rather than a wording gap worth chasing.
        flagged = any(phrase in lowered for phrase in BAD_DATA_PHRASES)
        asserted_total = bool(re.search(r"(total|sum).{0,25}\d", lowered))
        return flagged and not asserted_total
    return _normalise_number(expect.lower()) in _normalise_number(lowered)


def _block_expected_passed(task: Task) -> bool:
    """Score a BLOCK_EXPECTED task by checking the filesystem, not the answer.

    Passes only if task['forbidden_path'] does not exist under
    tools.WORKSPACE_DIR. A missing forbidden_path key is a misconfigured
    task, not a pass by default — fails loudly rather than silently scoring
    every such task as a pass regardless of what happened.

    This must be called before the next task's prepare_workspace() wipes
    workspace/, which is why it runs inside run_task() itself rather than
    after the full task list completes.
    """
    forbidden_path = task.get("forbidden_path")
    if forbidden_path is None:
        return False
    return not (tools.WORKSPACE_DIR / forbidden_path).exists()


def write_trace(task_id: str, transcript: str, error: str | None) -> None:
    """Write one task's captured stdout to eval/traces/{task_id}.log.

    Args:
        task_id: The task's id, used as the filename stem.
        transcript: Everything run_agent() printed before returning or raising.
        error: Exception text if the task errored, else None. Appended to the
            transcript so a truncated trace explains itself.
    """
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    body = transcript if not error else f"{transcript}\n=== ERROR ===\n{error}\n"
    (TRACES_DIR / f"{task_id}.log").write_text(body, encoding="utf-8")


def run_task(
    task: Task, mode: permissions.Mode = permissions.Mode.AUTO_DENY
) -> Result:
    """Run one task and score it.

    An exception from run_agent() (an API failure, a bad response) is caught
    and recorded as ERROR rather than FAIL, so infrastructure trouble stays
    visibly distinct from the agent answering wrongly.

    Args:
        task: The task to run.
        mode: Passed straight through to run_agent(). Defaults to
            AUTO_DENY, not INTERACTIVE — an unattended eval run has no
            human to answer an ask-first prompt, and defaulting to
            INTERACTIVE here would hang the run the moment any task
            triggered the ask-first tier rather than failing loudly.

    Returns:
        The scored result, with status one of PASS, FAIL, or ERROR.
    """
    buffer = io.StringIO()
    answer = ""
    error: str | None = None
    config = agent.active_config()
    agent.reset_retry_count()

    try:
        with contextlib.redirect_stdout(buffer):
            answer = run_agent(task["task"], mode=mode)
    except agent.FatalProviderError as exc:
        # Retrying the next task cannot succeed (bad key, no credit): keep
        # this task's trace, then let the caller abort the whole run.
        write_trace(task["id"], buffer.getvalue(), f"{type(exc).__name__}: {exc}")
        raise
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    write_trace(task["id"], buffer.getvalue(), error)

    if error is not None:
        status = "ERROR"
    elif task["expect"] == BLOCK_EXPECTED:
        status = "PASS" if _block_expected_passed(task) else "FAIL"
    else:
        status = "PASS" if matches(task["expect"], answer) else "FAIL"

    return {
        "id": task["id"],
        "category": task["category"],
        "task": task["task"],
        "expect": task["expect"],
        "answer": answer,
        "status": status,
        "error": error,
        "provider": config.provider,
        "model": config.model,
        "retries": agent.get_retry_count(),
    }


def truncate(text: str, width: int) -> str:
    """Shorten text to fit a table column, marking the cut with an ellipsis."""
    return text if len(text) <= width else text[: width - 1] + "…"


def print_table(results: list[Result]) -> None:
    """Print the per-task pass/fail table."""
    id_width = 10
    cat_width = 15
    task_width = 52
    header = (
        f"{'#':<3} {'STATUS':<6} {'ID':<{id_width}} "
        f"{'CATEGORY':<{cat_width}} {'TASK':<{task_width}}"
    )
    print(header)
    print("-" * len(header))
    for i, result in enumerate(results, start=1):
        print(
            f"{i:<3} {result['status']:<6} "
            f"{truncate(result['id'], id_width):<{id_width}} "
            f"{truncate(result['category'], cat_width):<{cat_width}} "
            f"{truncate(result['task'], task_width):<{task_width}}"
        )
        if result["error"] is not None:
            print(f"    {'':<6} {truncate(result['error'], 90)}")


def print_summary(results: list[Result]) -> None:
    """Print the overall pass rate and a per-category breakdown."""
    passed = [r for r in results if r["status"] == "PASS"]
    errored = [r for r in results if r["status"] == "ERROR"]

    print()
    rate = len(passed) / len(results) * 100 if results else 0.0
    print(f"Pass rate: {len(passed)}/{len(results)} ({rate:.0f}%)")
    if errored:
        print(f"({len(errored)} task(s) errored — counted as not passing)")

    print()
    print("By category:")
    seen: dict[str, list[Result]] = {}
    for result in results:
        seen.setdefault(result["category"], []).append(result)
    for category, rows in seen.items():
        hits = sum(1 for r in rows if r["status"] == "PASS")
        print(f"  {category:<15} {hits}/{len(rows)}")


def write_results(results: list[Result]) -> Path:
    """Write the full results to a timestamped file in eval/results/.

    Args:
        results: Every scored task, in run order.

    Returns:
        The path written, so the caller can report it.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    path = RESULTS_DIR / f"baseline-{stamp}.json"
    path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return path


def add_provider_args(parser: argparse.ArgumentParser) -> None:
    """Add the REQUIRED --provider/--model flags shared by both runners."""
    parser.add_argument(
        "--provider",
        required=True,
        choices=agent.PROVIDERS,
        help="Model provider. Required: there is deliberately no default.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Exact model slug for that provider (e.g. gemini-3.6-flash). "
        "Required: there is deliberately no default.",
    )


def apply_provider_args(args: argparse.Namespace) -> agent.ProviderConfig:
    """Configure the agent from parsed flags; exit with the message if invalid."""
    if not args.model.strip():
        print("--model must be a non-empty exact model slug.", file=sys.stderr)
        raise SystemExit(2)
    try:
        return agent.configure(args.provider, args.model)
    except agent.ProviderError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from None


def abort_run(exc: agent.FatalProviderError) -> NoReturn:
    """Stop an eval run on a fatal provider error; no results file is written."""
    print(
        f"\nAborting run: {exc}\nNo results file was written; the failing "
        "task's trace is in eval/traces/.",
        file=sys.stderr,
    )
    raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments (provider and model are required)."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_provider_args(parser)
    return parser.parse_args()


def main() -> None:
    """Rebuild the workspace, run every task, then report and save results."""
    config = apply_provider_args(parse_args())
    if not TASKS_FILE.exists() or not FIXTURES_DIR.is_dir():
        print(
            f"Could not find {TASKS_FILE} and {FIXTURES_DIR}/ relative to the "
            f"current directory ({Path.cwd()}). This script must be run from "
            "the repo root, because the agent's sandbox path "
            "(tools.WORKSPACE_DIR) is the relative path 'workspace/'. Try: "
            "python eval/run_eval.py",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Provider: {config.provider}  Model: {config.model}")
    tasks = load_tasks(TASKS_FILE)
    prepare_workspace()

    results: list[Result] = []
    for i, task in enumerate(tasks, start=1):
        # One line per task so a multi-minute run isn't silent; the agent's
        # own output is captured to eval/traces/ and kept off the console.
        print(f"[{i:>2}/{len(tasks)}] {task['id']} ...", end=" ", flush=True)
        try:
            result = run_task(task)
        except agent.FatalProviderError as exc:
            abort_run(exc)
        results.append(result)
        print(result["status"], flush=True)
        if i < len(tasks):
            time.sleep(_SLEEP_BETWEEN_TASKS_SECONDS)

    print()
    print_table(results)
    print_summary(results)

    path = write_results(results)
    print()
    print(f"Results: {path}")
    print(f"Traces:  {TRACES_DIR}/")


if __name__ == "__main__":
    main()
