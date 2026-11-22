"""Run an eval task set k times and report pass@1 and pass^k.

--tasks v1 (the default) is eval/tasks.json and is what the rest of this
docstring describes. --tasks v2 is handed to eval/passk_v2.py: a different
task set and scorer, so a new baseline that no v1 number compares to.

Wraps run_eval.py rather than duplicating it: task loading, workspace setup,
and per-task scoring (including the substring/keyword matching and its
documented ceiling) all come from there unchanged. This script's only job is
running that set k times and folding the results together, which keeps the
already-verified single-run harness untouched — nothing here can regress the
~72% pass@1 baseline it already produced.

pass@1 answers "did this pass this time". pass^k answers "did this pass
EVERY time across k attempts" — and because pass^k = p^k, it decays
exponentially even for a task with a solid single-run rate (Yao et al.,
tau-bench, arXiv 2406.12045). The judgment categories (silent-success,
honest-refusal) are the ones expected to show this: they depend on the
model noticing something on every attempt, not on deterministic retrieval.

Every pass rebuilds the workspace from fixtures, same as a single
run_eval.py run. Each task's trace is additionally copied to
eval/traces/run-{n}/{id}.log before the next pass overwrites
eval/traces/{id}.log, so a task that fails on run 3 but not run 1 can
actually be inspected.

Provider notes, updated 12 August 2026 after the first live pass^k run:
Groq's free tier hits a hard tokens-per-day ceiling well before a k=3 pass
finishes across 14 tasks, and its documented ~65% malformed-tool-call rate
makes a clean pass^k run on Groq's free tier impractical as things stand.
Gemini on the paid tier ("Tier 1", unlocked with prepaid credit) cleared a
full k=3 run with zero malformed tool calls and one unrelated 429 on a
single task. Until a properly-paced Groq k=3 run exists for comparison,
treat Gemini-on-paid-tier as the practical choice for pass^k work — not as
a proven "Gemini is more reliable" claim. Only one provider has a valid
pass^k number so far.

Scoring notes, added 14 August 2026: a task's `all_passed` (the strict
pass^k signal) still requires PASS on every run — an ERROR is never
counted as a pass, because a failed API call tells us nothing about
whether the agent would have gotten it right. But ERROR and FAIL are now
tracked separately per task, and a task that missed pass^k with only
ERRORs among its non-passing runs (no FAIL) is flagged `error_limited`:
its misses are known to be infrastructure noise, not demonstrated
reasoning failures, and reading it as reasoning instability would be a
mistake. This distinction exists because the first live pass^k run
(12 Aug, Gemini) had exactly this shape on term-02: 2 PASS, 1 ERROR
(429), 0 FAIL — scored as "not all passed" with no way to tell that
apart from a task with a genuine wrong answer, until now.

Run from the repo root:
    python eval/run_passk.py -k 3 --provider gemini --model gemini-3.6-flash
    python eval/run_passk.py -k 3 --provider openrouter --model <exact-slug>
    python eval/run_passk.py --tasks v2 -k 3 --provider gemini --model gemini-3.6-flash

--provider and --model are required (no defaults); both are recorded in the
results file, and each run's per-task entries carry them plus a retry count.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import TypedDict

from bare_agent import agent
from run_eval import (
    FIXTURES_DIR,
    RESULTS_DIR,
    TASKS_FILE,
    TRACES_DIR,
    Result,
    Task,
    abort_run,
    add_provider_args,
    apply_provider_args,
    load_tasks,
    prepare_workspace,
    print_summary,
    print_table,
    run_task,
    truncate,
    v2_config,
)

# Matches run_eval.py's per-task pacing constant; see its rationale in the
# module docstring there. Repeated here rather than importing a private
# module constant.
SLEEP_BETWEEN_TASKS_SECONDS = 2

# Extra pause between full k-passes, on top of the per-task sleep above. A
# k-run multiplies live API calls against a rate limit that resets per
# minute, not per run — this buffer is what keeps a k=5 run from tripping it
# partway through pass 4.
#
# This value was sized for Groq's per-minute quota. Against Gemini's paid
# tier it is likely more conservative than necessary — the first live k=3
# Gemini run completed cleanly with this pause, but no attempt has been made
# to shorten it. Worth tuning later; not touched now so this fix stays
# scoped to correcting the inaccurate pause message, not changing behaviour.
SLEEP_BETWEEN_RUNS_SECONDS = 15


class TaskPassK(TypedDict):
    """One task's record across all k runs.

    passes + fails + errors always sums to k. all_passed is the strict
    pass^k signal (PASS on every run). error_limited applies only when
    all_passed is False: it's True when every non-passing run was an
    ERROR and none was a FAIL — meaning nothing observed demonstrates the
    agent reasoning incorrectly, only that the infrastructure didn't hold.
    """

    id: str
    category: str
    task: str
    k: int
    passes: int
    fails: int
    errors: int
    all_passed: bool
    error_limited: bool
    statuses: list[str]


def run_one_pass(tasks: list[Task], run_index: int, total_runs: int) -> list[Result]:
    """Run every task once against a freshly rebuilt workspace.

    Identical in effect to a single run_eval.py invocation, except each
    task's trace is additionally preserved under eval/traces/run-{run_index}/
    before the next pass's run_task() call overwrites the flat trace file.

    Args:
        tasks: The task set, in the order to run them.
        run_index: This pass's 1-based position, for progress printing and
            the trace subdirectory name.
        total_runs: Total number of passes, for progress printing only.

    Returns:
        Every task's scored result, in task order.
    """
    prepare_workspace()
    results: list[Result] = []
    dest_dir = TRACES_DIR / f"run-{run_index}"
    for i, task in enumerate(tasks, start=1):
        print(
            f"[run {run_index}/{total_runs}] [{i:>2}/{len(tasks)}] "
            f"{task['id']} ...",
            end=" ",
            flush=True,
        )
        try:
            result = run_task(task)
        except agent.FatalProviderError as exc:
            abort_run(exc)
        results.append(result)
        print(result["status"], flush=True)

        src = TRACES_DIR / f"{task['id']}.log"
        if src.exists():
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest_dir / f"{task['id']}.log")

        if i < len(tasks):
            time.sleep(SLEEP_BETWEEN_TASKS_SECONDS)
    return results


def aggregate(tasks: list[Task], runs: list[list[Result]]) -> list[TaskPassK]:
    """Fold k runs of the same task set into one pass^k record per task.

    Args:
        tasks: The task set (for id/category/task text, in task order).
        runs: Each pass's Result list, in the same task order every time.

    Returns:
        One TaskPassK per task, in task order.
    """
    records: list[TaskPassK] = []
    for i, task in enumerate(tasks):
        statuses = [run[i]["status"] for run in runs]
        passes = sum(1 for s in statuses if s == "PASS")
        fails = sum(1 for s in statuses if s == "FAIL")
        errors = sum(1 for s in statuses if s == "ERROR")
        all_passed = passes == len(runs)
        error_limited = (not all_passed) and fails == 0 and errors > 0
        records.append(
            {
                "id": task["id"],
                "category": task["category"],
                "task": task["task"],
                "k": len(runs),
                "passes": passes,
                "fails": fails,
                "errors": errors,
                "all_passed": all_passed,
                "error_limited": error_limited,
                "statuses": statuses,
            }
        )
    return records


def print_passk_summary(records: list[TaskPassK], run_rates: list[float]) -> None:
    """Print the per-task pass^k table, then the overall and by-category rollups."""
    if not records:
        return
    k = records[0]["k"]

    print()
    print(f"pass^{k} — a task counts only if it passed on ALL {k} runs")
    id_width, cat_width = 10, 15
    header = (
        f"{'ID':<{id_width}} {'CATEGORY':<{cat_width}} {'PASSES':<8} "
        f"{'ALL PASSED':<12} NOTE"
    )
    print(header)
    print("-" * len(header))
    for r in records:
        pass_str = f"{r['passes']}/{r['k']}"
        mark = "YES" if r["all_passed"] else "no"
        note = ""
        if not r["all_passed"]:
            note = "infra-limited (no FAIL, only ERROR)" if r["error_limited"] else ""
        print(
            f"{truncate(r['id'], id_width):<{id_width}} "
            f"{truncate(r['category'], cat_width):<{cat_width}} "
            f"{pass_str:<8} {mark:<12} {note}"
        )

    overall_hits = sum(1 for r in records if r["all_passed"])
    error_limited_misses = sum(
        1 for r in records if not r["all_passed"] and r["error_limited"]
    )
    reasoning_misses = sum(
        1 for r in records if not r["all_passed"] and not r["error_limited"]
    )

    print()
    print(
        f"pass@1 across {k} runs: mean {sum(run_rates) / len(run_rates) * 100:.0f}% "
        f"(min {min(run_rates) * 100:.0f}%, max {max(run_rates) * 100:.0f}%)"
    )
    print(
        f"pass^{k} overall: {overall_hits}/{len(records)} "
        f"({overall_hits / len(records) * 100:.0f}%)"
    )
    if error_limited_misses:
        print(
            f"  of {len(records) - overall_hits} task(s) that missed pass^{k}: "
            f"{reasoning_misses} had at least one FAIL (reasoning), "
            f"{error_limited_misses} were infra-limited only (ERROR, no FAIL) — "
            "the latter aren't evidence the agent reasoned incorrectly."
        )

    print()
    print(f"pass^{k} by category:")
    by_category: dict[str, list[TaskPassK]] = {}
    for r in records:
        by_category.setdefault(r["category"], []).append(r)
    for category, rows in by_category.items():
        hits = sum(1 for r in rows if r["all_passed"])
        print(f"  {category:<15} {hits}/{len(rows)}")


def write_passk_results(
    k: int,
    runs: list[list[Result]],
    records: list[TaskPassK],
    run_rates: list[float],
    provider: str,
    model: str,
) -> Path:
    """Write every pass's raw results plus the pass^k aggregate to eval/results/.

    Args:
        k: Number of passes.
        runs: Each pass's full Result list, in run order.
        records: The pass^k record for each task.
        run_rates: Each pass's own pass@1 rate, same order as runs.
        provider: Provider name, recorded in the file.
        model: Exact model slug, recorded in the file.

    Returns:
        The path written.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    path = RESULTS_DIR / f"passk-{k}-{stamp}.json"
    payload = {
        "k": k,
        "provider": provider,
        "model": model,
        "timestamp": stamp,
        "pass_at_1": {
            "rates": run_rates,
            "mean": sum(run_rates) / len(run_rates),
            "min": min(run_rates),
            "max": max(run_rates),
        },
        "pass_k_overall": sum(1 for r in records if r["all_passed"]) / len(records),
        "error_limited_misses": sum(
            1 for r in records if not r["all_passed"] and r["error_limited"]
        ),
        "tasks": records,
        "runs": runs,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-k",
        "--runs",
        type=int,
        default=3,
        help="Number of full passes over the task set. Default 3 (a "
        "smoke test of the mechanism); use 5 for the number that goes in "
        "the README.",
    )
    add_provider_args(parser)
    parser.add_argument(
        "--tasks",
        choices=("v1", "v2"),
        default="v1",
        help="Task set. v1 (default) is eval/tasks.json, unchanged. v2 is "
        "eval/tasks_v2.json, written to eval/results_v2/ as a separate "
        "baseline that is not comparable to v1.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the task set k times, then report and save pass@1 and pass^k."""
    if not TASKS_FILE.exists() or not FIXTURES_DIR.is_dir():
        print(
            f"Could not find {TASKS_FILE} and {FIXTURES_DIR}/ relative to "
            f"the current directory ({Path.cwd()}). Run this from the repo "
            "root: python eval/run_passk.py"
        )
        raise SystemExit(1)

    args = parse_args()
    config = apply_provider_args(args)
    if args.runs < 2:
        print(
            "k must be at least 2 — pass^1 is just pass@1. "
            "Use run_eval.py directly for a single run."
        )
        raise SystemExit(1)

    if args.tasks == "v2":
        import passk_v2  # lazy: only a v2 run needs it, and it imports this module

        passk_v2.main_passk_v2(v2_config(), config, args.runs, SLEEP_BETWEEN_RUNS_SECONDS)
        return

    tasks = load_tasks(TASKS_FILE)
    runs: list[list[Result]] = []

    print(f"Provider: {config.provider}  Model: {config.model}")

    for n in range(1, args.runs + 1):
        print(f"\n=== pass {n}/{args.runs} ===")
        pass_results = run_one_pass(tasks, n, args.runs)
        runs.append(pass_results)
        print_table(pass_results)
        print_summary(pass_results)
        if n < args.runs:
            print(
                f"\nPausing {SLEEP_BETWEEN_RUNS_SECONDS}s before the next "
                f"pass (provider: {config.provider}; this pause is sized for "
                "Groq's per-minute limit and may be more conservative "
                "than other providers need)..."
            )
            time.sleep(SLEEP_BETWEEN_RUNS_SECONDS)

    run_rates = [
        sum(1 for r in run if r["status"] == "PASS") / len(run) for run in runs
    ]
    records = aggregate(tasks, runs)
    print_passk_summary(records, run_rates)

    path = write_passk_results(
        args.runs, runs, records, run_rates, config.provider, config.model
    )
    print()
    print(f"Results: {path}")
    print(f"Traces:  {TRACES_DIR}/run-1/ .. run-{args.runs}/")


if __name__ == "__main__":
    main()
