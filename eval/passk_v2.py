"""pass^k over eval/tasks_v2.json: a new baseline, not comparable to v1.

Entered through `python eval/run_passk.py --tasks v2 ...`. Every path comes
from the V2Config it is given. Running and scoring a task is runner_v2's
job; folding k trials into pass^k, including the infra-limited rule (ERROR
is never a pass, and a miss with ERRORs but no FAIL is flagged), is
run_passk.aggregate, so each rule has one definition.

v2 differs from v1 in tasks, scoring and memory isolation, and 18/21 is the
same fraction as v1's 12/14. So that a v2 number cannot pass for a v1 one:

* results are counts: no percentage and no rate is printed or stored;
* the result keys differ (pass_k_v2, trials), so code reading a v1 file
  raises KeyError on a v2 one instead of comparing;
* the headline line carries its own "not comparable" tag;
* the file records what it measured (identity). Two v2 files compare only
  if task_set_sha256, scorer_sha256, k, provider and model all match.

`retries` is the agent's own retry count. A retry inside a provider SDK
(Groq's client makes up to two) is not visible here.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import subprocess
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict, cast

import run_eval
import runner_v2
import scoring_v2
from bare_agent import agent
from run_eval import Result, Task, abort_run, print_table, truncate
from run_passk import TaskPassK, aggregate

SCHEMA = "passk-v2/1"
NOT_COMPARABLE = "v2 baseline, not comparable to v1"
COMPARABILITY_NOTE = (
    "Different task set (21 tasks), stricter scoring and fresh memory per trial. "
    "Do not compare with any v1 pass^k. Compare with another v2 file only if "
    "task_set_sha256, scorer_sha256, k, provider and model all match."
)
_CONSOLE_WIDTH = 200


class TrialV2(runner_v2.ResultV2):
    """One task in one trial."""

    trial: int


class FailedCheck(TypedDict):
    """One check that failed in one trial."""

    name: str
    detail: str


class TrialFailure(TypedDict):
    """Why one trial of one task did not pass."""

    trial: int
    status: str
    failure_reason: str | None
    failed_checks: list[FailedCheck]
    trace: str


class TaskPassKV2(TaskPassK):
    """A pass^k record plus the cause of each non-passing trial."""

    failures: list[TrialFailure]


def _git(*args: str) -> str | None:
    try:
        done = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).resolve().parent,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def task_set_sha256(cfg: runner_v2.V2Config) -> str:
    """Hash of what a trial is given: tasks, both fixture trees and the canary."""
    files = [("tasks", cfg.tasks_file), ("canary", cfg.canary_file)]
    for label, root in (("fixtures", cfg.fixtures_dir), ("overlay", cfg.overlay_dir)):
        files += [
            (f"{label}/{path.relative_to(root).as_posix()}", path)
            for path in sorted(root.rglob("*"))
            if path.is_file()
        ]
    digest = hashlib.sha256()
    for name, path in files:
        digest.update(f"{name}\0{hashlib.sha256(path.read_bytes()).hexdigest()}\n".encode())
    return digest.hexdigest()


def scorer_sha256() -> str:
    """Hash of what decides a verdict: scoring_v2 and the v1 matcher it delegates to.

    Only the matcher is taken from run_eval, so a CLI edit there does not
    make two otherwise identical baselines look different.
    """
    parts = (
        Path(scoring_v2.__file__).read_text(encoding="utf-8"),
        inspect.getsource(run_eval.matches),
        inspect.getsource(run_eval._normalise_number),
        repr((run_eval.REFUSAL_PHRASES, run_eval.BAD_DATA_PHRASES)),
    )
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()


def run_identity(cfg: runner_v2.V2Config) -> dict[str, Any]:
    """What this run measured. git_dirty ignores untracked files, such as results."""
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "task_set_sha256": task_set_sha256(cfg),
        "scorer_sha256": scorer_sha256(),
    }


def run_trial(
    cfg: runner_v2.V2Config,
    inputs: runner_v2.V2Inputs,
    trial: int,
    k: int,
    *,
    model: agent.ModelFn | None = None,
) -> list[TrialV2]:
    """Rebuild, re-check the canary, then run every task once."""
    runner_v2.prepare_v2_workspace(cfg)
    try:
        canary_text = runner_v2.check_canary_reachable(cfg)
    except runner_v2.CanaryUnreachable as exc:
        print(f"Aborting v2 pass^k before trial {trial}: {exc}\nNo results file was written.")
        raise SystemExit(1) from None

    rows: list[TrialV2] = []
    total = len(inputs.tasks)
    for i, task in enumerate(inputs.tasks, start=1):
        print(f"[v2 run {trial}/{k}] [{i:>2}/{total}] {task['id']} ...", end=" ", flush=True)
        try:
            result = runner_v2.run_task_v2(
                task, cfg, canary_text=canary_text, fixture_texts=inputs.fixture_texts, model=model
            )
        except agent.FatalProviderError as exc:
            abort_run(exc)
        rows.append({**result, "trial": trial})
        print(result["status"], flush=True)
        if i < total and cfg.sleep_seconds > 0:
            time.sleep(cfg.sleep_seconds)
    return rows


def aggregate_v2(
    tasks: list[dict[str, Any]], trials: list[list[TrialV2]], traces_root: Path
) -> list[TaskPassKV2]:
    """run_passk.aggregate, plus every failing check of every non-passing trial."""
    as_v1: list[list[Result]] = [list(rows) for rows in trials]
    records: list[TaskPassKV2] = []
    for i, record in enumerate(aggregate(cast("list[Task]", tasks), as_v1)):
        failures: list[TrialFailure] = [
            {
                "trial": row["trial"],
                "status": row["status"],
                "failure_reason": row["failure_reason"],
                "failed_checks": [
                    {"name": check["name"], "detail": check["detail"]}
                    for check in row["checks"]
                    if not check["passed"]
                ],
                "trace": str(traces_root / f"run-{row['trial']}" / f"{row['id']}.log"),
            }
            for row in (rows[i] for rows in trials)
            if row["status"] != "PASS"
        ]
        records.append({**record, "failures": failures})
    return records


def _one_line(text: str) -> str:
    return truncate(" ".join(text.split()), _CONSOLE_WIDTH)


def print_banner(task_count: int) -> None:
    """Printed first and last, so either end of a pasted log carries it."""
    print(
        f"=== EVAL V2 ({task_count} tasks): NEW BASELINE, not comparable to any v1 "
        "number (including the 12/14 in the README) ==="
    )


def print_passk_v2(records: list[TaskPassKV2], k: int) -> None:
    """The per-task table with each failing trial's cause, then the tagged headline."""
    print()
    header = f"{'ID':<11} {'CATEGORY':<15} {'PASSES':<7} {'ALL':<4} NOTE"
    print(header)
    print("-" * len(header))
    for record in records:
        mark = "YES" if record["all_passed"] else "no"
        note = "infra-limited (no FAIL, only ERROR)" if record["error_limited"] else ""
        passes = f"{record['passes']}/{k}"
        print(
            f"{record['id']:<11} {truncate(record['category'], 15):<15} "
            f"{passes:<7} {mark:<4} {note}".rstrip()
        )
        for failure in record["failures"]:
            causes = [f"{c['name']}: {c['detail']}" for c in failure["failed_checks"]]
            if not causes:  # an ERROR has no checks
                causes = [failure["failure_reason"] or ""]
            lead = f"  trial {failure['trial']} {failure['status']:<6}"
            for cause in causes:
                print(f"{lead} {_one_line(cause)}")
                lead = " " * len(lead)

    passed = sum(1 for r in records if r["all_passed"])
    limited = sum(1 for r in records if r["error_limited"])
    infra = f" ({limited} of the misses infra-limited)" if limited else ""
    print()
    print(f"v2 pass^{k}: {passed}/{len(records)} tasks{infra} [{NOT_COMPARABLE}]")


def main_passk_v2(
    cfg: runner_v2.V2Config,
    provider_config: agent.ProviderConfig,
    k: int,
    pause_seconds: float,
    *,
    model: agent.ModelFn | None = None,
) -> Path:
    """Run every v2 task k times; report and write v2 pass^k; return the file path.

    Nothing is written to results_dir unless all k trials complete.

    Args:
        model: A stand-in for the provider (tests). None uses the configured one.
    """
    inputs = runner_v2.load_v2(cfg)
    identity = run_identity(cfg)
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    traces_root = cfg.traces_dir / f"passk-{stamp}"

    print_banner(len(inputs.tasks))
    print(f"Provider: {provider_config.provider}  Model: {provider_config.model}  Tasks: v2")

    trials: list[list[TrialV2]] = []
    for n in range(1, k + 1):
        print(f"\n=== v2 trial {n}/{k} ===")
        trial_cfg = replace(cfg, traces_dir=traces_root / f"run-{n}")
        rows = run_trial(trial_cfg, inputs, n, k, model=model)
        trials.append(rows)
        print_table(list(rows))
        passed = sum(1 for row in rows if row["status"] == "PASS")
        print(f"v2 pass@1, trial {n}: {passed}/{len(rows)} tasks [{NOT_COMPARABLE}]")
        if n < k and pause_seconds > 0:
            print(f"\nPausing {pause_seconds}s before the next trial...")
            time.sleep(pause_seconds)

    records = aggregate_v2(inputs.tasks, trials, traces_root)
    print_passk_v2(records, k)

    payload = {
        "schema": SCHEMA,
        "task_set": "v2",
        "comparable_to_v1": False,
        "comparability_note": COMPARABILITY_NOTE,
        "k": k,
        "provider": provider_config.provider,
        "model": provider_config.model,
        "timestamp": stamp,
        "identity": identity,
        "pass_k_v2": {
            "passed": sum(1 for r in records if r["all_passed"]),
            "tasks": len(records),
            "infra_limited": sum(1 for r in records if r["error_limited"]),
        },
        "pass_at_1_v2": [
            {
                "trial": n,
                "passed": sum(1 for row in rows if row["status"] == "PASS"),
                "tasks": len(rows),
            }
            for n, rows in enumerate(trials, start=1)
        ],
        "tasks": records,
        "trials": trials,
    }
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.results_dir / f"passk-v2-{k}-{stamp}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print()
    print(f"Results: {path}")
    print(f"Traces:  {traces_root}/run-1/ .. run-{k}/")
    print_banner(len(inputs.tasks))
    return path
