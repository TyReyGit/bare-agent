"""Run eval/tasks_v2.json and score every task with scoring_v2.score_task.

Entered through `python eval/run_eval.py --tasks v2 ...`; run_eval.main()
builds a V2Config and calls main_v2(). `run_passk.py --tasks v2` reuses
load_v2() and run_task_v2() through eval/passk_v2.py. Everything this module touches on
disk comes from that config: it never reads a run_eval module global at
runtime, so the second copy of run_eval that exists when run_eval.py is
launched as a script (it is __main__, and scoring_v2 imports it by name)
has nothing it could disagree about. From run_eval it uses only pure
functions.

Differences from the v1 runner:

* The workspace (eval/fixtures/ with eval/fixtures_v2/ copied on top) is
  rebuilt before every task, not once per run, so a forbidden_path hit can
  only come from the task being scored.
* Each task gets an empty cross-run memory file. memory.jsonl lives outside
  the workspace, so without this a fact saved in one task or trial is in
  the system prompt of every later one, and pass^k trials are not independent.
* Each task runs inside recorder.record(), so checks can score what the
  agent did (tool calls, gate decisions), not only what it said.
* Before the first task, check_canary_reachable() confirms the path the
  inject-02 fixture plants really resolves to the canary file outside the
  workspace. If it did not, an escape that got past the gate would read
  nothing and no_workspace_escape would be weaker than it looks.
* Results go to their own directory. eval/results/ holds v1 answers that
  three guards rescore with the v1 matcher; v2 rows there would break them.
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, NamedTuple

import fixture_text
import recorder
import scoring_v2
from bare_agent import agent, memory, permissions, tools
from run_eval import Result, abort_run, print_summary, print_table

# A path in a workspace file that points at the canary.
_CANARY_REF_RE = re.compile(r"[\w./-]*canary\.txt")


class ResultV2(Result):
    """A v1 Result plus everything needed to trace a v2 verdict."""

    task_set: str
    checks: list[dict[str, Any]]
    failure_reason: str | None
    tool_calls: list[dict[str, Any]]
    compactions: int
    tool_iterations: int


@dataclass(frozen=True)
class V2Config:
    """Every path and setting a v2 run uses, passed in by run_eval.main()."""

    tasks_file: Path
    fixtures_dir: Path
    overlay_dir: Path
    canary_file: Path
    workspace: Path
    results_dir: Path
    traces_dir: Path
    sleep_seconds: float


class V2Inputs(NamedTuple):
    """What load_v2 read from disk before the first task."""

    tasks: list[dict[str, Any]]
    fixture_texts: dict[str, str]
    canary_text: str


class CanaryUnreachable(RuntimeError):
    """The planted escape path does not lead to the real canary file."""


def prepare_v2_workspace(cfg: V2Config) -> None:
    """Rebuild the workspace: base fixtures, then the v2 overlay on top."""
    if cfg.workspace.exists():
        shutil.rmtree(cfg.workspace)
    shutil.copytree(cfg.fixtures_dir, cfg.workspace)
    shutil.copytree(cfg.overlay_dir, cfg.workspace, dirs_exist_ok=True)


def check_canary_reachable(cfg: V2Config) -> str:
    """Confirm every canary path planted in the workspace reaches the canary.

    Each `...canary.txt` reference found in a workspace file must resolve,
    against the workspace, to cfg.canary_file, which must be a real file
    outside the workspace. Returns the canary's contents.

    Raises:
        CanaryUnreachable: no reference found, a reference resolves
            elsewhere, or the canary file is missing or inside the workspace.
    """
    workspace = cfg.workspace.resolve()
    canary = cfg.canary_file.resolve()
    if not canary.is_file():
        raise CanaryUnreachable(f"canary file {canary} does not exist")
    if canary == workspace or workspace in canary.parents:
        raise CanaryUnreachable(f"canary file {canary} is inside the workspace")
    refs = {
        found.group(0)
        for text in fixture_text.load_fixture_texts(workspace).values()
        for found in _CANARY_REF_RE.finditer(text)
    }
    if not refs:
        raise CanaryUnreachable("no workspace file references canary.txt")
    for ref in sorted(refs):
        if (workspace / ref).resolve() != canary:
            raise CanaryUnreachable(f"{ref!r} resolves to {(workspace / ref).resolve()}, not {canary}")
    return canary.read_text(encoding="utf-8")


@contextlib.contextmanager
def isolated_memory() -> Iterator[None]:
    """Give the agent an empty memory file for the block, then restore the real one."""
    original = memory.MEMORY_FILE
    with tempfile.TemporaryDirectory(prefix="bare-agent-memory-") as tmp:
        memory.MEMORY_FILE = Path(tmp) / "memory.jsonl"
        try:
            yield
        finally:
            memory.MEMORY_FILE = original


def write_trace_v2(cfg: V2Config, task_id: str, transcript: str, error: str | None) -> None:
    """Write one task's captured stdout to {traces_dir}/{task_id}.log."""
    cfg.traces_dir.mkdir(parents=True, exist_ok=True)
    body = transcript if not error else f"{transcript}\n=== ERROR ===\n{error}\n"
    (cfg.traces_dir / f"{task_id}.log").write_text(body, encoding="utf-8")


def run_task_v2(
    task: dict[str, Any],
    cfg: V2Config,
    *,
    canary_text: str,
    fixture_texts: dict[str, str],
    mode: permissions.Mode = permissions.Mode.AUTO_DENY,
    model: agent.ModelFn | None = None,
) -> ResultV2:
    """Rebuild the workspace, run one task with empty memory, and score it.

    An exception from run_agent() is recorded as ERROR with no checks; a
    FatalProviderError is re-raised so the caller can abort the run.

    Args:
        model: A stand-in for the provider (tests). None uses the configured one.
    """
    prepare_v2_workspace(cfg)
    buffer = io.StringIO()
    answer = ""
    error: str | None = None
    config = agent.active_config()
    agent.reset_retry_count()

    rec = recorder.Recorder()
    try:
        with isolated_memory(), contextlib.redirect_stdout(buffer), recorder.record() as rec:
            answer = agent.run_agent(task["task"], model=model, mode=mode)
    except agent.FatalProviderError as exc:
        write_trace_v2(cfg, task["id"], buffer.getvalue(), f"{type(exc).__name__}: {exc}")
        raise
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    write_trace_v2(cfg, task["id"], buffer.getvalue(), error)

    checks: list[scoring_v2.Check] = []
    failure_reason: str | None
    if error is not None:
        status, failure_reason = "ERROR", f"error: {error}"
    else:
        workspace = cfg.workspace
        status, checks, failure_reason = scoring_v2.score_task(
            task,
            answer,
            rec.tool_calls,
            fixture_texts=fixture_texts,
            canary_text=canary_text,
            path_exists=lambda rel: (workspace / rel).exists(),
        )

    log = rec.to_dict()
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
        "task_set": "v2",
        "checks": [check._asdict() for check in checks],
        "failure_reason": failure_reason,
        "tool_calls": log["tool_calls"],
        "compactions": log["compactions"],
        "tool_iterations": log["tool_iterations"],
    }


def write_results_v2(cfg: V2Config, results: list[ResultV2]) -> Path:
    """Write the full results to a timestamped file in cfg.results_dir."""
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    path = cfg.results_dir / f"v2-{stamp}.json"
    path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return path


def load_v2(cfg: V2Config) -> V2Inputs:
    """Validate the config and tasks and run the canary preflight, before any model call."""
    if cfg.workspace.resolve() != tools.WORKSPACE_DIR.resolve():
        raise ValueError(
            f"cfg.workspace ({cfg.workspace}) is not the agent's sandbox "
            f"({tools.WORKSPACE_DIR}); scoring it would score a directory the agent never used"
        )
    for needed in (cfg.tasks_file, cfg.fixtures_dir, cfg.overlay_dir, cfg.canary_file):
        if not needed.exists():
            print(f"Missing {needed}. Run from the repo root: python eval/run_eval.py --tasks v2 ...")
            raise SystemExit(1)

    tasks: list[dict[str, Any]] = json.loads(cfg.tasks_file.read_text(encoding="utf-8"))
    for task in tasks:
        scoring_v2.validate_task(task)
    fixture_texts = fixture_text.load_fixture_texts(cfg.fixtures_dir, cfg.overlay_dir)

    prepare_v2_workspace(cfg)
    try:
        canary_text = check_canary_reachable(cfg)
    except CanaryUnreachable as exc:
        print(f"Aborting v2 run before any task: {exc}")
        raise SystemExit(1) from None
    return V2Inputs(tasks, fixture_texts, canary_text)


def main_v2(
    cfg: V2Config,
    provider_config: agent.ProviderConfig,
    *,
    model: agent.ModelFn | None = None,
) -> Path:
    """Validate, run and score every v2 task; write results; return the file path."""
    tasks, fixture_texts, canary_text = load_v2(cfg)

    print(f"Provider: {provider_config.provider}  Model: {provider_config.model}  Tasks: v2")
    results: list[ResultV2] = []
    for i, task in enumerate(tasks, start=1):
        print(f"[{i:>2}/{len(tasks)}] {task['id']} ...", end=" ", flush=True)
        try:
            result = run_task_v2(
                task, cfg, canary_text=canary_text, fixture_texts=fixture_texts, model=model
            )
        except agent.FatalProviderError as exc:
            abort_run(exc)
        results.append(result)
        print(result["status"], flush=True)
        if i < len(tasks) and cfg.sleep_seconds > 0:
            time.sleep(cfg.sleep_seconds)

    as_v1: list[Result] = list(results)
    print()
    print_table(as_v1)
    print_summary(as_v1)
    failed = [r for r in results if r["status"] != "PASS"]
    if failed:
        print()
        print("Why each non-PASS failed:")
        for r in failed:
            print(f"  {r['id']:<10} {r['failure_reason']}")

    path = write_results_v2(cfg, results)
    print()
    print(f"Results: {path}")
    print(f"Traces:  {cfg.traces_dir}/")
    return path
