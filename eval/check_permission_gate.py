"""One-off check: run just the perm-01 task and report what actually happened.

Not part of the regular eval suite - run_eval.py and run_passk.py both
already pick up perm-01 automatically since it's just another row in
tasks.json. This script exists because a first live check of a brand-new
scoring path (BLOCK_EXPECTED) is worth confirming in isolation before
spending API quota running it inside a full 14-or-15-task pass, let alone a
k=3 or k=5 pass^k run across all of them.

Prints the task's PASS/FAIL/ERROR status, the agent's own final answer (for
context - not what scoring is actually based on), and, explicitly, whether
notes_backup.txt exists on disk after the run. That last line is the real
check: see run_eval.py's module docstring for why a BLOCK_EXPECTED task is
scored against the filesystem rather than the model's self-report.

Run from the repo root:
    AGENT_PROVIDER=gemini python eval/check_permission_gate.py
"""

from __future__ import annotations

from bare_agent import tools
from run_eval import FIXTURES_DIR, TASKS_FILE, load_tasks, prepare_workspace, run_task


def main() -> None:
    tasks = load_tasks(TASKS_FILE)
    task = next((t for t in tasks if t["id"] == "perm-01"), None)
    if task is None:
        print("perm-01 not found in eval/tasks.json - check it was added correctly.")
        raise SystemExit(1)

    print(f"Task: {task['task']}")
    print(f"Expect: {task['expect']}, forbidden_path: {task.get('forbidden_path')}")
    print()

    prepare_workspace()
    result = run_task(task)

    print(f"Status: {result['status']}")
    print(f"Agent's own final answer: {result['answer']!r}")
    if result["error"]:
        print(f"Error: {result['error']}")

    forbidden = tools.WORKSPACE_DIR / str(task.get("forbidden_path"))
    print()
    print(f"Filesystem check - does {forbidden.name} exist? {forbidden.exists()}")
    print(f"(The task PASSES only when this is False - see run_eval.py)")


if __name__ == "__main__":
    main()
