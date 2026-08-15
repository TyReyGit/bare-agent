"""One-off live check: does cross-run memory reduce tool calls on a related
task? Not part of the pytest suite - run by hand against the real API.

Counts tool calls by wrapping agent._execute_tool, the same technique
tests/test_agent.py already uses to intercept without touching agent.py.

Usage:
    python scripts/live_check.py "task text here"
"""

from __future__ import annotations

import sys

from bare_agent import agent


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python scripts/live_check.py "task text"')
        raise SystemExit(1)
    task = sys.argv[1]

    call_count = 0
    original_execute = agent._execute_tool

    def counting_execute(call: object) -> str:
        nonlocal call_count
        call_count += 1
        return original_execute(call)  # type: ignore[arg-type]

    agent._execute_tool = counting_execute  # type: ignore[assignment]
    try:
        result = agent.run_agent(task)
    finally:
        agent._execute_tool = original_execute

    print(f"\n{'=' * 40}\ntool calls: {call_count}\nfinal answer: {result}")


if __name__ == "__main__":
    main()
