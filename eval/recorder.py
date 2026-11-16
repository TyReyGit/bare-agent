"""Record what the agent actually did during one eval task, from outside.

The eval runner only sees run_agent's final answer string; everything else
goes to stdout. Scoring tasks on behaviour (was an injected path touched, how
many tool calls were made, did compaction run) needs the real events, so this
patches three names for the duration of one task and restores them after.

Patch points, none of which need a change under src/:

* agent._execute_tool: run_agent resolves it as a module global on every
  call, so replacing the attribute is enough.
* agent.permissions.check_permission: agent.py does
  `from bare_agent import ..., permissions, ...` and _execute_tool calls
  `permissions.check_permission(name, arguments, mode)`, an attribute lookup
  on the module object. There is no `agent.check_permission`. The attribute
  being replaced is therefore the shared one on bare_agent.permissions, for
  the duration of the `with` block only. The wrapper calls the real function
  and stores the Decision it really returned; nothing is recomputed.
* agent._compact_if_needed: returns its input list unchanged when it does
  nothing, and a new list when it folded history. An identity check tells
  the two apart.

`tool_iterations` counts completed tool iterations (one per compaction
check). The final answering turn is not included, and an iteration cut short
by the loop detector returns before the check and is not counted either.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from bare_agent import agent, permissions


@dataclass
class ToolRecord:
    """One dispatched tool call."""

    tool: str
    args: Any
    decision: permissions.Decision | None
    result: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe form for result files."""
        decision = None
        if self.decision is not None:
            decision = {
                "tier": self.decision.tier.value,
                "allowed": self.decision.allowed,
                "reason": self.decision.reason,
            }
        return {
            "tool": self.tool,
            "args": self.args,
            "decision": decision,
            "result": self.result,
        }


@dataclass
class Recorder:
    """Everything observed during one `record()` block."""

    tool_calls: list[ToolRecord] = field(default_factory=list)
    compactions: int = 0
    tool_iterations: int = 0
    _pending: permissions.Decision | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "compactions": self.compactions,
            "tool_iterations": self.tool_iterations,
        }


def _parse_args(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


@contextmanager
def record() -> Iterator[Recorder]:
    """Patch the agent's dispatch, gate and compaction hooks; yield the log."""
    rec = Recorder()

    original_execute: Callable[..., str] = agent._execute_tool
    original_check: Callable[..., permissions.Decision] = permissions.check_permission
    original_compact: Callable[..., Any] = agent._compact_if_needed

    def check(
        tool_name: str, args: dict[str, object], mode: permissions.Mode
    ) -> permissions.Decision:
        decision = original_check(tool_name, args, mode)
        rec._pending = decision
        return decision

    def execute(tool_call: Any, mode: permissions.Mode) -> str:
        rec._pending = None
        name = tool_call.function.name
        args = _parse_args(tool_call.function.arguments)
        try:
            result = original_execute(tool_call, mode)
        except BaseException as exc:
            rec.tool_calls.append(
                ToolRecord(name, args, rec._pending, f"<raised {type(exc).__name__}: {exc}>")
            )
            raise
        rec.tool_calls.append(ToolRecord(name, args, rec._pending, result))
        return result

    def compact(messages: list[Any], keep_recent_exchanges: int) -> list[Any]:
        out: list[Any] = original_compact(messages, keep_recent_exchanges)
        rec.tool_iterations += 1
        if out is not messages:
            rec.compactions += 1
        return out

    setattr(agent, "_execute_tool", execute)
    setattr(permissions, "check_permission", check)
    setattr(agent, "_compact_if_needed", compact)
    try:
        yield rec
    finally:
        setattr(agent, "_execute_tool", original_execute)
        setattr(permissions, "check_permission", original_check)
        setattr(agent, "_compact_if_needed", original_compact)
