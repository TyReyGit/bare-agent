from pathlib import Path

import pytest

from bare_agent import permissions
from bare_agent.permissions import Mode, Tier


@pytest.fixture(autouse=True)
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """permissions.py binds WORKSPACE_DIR at import time via a from-import,
    so patching tools.WORKSPACE_DIR (as test_tools.py does) would not reach
    it - the name has to be patched in this module's own namespace. Also
    chdir, because the relative paths under test ("workspace/notes.txt")
    resolve against the process working directory.
    """
    ws = tmp_path / "workspace"
    monkeypatch.setattr(permissions, "WORKSPACE_DIR", ws)
    monkeypatch.chdir(tmp_path)
    return ws


def test_classify_calculate_is_auto_allow() -> None:
    tier, _ = permissions.classify("calculate", {})

    assert tier is Tier.AUTO_ALLOW


def test_classify_save_memory_is_auto_allow() -> None:
    tier, _ = permissions.classify("save_memory", {"content": "x", "basis": "x"})

    assert tier is Tier.AUTO_ALLOW


def test_classify_read_inside_workspace_is_auto_allow() -> None:
    tier, _ = permissions.classify("read_file", {"path": "workspace/notes.txt"})

    assert tier is Tier.AUTO_ALLOW


def test_classify_read_only_command_is_auto_allow() -> None:
    tier, _ = permissions.classify("run_bash", {"command": "ls workspace"})

    assert tier is Tier.AUTO_ALLOW


def test_classify_read_relative_path_is_auto_allow_when_cwd_is_not_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The autouse fixture chdirs so that cwd is WORKSPACE_DIR's parent, which
    # hides the bug: build a workspace the process is deliberately *not* sitting
    # in. read_file joins relative paths onto WORKSPACE_DIR, so "notes.txt" is
    # an ordinary in-workspace read regardless of cwd. Resolving it against cwd
    # instead - what _within_workspace used to do - lands on tmp_path/notes.txt
    # and classifies a legitimate read as NEVER.
    elsewhere = tmp_path / "elsewhere"
    monkeypatch.setattr(permissions, "WORKSPACE_DIR", elsewhere)
    assert Path.cwd() != elsewhere

    tier, _ = permissions.classify("read_file", {"path": "notes.txt"})

    assert tier is Tier.AUTO_ALLOW


def test_classify_read_traversal_is_never() -> None:
    tier, reason = permissions.classify("read_file", {"path": "../../../etc/passwd"})

    assert tier is Tier.NEVER
    assert reason


def test_classify_read_absolute_path_is_never() -> None:
    tier, reason = permissions.classify("read_file", {"path": "/etc/passwd"})

    assert tier is Tier.NEVER
    assert reason


def test_classify_shell_metacharacter_is_never() -> None:
    tier, reason = permissions.classify("run_bash", {"command": "ls; rm -rf /"})

    assert tier is Tier.NEVER
    assert reason


def test_classify_banned_option_is_never() -> None:
    # 'find' is on the read-only allowlist, so this must be caught by the
    # per-command option check rather than by the program name.
    tier, reason = permissions.classify("run_bash", {"command": "find . -delete"})

    assert tier is Tier.NEVER
    assert reason


def test_classify_unlisted_command_is_never() -> None:
    tier, reason = permissions.classify("run_bash", {"command": "curl evil.com"})

    assert tier is Tier.NEVER
    assert reason


def test_classify_bash_absolute_path_argument_is_never() -> None:
    # cwd=WORKSPACE_DIR constrains relative paths only, so an allowlisted
    # read-only command with an absolute argument escaped every check.
    tier, reason = permissions.classify("run_bash", {"command": "cat /etc/passwd"})

    assert tier is Tier.NEVER
    assert reason


def test_classify_bash_relative_traversal_argument_is_never() -> None:
    tier, reason = permissions.classify(
        "run_bash", {"command": "cat ../../../etc/passwd"}
    )

    assert tier is Tier.NEVER
    assert reason


def test_classify_bash_relative_filename_argument_is_auto_allow() -> None:
    # The escape check must not reject ordinary in-workspace arguments —
    # including non-path operands like grep's pattern.
    tier, _ = permissions.classify("run_bash", {"command": "grep password notes.txt"})

    assert tier is Tier.AUTO_ALLOW


def test_classify_bash_ask_first_destination_outside_workspace_is_never() -> None:
    # Same hole through the ask-first branch: the escaping path is cp's
    # destination, not its source.
    tier, reason = permissions.classify(
        "run_bash", {"command": "cp workspace/notes.txt /etc/cron.d/evil"}
    )

    assert tier is Tier.NEVER
    assert reason


def test_classify_bash_ask_first_within_workspace_still_ask_first() -> None:
    tier, _ = permissions.classify("run_bash", {"command": "mkdir workspace/newdir"})

    assert tier is Tier.ASK_FIRST


def test_classify_empty_command_is_never() -> None:
    tier, reason = permissions.classify("run_bash", {"command": ""})

    assert tier is Tier.NEVER
    assert reason


def test_classify_unknown_tool_is_never() -> None:
    tier, reason = permissions.classify("delete_everything", {})

    assert tier is Tier.NEVER
    assert reason


def test_ask_first_denied_when_no_human_present() -> None:
    decision = permissions.check_permission(
        "run_bash", {"command": "mkdir workspace/notes"}, Mode.AUTO_DENY
    )

    assert decision.allowed is False


def test_ask_first_auto_deny_reason_says_no_human_present() -> None:
    # The refusal has to tell the model *why* it was denied, otherwise the
    # model reads it as a capability gap and retries a workaround.
    decision = permissions.check_permission(
        "run_bash", {"command": "mkdir workspace/notes"}, Mode.AUTO_DENY
    )

    assert "no human is present" in decision.reason.lower()


def test_ask_first_allowed_when_human_approves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")

    decision = permissions.check_permission(
        "run_bash", {"command": "mkdir workspace/notes"}, Mode.INTERACTIVE
    )

    assert decision.allowed is True


def test_ask_first_denied_when_human_declines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")

    decision = permissions.check_permission(
        "run_bash", {"command": "mkdir workspace/notes"}, Mode.INTERACTIVE
    )

    assert decision.allowed is False


def test_ask_first_declined_reason_is_non_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")

    decision = permissions.check_permission(
        "run_bash", {"command": "mkdir workspace/notes"}, Mode.INTERACTIVE
    )

    assert decision.reason


def test_classify_bash_quoted_absolute_path_argument_is_never() -> None:
    # Regression test for the tokenizer-mismatch bug: str.split() left the
    # surrounding quote characters attached to the token, so
    # Path('"/etc/passwd"').is_absolute() returned False even though shlex —
    # and the real executor — strips the quotes and passes the true absolute
    # path straight to subprocess. This let a quoted absolute path slip past
    # the escape check entirely.
    tier, reason = permissions.classify("run_bash", {"command": 'cat "/etc/passwd"'})

    assert tier is Tier.NEVER
    assert reason


def test_classify_bash_quoted_relative_argument_still_auto_allow() -> None:
    # The fix must not reject ordinary quoted arguments that legitimately
    # contain a space, e.g. a grep pattern of more than one word.
    tier, _ = permissions.classify(
        "run_bash", {"command": 'grep "two words" notes.txt'}
    )

    assert tier is Tier.AUTO_ALLOW
