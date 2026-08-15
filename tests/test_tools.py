import getpass
from pathlib import Path

import pytest
import subprocess

from bare_agent import memory, tools


@pytest.fixture(autouse=True)
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    ws = tmp_path / "workspace"
    monkeypatch.setattr(tools, "WORKSPACE_DIR", ws)
    return ws


@pytest.fixture(autouse=True)
def isolated_memory_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """save_memory calls straight through to memory.write_derived_memory,
    so any test that touches it needs its own memory.jsonl - otherwise it
    writes to the real project file. Separate tmp_path subdirectory from
    the workspace fixture above so the two never collide.
    """
    monkeypatch.setattr(memory, "MEMORY_FILE", tmp_path / "memory-isolated" / "memory.jsonl")


def test_read_file_success(workspace: Path) -> None:
    (workspace).mkdir(parents=True, exist_ok=True)
    (workspace / "notes.txt").write_text("hello world", encoding="utf-8")

    result = tools.read_file("notes.txt")

    assert result == "hello world"


def test_read_file_not_found_lists_available_files(workspace: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "a.txt").write_text("a", encoding="utf-8")

    result = tools.read_file("missing.txt")

    assert "not found in workspace/" in result
    assert "a.txt" in result


def test_read_file_rejects_path_traversal(workspace: Path) -> None:
    result = tools.read_file("../secret.txt")

    assert "Invalid path" in result
    assert ".." in result or "not permitted" in result


def test_read_file_rejects_absolute_path(workspace: Path) -> None:
    result = tools.read_file("/etc/passwd")

    assert "Invalid path" in result


def test_read_file_creates_workspace_if_missing(workspace: Path) -> None:
    assert not workspace.exists()

    tools.read_file("anything.txt")

    assert workspace.exists()


def test_run_bash_success() -> None:
    result = tools.run_bash("echo hello")

    assert result.strip() == "hello"


def test_run_bash_rejects_disallowed_command() -> None:
    result = tools.run_bash("this_command_does_not_exist_xyz")
    assert "not permitted" in result
    # The refusal must name the alternatives: the model reads this string and
    # needs to know what it *can* run, not just that it failed.
    assert "grep" in result


def test_run_bash_blocks_privilege_escalation() -> None:
    result = tools.run_bash("sudo -l")
    assert "not permitted" in result


def test_run_bash_does_not_interpret_shell_operators() -> None:
    # Without shell=True, '&&' is a literal argument rather than an operator,
    # so the second command never runs. Uses 'ls' rather than 'echo': this
    # system has a cargo-installed coreutils 'echo' on PATH that can block.
    result = tools.run_bash("ls && whoami")
    # 'ls' receives '&&' and 'whoami' as filenames; neither exists.
    assert "whoami" in result
    assert "No such file" in result or "cannot access" in result


def test_run_bash_blocks_find_exec() -> None:
    # 'find' is allowlisted, but -exec makes it an arbitrary-execution
    # primitive with no shell involved at all.
    result = tools.run_bash("find . -maxdepth 0 -exec id -un ;")

    assert "not permitted" in result
    assert "-exec" in result
    # The actual security property: 'id -un' never ran, so the caller's
    # username is nowhere in the output.
    assert getpass.getuser() not in result


def test_run_bash_blocks_find_delete() -> None:
    result = tools.run_bash("find . -name '*.txt' -delete")

    assert "not permitted" in result
    assert "-delete" in result


def test_run_bash_blocks_sort_compress_program() -> None:
    result = tools.run_bash("sort --compress-program=id notes.txt")

    assert "not permitted" in result
    assert "--compress-program" in result
    assert getpass.getuser() not in result


def test_run_bash_allows_read_only_options_that_look_similar(workspace: Path) -> None:
    # '-o' is banned for sort but is legitimate for grep (--only-matching)
    # and for find (logical OR); the check must be scoped per command.
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "notes.txt").write_text("overdue invoice\n", encoding="utf-8")

    result = tools.run_bash("grep -o overdue notes.txt")

    assert result.strip() == "overdue"
    assert "not permitted" not in result


def test_run_bash_nonzero_exit_returns_stderr() -> None:
    result = tools.run_bash("ls /no/such/directory/xyz")

    assert "No such file or directory" in result


def test_run_bash_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    # 'sleep' is no longer allowlisted, so trigger the timeout path directly
    # by making subprocess.run raise what a real timeout would raise.
    def fake_run(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired(cmd="cat", timeout=1)

    monkeypatch.setattr("bare_agent.tools.subprocess.run", fake_run)
    result = tools.run_bash("cat notes.txt")
    assert "timed out" in result


def test_calculate_success() -> None:
    result = tools.calculate("2 + 3 * 4")

    assert result == "14"


def test_calculate_parentheses_and_division() -> None:
    result = tools.calculate("(2 + 3) / 5")

    assert result == "1.0"


def test_calculate_rejects_syntax_error() -> None:
    result = tools.calculate("2 + * 3")

    assert "Invalid expression" in result


def test_calculate_rejects_non_arithmetic_syntax() -> None:
    result = tools.calculate("__import__('os').system('echo hi')")

    assert "Invalid expression" in result


def test_calculate_rejects_division_by_zero() -> None:
    result = tools.calculate("1 / 0")

    assert "division by zero" in result


def test_save_memory_success() -> None:
    result = tools.save_memory("tests run via pytest tests/ -v", "output of run_bash")

    assert "unverified" in result
    assert "tests run via pytest tests/ -v" in result


def test_save_memory_rejects_empty_content() -> None:
    result = tools.save_memory("", "some basis")

    assert "No content given" in result


def test_save_memory_rejects_empty_basis() -> None:
    result = tools.save_memory("some fact", "")

    assert "No basis given" in result


def test_save_memory_stores_as_derived_tier() -> None:
    # Wiring check, not a re-test of memory.py's tier logic (that's
    # covered thoroughly, including the adversarial case, in
    # test_memory.py). This just confirms the tool actually reaches
    # write_derived_memory rather than, say, silently doing nothing.
    tools.save_memory("an observed fact", "run_bash output")

    entries = memory.load_memory()
    assert len(entries) == 1
    assert entries[0].tier == "DERIVED"
