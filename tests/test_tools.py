from pathlib import Path

import pytest

from bare_agent import tools


@pytest.fixture(autouse=True)
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    ws = tmp_path / "workspace"
    monkeypatch.setattr(tools, "WORKSPACE_DIR", ws)
    return ws


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


def test_run_bash_command_not_found() -> None:
    result = tools.run_bash("this_command_does_not_exist_xyz")

    assert "Hint" in result


def test_run_bash_nonzero_exit_returns_stderr() -> None:
    result = tools.run_bash("ls /no/such/directory/xyz")

    assert "No such file or directory" in result


def test_run_bash_timeout() -> None:
    result = tools.run_bash("sleep 15")

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
