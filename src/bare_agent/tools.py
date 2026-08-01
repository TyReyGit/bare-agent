"""ReAct agent tools: sandboxed file reads, shell execution, and arithmetic.

Each function is a self-contained "tool" meant to be exposed to an LLM in a
ReAct-style loop. They never raise on expected failure modes; instead they
return a string that explains what went wrong and how to correct the next
call, since that string is what the calling model will see and act on.
"""

from __future__ import annotations

import ast
import operator
import subprocess
from pathlib import Path
from typing import Callable

WORKSPACE_DIR = Path("workspace")

_BASH_TIMEOUT_SECONDS = 10

_BIN_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def read_file(path: str) -> str:
    """Read a text file from the sandboxed workspace/ directory.

    Args:
        path: Path to the file, relative to workspace/. Absolute paths and
            paths containing '..' are rejected.

    Returns:
        The file's contents as a string on success. On failure, a message
        describing the problem and, where useful, the files actually
        available in workspace/ so the caller can retry with a valid path.

    Tool schema notes:
        name: read_file
        description: Read the contents of a file from the workspace directory.
        parameters:
            path (string, required): Relative path to the file within
                workspace/, e.g. "notes.txt" or "data/input.csv".
    """
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

    if not path or not path.strip():
        return "No path given. Provide a non-empty relative path within workspace/."

    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts:
        return (
            f"Invalid path '{path}': only relative paths inside workspace/ are "
            "allowed, and '..' segments are not permitted. Provide a path like "
            "'file.txt' or 'subdir/file.txt'."
        )

    resolved_workspace = WORKSPACE_DIR.resolve()
    resolved_target = (WORKSPACE_DIR / candidate).resolve()
    if resolved_workspace not in resolved_target.parents and resolved_target != resolved_workspace:
        return (
            f"Invalid path '{path}': resolves outside of workspace/. Provide a "
            "path contained within the workspace directory."
        )

    if not resolved_target.exists():
        available = sorted(
            str(p.relative_to(resolved_workspace))
            for p in resolved_workspace.rglob("*")
            if p.is_file()
        )
        listing = ", ".join(available) if available else "(none)"
        return f"File '{path}' not found in workspace/. Files present: [{listing}]"

    if resolved_target.is_dir():
        entries = sorted(p.name for p in resolved_target.iterdir())
        listing = ", ".join(entries) if entries else "(empty)"
        return f"'{path}' is a directory, not a file. Contents: [{listing}]"

    try:
        return resolved_target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return (
            f"File '{path}' could not be decoded as UTF-8 text. It may be a "
            "binary file; this tool only supports reading text files."
        )
    except OSError as exc:
        return f"Could not read '{path}': {exc}"


def run_bash(command: str) -> str:
    """Execute a shell command and return its output.

    Args:
        command: The shell command to run.

    Returns:
        Standard output on success. On failure (non-zero exit, timeout, or
        the command not being found), a message including stderr and a hint
        about likely causes so the caller can adjust the command.

    Tool schema notes:
        name: run_bash
        description: Run a shell command and return its stdout. Times out
            after 10 seconds.
        parameters:
            command (string, required): The command to execute, e.g.
                "ls -la" or "python script.py".
    """
    if not command or not command.strip():
        return "No command given. Provide a non-empty shell command to run."

    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=_BASH_TIMEOUT_SECONDS,
            cwd=WORKSPACE_DIR,
        )
    except subprocess.TimeoutExpired:
        return (
            f"Command timed out after {_BASH_TIMEOUT_SECONDS} seconds: '{command}'. "
            "Try a faster command, or one that doesn't block waiting for input."
        )

    if result.returncode == 0:
        return result.stdout

    stderr = result.stderr.strip()
    hint = ""
    lowered = stderr.lower()
    if "not found" in lowered or "not recognized" in lowered:
        hint = (
            "\nHint: the program may not be installed or not on PATH. Check "
            "the spelling, or try an alternative command."
        )
    elif "permission denied" in lowered:
        hint = "\nHint: the command lacks permission to run or access a resource."
    elif "no such file or directory" in lowered:
        hint = "\nHint: check that the file or directory path is correct."

    message = stderr if stderr else f"Command exited with status {result.returncode}."
    return f"{message}{hint}"


def calculate(expression: str) -> str:
    """Safely evaluate a basic arithmetic expression.

    Supports +, -, *, /, //, %, ** and parentheses over int/float literals.
    Does not use eval(); the expression is parsed into an AST and only a
    whitelist of numeric operator nodes is evaluated.

    Args:
        expression: The arithmetic expression to evaluate, e.g. "2 + 3 * 4".

    Returns:
        The numeric result as a string on success. On failure, a message
        explaining what was invalid about the expression.

    Tool schema notes:
        name: calculate
        description: Evaluate a basic arithmetic expression (+, -, *, /, //,
            %, **, parentheses) and return the numeric result.
        parameters:
            expression (string, required): The arithmetic expression to
                evaluate, e.g. "(2 + 3) * 4".
    """
    if not expression or not expression.strip():
        return "No expression given. Provide an arithmetic expression, e.g. '2 + 2'."

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        return f"Invalid expression '{expression}': {exc.msg}."

    try:
        result = _eval_node(tree.body)
    except _CalculationError as exc:
        return f"Invalid expression '{expression}': {exc}"
    except ZeroDivisionError:
        return f"Invalid expression '{expression}': division by zero."

    return str(result)


class _CalculationError(Exception):
    """Raised for expression content that calculate() cannot evaluate."""


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise _CalculationError(
                f"only numeric literals are allowed, got {node.value!r}."
            )
        return node.value

    if isinstance(node, ast.BinOp):
        op_func = _BIN_OPS.get(type(node.op))
        if op_func is None:
            raise _CalculationError(
                f"operator '{type(node.op).__name__}' is not supported. "
                "Allowed operators: + - * / // % **."
            )
        return op_func(_eval_node(node.left), _eval_node(node.right))

    if isinstance(node, ast.UnaryOp):
        unary_op_func = _UNARY_OPS.get(type(node.op))
        if unary_op_func is None:
            raise _CalculationError(
                f"unary operator '{type(node.op).__name__}' is not supported."
            )
        return unary_op_func(_eval_node(node.operand))

    raise _CalculationError(
        f"unsupported syntax: '{type(node).__name__}'. Only numbers and "
        "+ - * / // % ** with parentheses are allowed."
    )
