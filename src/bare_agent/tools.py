"""ReAct agent tools: sandboxed file reads, shell execution, and arithmetic.

Each function is a self-contained "tool" meant to be exposed to an LLM in a
ReAct-style loop. They never raise on expected failure modes; instead they
return a string that explains what went wrong and how to correct the next
call, since that string is what the calling model will see and act on.
"""

from __future__ import annotations

import ast
import operator
import shlex
import subprocess
from pathlib import Path
from typing import Callable

WORKSPACE_DIR = Path("workspace")

# Allowlist, not denylist: an unknown command is refused rather than permitted.
# A denylist fails open — shell syntax offers too many ways around it — while
# this fails closed. These are read-only inspection commands, which is all the
# agent needs inside a sandboxed workspace.
_ALLOWED_COMMANDS = frozenset(
    {"ls", "cat", "grep", "wc", "find", "head", "tail", "sort", "uniq", "echo"}
)

# Some allowlisted commands take options that turn them into execution or write
# primitives, which would defeat the allowlist without any shell involved:
# find's -exec family runs arbitrary programs, its -delete and -fprint* family
# write files, and sort's --compress-program runs a program while -o overwrites
# one. Scoped per command because the same spelling is harmless elsewhere —
# 'grep -o' and find's '-o' (logical OR) are both read-only.
_BANNED_OPTIONS: dict[str, frozenset[str]] = {
    "find": frozenset(
        {
            "-exec",
            "-execdir",
            "-ok",
            "-okdir",
            "-delete",
            "-fprintf",
            "-fprint",
            "-fls",
        }
    ),
    "sort": frozenset({"--compress-program", "-o", "--output"}),
}

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
        description: Run a read-only inspection command and return its
                stdout. Allowed: ls, cat, grep, wc, find, head, tail, sort,
                uniq, echo. No pipes, redirection or chaining. Times out
                after 10 seconds.
        parameters:
            command (string, required): The command to execute, e.g.
                "ls -la" or "grep overdue notes.txt".
    """
    if not command or not command.strip():
        return "No command given. Provide a non-empty shell command to run."

    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        parts = shlex.split(command)
    except ValueError as exc:
        return (
            f"Could not parse command '{command}': {exc}. Check for unmatched "
            "quotes."
        )

    if not parts:
        return "No command given. Provide a non-empty shell command to run."

    program = parts[0]
    if program not in _ALLOWED_COMMANDS:
        allowed = ", ".join(sorted(_ALLOWED_COMMANDS))
        return (
            f"Command '{program}' is not permitted. This sandbox allows only "
            f"read-only inspection commands: {allowed}. Pipes, redirection and "
            "command chaining are also unavailable — run one command at a time "
            "and combine the results yourself."
        )

    banned_option = _banned_option_in(program, parts[1:])
    if banned_option is not None:
        return (
            f"Option '{banned_option}' is not permitted: it lets '{program}' run "
            f"another program or write files, and this sandbox is inspection-only. "
            f"Re-run '{program}' without it."
        )

    globbed = [arg for arg in parts[1:] if "*" in arg or "?" in arg]
    if globbed:
        return (
            f"Argument '{globbed[0]}' looks like a glob pattern, but this "
            "sandbox runs commands directly rather than through a shell, so "
            "'*' and '?' are passed through literally and will not match any "
            "files. Name files explicitly, or use 'find' or 'grep -r' to "
            "search recursively instead."
        )

    # argv form, never shell=True: the allowlist is checked against these exact
    # tokens, so these must be what actually executes. Handing the raw string to
    # a shell would let it re-parse operators the allowlist never inspected.
    try:
        result = subprocess.run(
            parts,
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
    except OSError as exc:
        # With no shell in the loop, a missing or unrunnable binary raises here
        # instead of exiting non-zero. This function's contract is to return an
        # explanatory string rather than raise, so it has to be caught.
        return (
            f"Could not run '{program}': {exc}. It may not be installed or not "
            "on PATH."
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


def _banned_option_in(program: str, args: list[str]) -> str | None:
    """Return the first option in `args` that `program` must not be given.

    Returns None when every argument is acceptable. Long options are compared
    on their name only, so '--compress-program=gzip' and '--compress-program
    gzip' are both caught.
    """
    banned = _BANNED_OPTIONS.get(program)
    if banned is None:
        return None

    for arg in args:
        name = arg.split("=", 1)[0]
        if name in banned:
            return name

    return None


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
