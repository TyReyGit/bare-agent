"""ReAct agent tools: sandboxed file reads, shell execution, and arithmetic.

Each function is a self-contained "tool" meant to be exposed to an LLM in a
ReAct-style loop. They never raise on expected failure modes; instead they
return a string that explains what went wrong and how to correct the next
call, since that string is what the calling model will see and act on.

One function here is deliberately NOT a tool. run_bash_approved() is the
executor for the ask-first permission tier and must never appear in the
registry the model selects from. See its docstring.
"""

from __future__ import annotations

import ast
import operator
import shlex
import subprocess
from pathlib import Path
from typing import Callable

from bare_agent import memory

# Anchored to the repo root via this file's location, not the process
# working directory — a relative path would make "inside the workspace"
# mean something different depending on where the agent was launched from.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WORKSPACE_DIR = _REPO_ROOT / "workspace"

# Allowlist, not denylist: an unknown command is refused rather than permitted.
# A denylist fails open — shell syntax offers too many ways around it — while
# this fails closed. These are read-only inspection commands, which is all the
# agent needs inside a sandboxed workspace.
#
# Public (no leading underscore): permissions.classify() imports this directly
# so the permission gate and this executor can never disagree about which
# programs are read-only. Same reasoning as prepare_command being public.
ALLOWED_COMMANDS = frozenset(
    {"ls", "cat", "grep", "wc", "find", "head", "tail", "sort", "uniq", "echo"}
)

# Workspace-confined, additive mutations. Not new agent capability — these
# stay refused with no human present. They exist so a human can approve
# them interactively; that's what makes the ask-first tier real rather
# than ceremonial. 'rm' is deliberately absent — deletion gets its own
# decision later, not a quiet arrival inside this batch.
#
# Public for the same reason as ALLOWED_COMMANDS above.
ASK_FIRST_COMMANDS = frozenset({"mkdir", "touch", "cp", "mv"})

# The set reachable only through run_bash_approved(), i.e. only after a human
# has said yes. run_bash() itself is still limited to ALLOWED_COMMANDS, so
# widening this set cannot widen what the model can reach unattended.
_APPROVED_COMMANDS = ALLOWED_COMMANDS | ASK_FIRST_COMMANDS

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


def extract_program(command: str) -> str | None:
    """Return the first token of a shell command, or None if unparseable.

    Used by permissions.classify() to determine a run_bash call's tier from
    the program name alone, before prepare_command() performs full
    validation. Kept separate from prepare_command so a banned-option or
    glob failure — a validation concern, not a permission concern — can
    never silently prevent tier classification: classify() needs an answer
    for every command string, including ones that will go on to be refused.
    """
    if not command or not command.strip():
        return None
    try:
        parts = shlex.split(command)
    except ValueError:
        return None
    return parts[0] if parts else None


def prepare_command(
    command: str, allowed: frozenset[str]
) -> tuple[list[str] | None, str | None]:
    """Parse and validate a command string against an allowlist.

    Exactly one of the two returned values is None. On success, returns the
    argv token list and None. On refusal, returns None and the explanatory
    string the caller should surface.

    Public rather than private because permissions.classify() imports it: the
    permission gate and the executor must agree about what a command is, and
    the only way to guarantee that is for them to run the same parser rather
    than two implementations that can drift apart.

    Args:
        command: The raw command string.
        allowed: The set of program names permitted for this call site.
    """
    if not command or not command.strip():
        return None, "No command given. Provide a non-empty shell command to run."

    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        parts = shlex.split(command)
    except ValueError as exc:
        return None, (
            f"Could not parse command '{command}': {exc}. Check for unmatched "
            "quotes."
        )

    if not parts:
        return None, "No command given. Provide a non-empty shell command to run."

    program = parts[0]
    if program not in allowed:
        allowed_list = ", ".join(sorted(allowed))
        return None, (
            f"Command '{program}' is not permitted. This sandbox allows only "
            f"read-only inspection commands: {allowed_list}. Pipes, redirection "
            "and command chaining are also unavailable — run one command at a "
            "time and combine the results yourself."
        )

    banned_option = _banned_option_in(program, parts[1:])
    if banned_option is not None:
        return None, (
            f"Option '{banned_option}' is not permitted: it lets '{program}' run "
            f"another program or write files, and this sandbox is inspection-only. "
            f"Re-run '{program}' without it."
        )

    globbed = [arg for arg in parts[1:] if "*" in arg or "?" in arg]
    if globbed:
        return None, (
            f"Argument '{globbed[0]}' looks like a glob pattern, but this "
            "sandbox runs commands directly rather than through a shell, so "
            "'*' and '?' are passed through literally and will not match any "
            "files. Name files explicitly, or use 'find' or 'grep -r' to "
            "search recursively instead."
        )

    return parts, None


def _execute(parts: list[str], command: str) -> str:
    """Run a validated argv list inside the workspace and format the result.

    Assumes `parts` has already been through prepare_command(). Kept separate
    so both executors share one subprocess call site and one error contract.
    """
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
            f"Could not run '{parts[0]}': {exc}. It may not be installed or not "
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


def run_bash(command: str) -> str:
    """Execute a read-only inspection command and return its output.

    This is the model-facing executor. Its allowlist is ALLOWED_COMMANDS and
    nothing else; the mutating commands in ASK_FIRST_COMMANDS are refused here
    no matter what the permission gate decided, so a bypassed or unwired gate
    cannot make this function more permissive than it is today.

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
    parts, error = prepare_command(command, ALLOWED_COMMANDS)
    if error is not None:
        return error
    assert parts is not None
    return _execute(parts, command)


def run_bash_approved(command: str) -> str:
    """Execute a command a human has explicitly approved. NOT A TOOL.

    This function must never be registered in the tool dispatch table. The
    model selects tools by emitting a name; if this name is not in the table,
    no string the model can produce reaches this code path. That is what makes
    the ask-first tier a constraint rather than a preference — the same
    argument as memory's VERIFIED tier being reachable only from __main__.

    The only legitimate caller is the human-approval branch of the agent loop,
    after a person has answered yes at an interactive prompt.

    Beyond the wider allowlist, this adds a check run_bash does not need: the
    mutating commands can write anywhere the process can reach, and cwd only
    governs where *relative* paths resolve. Absolute paths and '..' segments
    are therefore rejected outright, matching read_file. Relative paths without
    '..' always resolve inside the workspace, so nothing legitimate is refused.

    Args:
        command: The approved command string, as shown to the human.

    Returns:
        Standard output on success, or an explanatory failure string.
    """
    parts, error = prepare_command(command, _APPROVED_COMMANDS)
    if error is not None:
        return error
    assert parts is not None

    escaping = _path_escaping_workspace(parts[1:])
    if escaping is not None:
        return (
            f"Argument '{escaping}' points outside the workspace directory. "
            "Approved commands are still confined to workspace/: use a relative "
            "path with no '..' segments, e.g. 'notes.txt' or 'subdir/notes.txt'."
        )

    return _execute(parts, command)


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


def _path_escaping_workspace(args: list[str]) -> str | None:
    """Return the first argument that could name a location outside workspace/.

    Arguments beginning with '-' are treated as options and skipped. Everything
    else is treated as a possible path and must be relative with no '..'
    segments. Non-path arguments (a grep pattern, an echo string) are relative
    and '..'-free by construction, so they pass without special-casing.
    """
    for arg in args:
        if arg.startswith("-"):
            continue
        candidate = Path(arg)
        if candidate.is_absolute() or ".." in candidate.parts:
            return arg

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


def save_memory(content: str, basis: str) -> str:
    """Persist a fact for future runs of this agent.

    Entries saved through this tool are always stored as unverified and
    expire automatically - this tool has no argument that produces a
    verified, permanent entry. That tier only exists for facts a human has
    separately confirmed outside this loop. Treat this as a note for a
    future run to sanity-check, not a way to make something permanently
    true by asserting it.

    Args:
        content: The fact to remember.
        basis: What you're grounding this in, so a future reader can judge
            it - e.g. "output of run_bash: wc -l tests/" or "the user
            stated this directly in the conversation".

    Returns:
        A confirmation string including the expiry, or an explanation of
        what was missing if the call was malformed.

    Tool schema notes:
        name: save_memory
        description: Save a fact for future runs of this agent to see.
            Always stored as unverified with an automatic expiry - this
            tool cannot create a permanent or verified entry.
        parameters:
            content (string, required): The fact to remember, e.g. "tests
                run via pytest tests/ -v".
            basis (string, required): What this is grounded in, e.g.
                "output of run_bash: wc -l tests/".
    """
    return memory.write_derived_memory(content, basis)
