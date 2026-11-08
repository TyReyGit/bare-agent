"""A bare ReAct loop over the Groq, Gemini or OpenRouter API — no agent framework.

The loop is deliberately tiny: send the conversation to the model, let it
either call a tool or answer, run the tool, append the result, repeat. The
model never executes anything; it only names a function and supplies JSON
arguments, and this module decides whether to honour that.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable

import openai
from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types
from groq import BadRequestError, Groq
from groq.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionToolParam,
)

from bare_agent import memory, permissions, tools

load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"
GEMINI_MODEL = "gemini-3.6-flash"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"
PROVIDERS = ("groq", "gemini", "openrouter")
DEFAULT_PROVIDER = "groq"
# OpenRouter deliberately has no default model: it fronts hundreds, and a
# silent default would let two runs be confused. The caller must name one.
DEFAULT_MODELS: dict[str, str | None] = {
    "groq": GROQ_MODEL,
    "gemini": GEMINI_MODEL,
    "openrouter": None,
}

class ProviderError(RuntimeError):
    """A provider problem the user can act on (bad config, key, credit, quota).

    Its message is written to be shown as-is, and never contains key values.
    """


class FatalProviderError(ProviderError):
    """A ProviderError that retrying the next task cannot fix (bad key, no credit).

    Eval runners abort the whole run on this instead of recording an ERROR
    for every remaining task.
    """


@dataclass(frozen=True)
class ProviderConfig:
    """Which provider and exact model slug a run is using."""

    provider: str
    model: str


def resolve_provider(
    provider: str | None = None, model: str | None = None
) -> ProviderConfig:
    """Pick provider and model: explicit argument, then env var, then default.

    Environment variables are AGENT_PROVIDER and AGENT_MODEL. The default
    provider is groq (with its default model); gemini has a default model
    too; openrouter has none and requires one to be named. Model slugs are
    passed through exactly as given - never rewritten or substituted.

    Raises:
        ProviderError: Unknown provider, or openrouter with no model.
    """
    name = (provider or os.getenv("AGENT_PROVIDER") or DEFAULT_PROVIDER).lower()
    if name not in PROVIDERS:
        raise ProviderError(
            f"Unknown provider {name!r}. Choose one of: {', '.join(PROVIDERS)}."
        )
    chosen = model or os.getenv("AGENT_MODEL") or DEFAULT_MODELS[name]
    if not chosen:
        raise ProviderError(
            f"Provider {name!r} has no default model. Pass --model or set "
            "AGENT_MODEL to an exact model slug."
        )
    return ProviderConfig(name, chosen)


_config: ProviderConfig | None = None


def configure(provider: str | None = None, model: str | None = None) -> ProviderConfig:
    """Fix the provider and model for subsequent run_agent() calls."""
    global _config
    _config = resolve_provider(provider, model)
    return _config


def active_config() -> ProviderConfig:
    """The configured provider/model, resolving from env/defaults if unset."""
    return _config if _config is not None else resolve_provider()


# Count of provider-specific retries (currently Groq's malformed-tool-call
# retry) since the last reset, so eval results can show them per task.
_retry_count = 0


def reset_retry_count() -> None:
    """Zero the retry counter; call before each task."""
    global _retry_count
    _retry_count = 0


def get_retry_count() -> int:
    """Retries made since the last reset_retry_count()."""
    return _retry_count

MAX_ITERATIONS = 15
MAX_TOOL_CALL_RETRIES = 8
_RETRY_BACKOFF_SECONDS = 1.0

# Loop-detection: how many consecutive identical (same tool, same
# arguments) calls trigger an early stop. This catches the narrow case of
# the model retrying one exact failing call. It does NOT catch the
# broader thrash pattern already seen once in practice - 14 iterations
# reading every file in the workspace with no repeated call - because
# that failure has no identical signature to detect against. Known gap,
# not fixed by this version.
MAX_CONSECUTIVE_REPEATS = 3

# Compaction: how many of the most recent ReAct iterations stay verbatim
# once this many have accumulated. Older iterations fold into one
# deterministic summary - not sent to the model to summarise, which
# would add cost, latency, and non-determinism to something meant to be
# cheap to verify. Default sits above anything the current eval suite
# naturally produces (longest task now runs 1-4 iterations), so this
# cannot fire on any existing task - same "safe against the trusted
# baseline" property MAX_CONSECUTIVE_REPEATS has.
KEEP_RECENT_EXCHANGES = 5
_COMPACTED_HEADER = "[Compacted history - a record of earlier tool activity, not something you said:]"

# Cross-run memory: entries are loaded from disk once at the start of each
# run_agent() call and folded into the system prompt as a labelled block -
# see memory.py for tier rules (VERIFIED vs DERIVED), expiry, and why the
# model has no code path to producing a VERIFIED entry itself.
#
# 15 Aug 2026: the original wording here ("use save_memory to record a
# fact that would help a future run") did not produce spontaneous use
# across two live runs that both involved a genuine multi-file search -
# exactly the situation the tool exists for. This version narrows it to
# an explicit before-final-answer check with a concrete trigger
# condition, in the same spirit as the term-02 capability-absence fix:
# a general "you may use this" statement and a specific "here is when
# you are expected to" instruction are not the same prompt. Whether this
# version actually changes behaviour is a live-test question, not an
# assumption - see the write-up in the repo notes for the result.

SYSTEM_PROMPT = (
    "You are a concise assistant that solves tasks by calling tools. Files "
    "live in a sandboxed workspace/ directory; read them with read_file using "
    "paths relative to that directory. Use calculate for arithmetic rather "
    "than doing it in your head. Call one tool at a time, then use its result "
    "to decide the next step. When you have enough information, reply with "
    "the final answer as plain text and no tool call. Your tools are "
    "read_file, run_bash, calculate, and save_memory - nothing else. If a "
    "task needs something none of them can do (connecting to a network "
    "address, a database, an external API, or any system outside this "
    "sandbox), say so immediately as your final answer. That is a different "
    "situation from not having searched enough yet: no amount of additional "
    "read_file or run_bash calls will change what these tools are capable of."
    "\n\n"
    "Before giving your final answer, check: did you have to search through "
    "multiple files, or try several wrong guesses, to find something a "
    "future run of this same task would need again? If so, call save_memory "
    "now, before your final answer, recording what you found and where. "
    "Skip it if the task was answerable without any searching or guessing. "
    "If earlier memory is shown to you below, entries marked VERIFIED were "
    "confirmed by a person and can be trusted directly. Entries marked "
    "DERIVED were written by a previous run of this same agent and were "
    "never checked by a person - treat them as a hint worth sanity-checking "
    "against the current task, not as a fact to act on unquestioned."
)
# Maps the tool names the model may emit to the real Python functions.
# Anything not in this dict is refused in _execute_tool.
TOOL_FUNCTIONS: dict[str, Callable[..., str]] = {
    "read_file": tools.read_file,
    "run_bash": tools.run_bash,
    "calculate": tools.calculate,
    "save_memory": tools.save_memory,
}


def _schema(
    name: str, description: str, args: dict[str, str]
) -> ChatCompletionToolParam:
    """Build the tool schema the API expects, for a tool whose arguments are
    all required strings.

    Produces, in JSON terms:
        {"type": "function", "function": {"name": ..., "description": ...,
         "parameters": {"type": "object", "properties": {arg: {...}, ...},
                        "required": [arg, ...]}}}

    args maps each argument's name to its description. All four current
    tools (read_file, run_bash, calculate, save_memory) take only required
    string arguments, so that's the only shape this needs to support.
    """
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    arg: {"type": "string", "description": desc}
                    for arg, desc in args.items()
                },
                "required": list(args),
            },
        },
    }


# Transcribed from the "Tool schema notes" section of each docstring in tools.py.
TOOL_SCHEMAS: list[ChatCompletionToolParam] = [
    _schema(
        "read_file",
        "Read the contents of a file from the workspace directory.",
        {
            "path": 'Relative path to the file within workspace/, e.g. '
            '"notes.txt" or "data/input.csv".'
        },
    ),
    _schema(
        "run_bash",
        "Run a shell command and return its stdout. Times out after 10 seconds.",
        {"command": 'The command to execute, e.g. "ls -la" or "python script.py".'},
    ),
    _schema(
        "calculate",
        "Evaluate a basic arithmetic expression (+, -, *, /, //, %, **, "
        "parentheses) and return the numeric result.",
        {"expression": 'The arithmetic expression to evaluate, e.g. "(2 + 3) * 4".'},
    ),
    _schema(
        "save_memory",
        "Save a fact for future runs of this agent to see. Always stored as "
        "unverified with an automatic expiry - this tool cannot create a "
        "permanent or verified entry.",
        {
            "content": 'The fact to remember, e.g. "tests run via pytest tests/ -v".',
            "basis": 'What this is grounded in, e.g. "output of run_bash: '
            'wc -l tests/".',
        },
    ),
]


def _gemini_tool() -> genai_types.Tool:
    """Re-express TOOL_SCHEMAS in Gemini's function-declaration format.

    Both providers need the same information — name, description, and a
    JSON Schema for the arguments — but wrap it differently. Deriving this
    from TOOL_SCHEMAS rather than writing it out twice means the two
    providers cannot drift apart.
    """
    declarations = [
        genai_types.FunctionDeclaration(
            name=schema["function"]["name"],
            description=schema["function"]["description"],
            parameters_json_schema=schema["function"]["parameters"],
        )
        for schema in TOOL_SCHEMAS
    ]
    return genai_types.Tool(function_declarations=declarations)


def _tool_name_for_call_id(
    messages: list[ChatCompletionMessageParam],
    tool_message: ChatCompletionMessageParam,
) -> str:
    """Find the tool name matching a tool result's call id.

    Groq pairs a tool result to its call by id; Gemini pairs it by name.
    The id is the only link the loop's message format preserves, so the
    name has to be recovered from the assistant turn that requested it.
    """
    call_id = tool_message.get("tool_call_id")
    for message in messages:
        if message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            tool_calls = []
        for call in tool_calls:
            if call["id"] == call_id:
                name = call["function"]["name"]
                return str(name)
    return "unknown_tool"


def _to_gemini_contents(
    messages: list[ChatCompletionMessageParam],
) -> tuple[str, list[genai_types.Content]]:
    """Translate the loop's Groq-shaped messages into Gemini's format.

    Returns the system instruction separately, because Gemini takes it as
    config rather than as a message. Gemini also has no 'tool' role: a tool
    result goes back as a function_response part on a 'user' turn, and the
    assistant's own turn is called 'model'.
    """
    system_instruction = ""
    contents: list[genai_types.Content] = []

    for message in messages:
        role = message.get("role")

        if role == "system":
            system_instruction = str(message.get("content") or "")
            continue

        if role == "user":
            text = str(message.get("content") or "")
            contents.append(
                genai_types.Content(
                    role="user", parts=[genai_types.Part.from_text(text=text)]
                )
            )
            continue

        if role == "assistant":
            tool_calls = message.get("tool_calls")
            if not isinstance(tool_calls, list):
                tool_calls = []

            cached = None
            for call in tool_calls:
                cached = _gemini_turns.get(str(call["id"]))
                if cached is not None:
                    break
            if cached is not None:
                contents.append(cached)
                continue

            parts: list[genai_types.Part] = []
            content = message.get("content")
            if content:
                parts.append(genai_types.Part.from_text(text=str(content)))
            for call in tool_calls:
                function = call["function"]
                parts.append(
                    genai_types.Part.from_function_call(
                        name=function["name"],
                        args=json.loads(function["arguments"]),
                    )
                )
            if parts:
                contents.append(genai_types.Content(role="model", parts=parts))
            continue

        if role == "tool":
            contents.append(
                genai_types.Content(
                    role="user",
                    parts=[
                        genai_types.Part.from_function_response(
                            name=_tool_name_for_call_id(messages, message),
                            response={"result": str(message.get("content") or "")},
                        )
                    ],
                )
            )

    return system_instruction, contents


# A model is anything that turns a conversation into one assistant message.
# Injecting this is what lets the tests run without touching the network.
ModelFn = Callable[[list[ChatCompletionMessageParam]], ChatCompletionMessage]

_client: Groq | None = None
_gemini_client: genai.Client | None = None
_openrouter_client: openai.OpenAI | None = None

# Gemini attaches an opaque thought_signature to each functionCall part, and
# rejects a replayed turn that is missing it. That signature cannot survive
# the Groq-shaped message format the loop stores, so the original Content is
# kept here and reused verbatim, keyed by the call ids minted for it.
_gemini_turns: dict[str, genai_types.Content] = {}
_gemini_call_counter = 0
# Gemini's free tier caps requests per minute, not tokens per day, so calls
# must be paced rather than merely bounded. 5 RPM means one call every 12
# seconds; the margin absorbs clock skew and the server's own accounting.
_GEMINI_MIN_INTERVAL_SECONDS = 0.5
_gemini_last_call_at = 0.0


_RATE_LIMIT_BACKOFFS = (2.0, 5.0)


def _is_rate_limited(exc: Exception) -> bool:
    """True for an HTTP 429 from any SDK (groq/openai: status_code, genai: code)."""
    return 429 in (getattr(exc, "status_code", None), getattr(exc, "code", None))


def _send_with_429_retry(send: Callable[[], Any]) -> Any:
    """Call send(), retrying HTTP 429 up to twice (2s then 5s backoff).

    Each retry counts in the per-task retry counter. Anything other than a
    429 - including 401/402 - is raised at once, and a 429 that outlasts the
    retries is re-raised unchanged for the caller to report.
    """
    global _retry_count
    for attempt, wait in enumerate(_RATE_LIMIT_BACKOFFS, start=1):
        try:
            return send()
        except Exception as exc:
            if not _is_rate_limited(exc):
                raise
            _retry_count += 1
            print(
                f"  retry {attempt}/{len(_RATE_LIMIT_BACKOFFS)}: HTTP 429 "
                f"(rate limited), waiting {wait:.0f}s..."
            )
            time.sleep(wait)
    return send()


def _normalise_message(message: Any) -> ChatCompletionMessage:
    """Return a groq-typed assistant message from any OpenAI-shaped one.

    The loop stores Groq-shaped messages. Groq's own reply already is one and
    passes through untouched; an openai-SDK reply is rebuilt field by field.
    """
    if isinstance(message, ChatCompletionMessage):
        return message
    calls = [
        ChatCompletionMessageToolCall(
            id=call.id,
            type="function",
            function={  # type: ignore[arg-type]
                "name": call.function.name,
                "arguments": call.function.arguments,
            },
        )
        for call in (message.tool_calls or [])
        if call.type == "function"
    ]
    return ChatCompletionMessage(
        role="assistant", content=message.content, tool_calls=calls or None
    )


def _openai_compat_completion(
    create: Callable[..., Any],
    model: str,
    messages: list[ChatCompletionMessageParam],
    *,
    retry_tool_use_failed: bool,
) -> ChatCompletionMessage:
    """Send the conversation to an OpenAI-compatible endpoint (Groq, OpenRouter).

    Args:
        create: The client's chat.completions.create.
        model: Exact model slug to send.
        messages: The conversation.
        retry_tool_use_failed: Groq only. Retry a bounded number of times if
            Groq rejects the request with 'tool_use_failed' - the model
            emitted a malformed tool call (a pseudo-XML wrapper instead of
            structured JSON), a sampling issue and not a problem with our
            request. Any other BadRequestError is real and raised at once.
            Off for OpenRouter, whose errors are not retried here.
    """
    global _retry_count
    for attempt in range(1, MAX_TOOL_CALL_RETRIES + 1):
        try:
            response = _send_with_429_retry(
                lambda: create(model=model, messages=messages, tools=TOOL_SCHEMAS)
            )
            return _normalise_message(response.choices[0].message)
        except BadRequestError as exc:
            if not retry_tool_use_failed or "tool_use_failed" not in str(exc):
                raise
            if attempt == MAX_TOOL_CALL_RETRIES:
                raise
            _retry_count += 1
            wait = _RETRY_BACKOFF_SECONDS * attempt
            print(
                f"  retry {attempt}/{MAX_TOOL_CALL_RETRIES}: malformed tool call "
                f"from model, retrying in {wait}s..."
            )
            time.sleep(wait)

    raise RuntimeError("Unreachable: retry loop exited without returning or raising.")


def _groq_completion(messages: list[ChatCompletionMessageParam]) -> ChatCompletionMessage:
    """Send the conversation to Groq and return the assistant's reply."""
    global _client
    if _client is None:
        _client = Groq()  # reads GROQ_API_KEY from the environment
    return _openai_compat_completion(
        _client.chat.completions.create,
        active_config().model,
        messages,
        retry_tool_use_failed=True,
    )


def _openrouter_error(exc: openai.APIStatusError, model: str) -> ProviderError | None:
    """Map an OpenRouter 401/402/429 to an actionable message, else None.

    401 and 402 are fatal (the next call will fail identically); 429 is not.
    """
    code = exc.status_code
    if code == 401:
        return FatalProviderError(
            f"OpenRouter returned 401 (unauthorized). Check that {OPENROUTER_KEY_ENV} "
            "is set, correct and not revoked."
        )
    if code == 402:
        return FatalProviderError(
            f"OpenRouter returned 402 (payment required) for {model!r}. Check your "
            "OpenRouter credit balance - it is exhausted or too low for this model."
        )
    if code == 429:
        return ProviderError(
            f"OpenRouter returned 429 (rate limited) for {model!r}. Check your "
            "rate limit (or the model's upstream limit), wait, and retry."
        )
    return None


def _openrouter_completion(
    messages: list[ChatCompletionMessageParam],
) -> ChatCompletionMessage:
    """Send the conversation to OpenRouter via the openai SDK.

    401/402/429 become a ProviderError saying what to check. The SDK's own
    silent retries are disabled (max_retries=0) so a 429 surfaces as such
    instead of being hidden inside the client.
    """
    global _openrouter_client
    if _openrouter_client is None:
        key = os.getenv(OPENROUTER_KEY_ENV)
        if not key:
            raise FatalProviderError(
                f"{OPENROUTER_KEY_ENV} is not set. Add it to .env."
            )
        _openrouter_client = openai.OpenAI(
            base_url=OPENROUTER_BASE_URL, api_key=key, max_retries=0
        )
    model = active_config().model
    try:
        return _openai_compat_completion(
            _openrouter_client.chat.completions.create,
            model,
            messages,
            retry_tool_use_failed=False,
        )
    except openai.APIStatusError as exc:
        friendly = _openrouter_error(exc, model)
        if friendly is None:
            raise
        raise friendly from None


def _gemini_completion(
    messages: list[ChatCompletionMessageParam],
) -> ChatCompletionMessage:
    """Send the conversation to Gemini and return the reply, Groq-shaped.

    The loop's message format stays Groq-shaped throughout; this function
    translates in and out at the boundary so run_agent never learns which
    provider it is talking to. Automatic function calling is disabled: the
    SDK would otherwise execute the tools itself, bypassing the sandbox in
    tools.py entirely.
    """
    global _gemini_client, _gemini_call_counter, _gemini_last_call_at
    if _gemini_client is None:
        _gemini_client = genai.Client()  # reads GOOGLE_API_KEY from the environment

    system_instruction, contents = _to_gemini_contents(messages)

    elapsed = time.monotonic() - _gemini_last_call_at
    if _gemini_last_call_at and elapsed < _GEMINI_MIN_INTERVAL_SECONDS:
        wait = _GEMINI_MIN_INTERVAL_SECONDS - elapsed
        print(f"  pacing: waiting {wait:.1f}s for Gemini's 5 RPM limit...")
        time.sleep(wait)
    _gemini_last_call_at = time.monotonic()

    client = _gemini_client
    response = _send_with_429_retry(
        lambda: client.models.generate_content(
            model=active_config().model,
            contents=contents,
            config=genai_types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=[_gemini_tool()],
                automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
            ),
        )
    )

    tool_calls = []
    for call in response.function_calls or []:
        _gemini_call_counter += 1
        tool_calls.append(
            ChatCompletionMessageToolCall(
                id=f"gemini_call_{_gemini_call_counter}",
                type="function",
                function={  # type: ignore[arg-type]
                    "name": call.name or "",
                    "arguments": json.dumps(dict(call.args or {})),
                },
            )
        )

    if tool_calls and response.candidates:
        original = response.candidates[0].content
        if original is not None:
            for made_call in tool_calls:
                _gemini_turns[str(made_call.id)] = original

    return ChatCompletionMessage(
        role="assistant",
        content=None if tool_calls else (response.text or ""),
        tool_calls=tool_calls or None,
    )


def _execute_tool(
    tool_call: ChatCompletionMessageToolCall, mode: permissions.Mode
) -> str:
    """Run one requested tool, returning the model-visible result string.

    Every failure becomes a returned string rather than an exception: the
    model reads this text and gets a chance to correct itself next turn.

    Every call passes through permissions.check_permission() first,
    including calls that will simply come back AUTO_ALLOW - read_file and
    calculate are not special-cased around the gate, because a gate only
    consulted for calls that already look risky has a blind spot exactly
    where the risk analysis turns out to be wrong. A NEVER-tier call never
    reaches the underlying tool function at all: that is what makes NEVER a
    code-level constraint rather than a prompt preference (see
    permissions.py's own module docstring for the same argument).

    An approved ASK_FIRST run_bash call is dispatched to
    tools.run_bash_approved, not tools.run_bash - the model-facing run_bash
    refuses mutating commands unconditionally regardless of what any gate
    decides (see tools.py), so approval has to route to the separate
    privileged executor to have any effect at all.
    """
    name = tool_call.function.name
    function = TOOL_FUNCTIONS.get(name)
    if function is None:
        return f"Unknown tool '{name}'. Available tools: {', '.join(TOOL_FUNCTIONS)}."

    try:
        arguments = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError as exc:
        return f"Arguments for '{name}' were not valid JSON: {exc}."
    if not isinstance(arguments, dict):
        return f"Arguments for '{name}' must be a JSON object, got {type(arguments).__name__}."

    decision = permissions.check_permission(name, arguments, mode)
    print(f"  permission {decision.tier.value}: {decision.reason}")

    if not decision.allowed:
        return decision.reason

    if name == "run_bash" and decision.tier is permissions.Tier.ASK_FIRST:
        try:
            return tools.run_bash_approved(**arguments)
        except TypeError as exc:
            return f"Invalid arguments for '{name}': {exc}."

    try:
        return function(**arguments)
    except TypeError as exc:
        return f"Invalid arguments for '{name}': {exc}."


def _as_assistant_param(
    message: ChatCompletionMessage, tool_calls: list[ChatCompletionMessageToolCall]
) -> ChatCompletionAssistantMessageParam:
    """Convert the model's response back into a message we can send again."""
    return {
        "role": "assistant",
        "content": message.content,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in tool_calls
        ],
    }


def _truncate(text: str, limit: int = 300) -> str:
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else collapsed[:limit] + "..."


def _truncate_middle(text: str, limit: int = 120) -> str:
    """Shorten text to fit within limit, keeping both the start and the end.

    _truncate() above keeps only the front, which is right for the live
    console display - the first thing a call did usually matters most
    there. Compacted summaries are different: tool results routinely end
    with the one figure that matters (the "39 total" at the end of a
    wc -l line, for example - a real bug found via a live run on 15 Aug
    2026, where the old front-only truncation silently dropped it and the
    model only survived by re-deriving the total itself). Splits the
    budget evenly between both ends with a short marker showing the cut.
    """
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    marker = " ... "
    keep = limit - len(marker)
    head = keep // 2
    tail = keep - head
    return collapsed[:head] + marker + collapsed[-tail:]


def _call_signature(call: ChatCompletionMessageToolCall) -> tuple[str, str]:
    """A (name, canonical-arguments) pair identifying what a call actually did.

    Arguments are re-serialised with sorted keys so two calls that are
    identical in substance but arrived with keys in a different order
    still compare equal - the model's JSON key order isn't guaranteed
    stable. Malformed JSON falls back to the raw string, which still
    compares correctly for exact repeats.
    """
    try:
        parsed = json.loads(call.function.arguments)
        args = json.dumps(parsed, sort_keys=True)
    except json.JSONDecodeError:
        args = call.function.arguments
    return (call.function.name, args)


def _iteration_start_indices(messages: list[ChatCompletionMessageParam]) -> list[int]:
    """Indices where each ReAct iteration's turn begins.

    Every iteration begins with exactly one assistant message requesting
    tool calls, immediately followed by that many tool-role results.
    Scanning for assistant-role positions after the initial system+user
    pair recovers the boundaries without tracking them separately as
    run_agent executes.
    """
    return [i for i, m in enumerate(messages) if i >= 2 and m.get("role") == "assistant"]


def _compact_if_needed(
    messages: list[ChatCompletionMessageParam], keep_recent_exchanges: int
) -> list[ChatCompletionMessageParam]:
    """Fold everything older than the most recent N iterations into one summary.

    Deterministic truncation, not model summarisation - see
    KEEP_RECENT_EXCHANGES. Uses role="user", never role="system": a
    system-role message mid-conversation would silently overwrite the
    real system instruction on Gemini (see _to_gemini_contents) rather
    than becoming a turn. Re-compaction merges forward rather than
    re-summarising an existing summary, so nothing already folded is lost
    across repeated calls.
    """
    starts = _iteration_start_indices(messages)
    if len(starts) <= keep_recent_exchanges:
        return messages

    print(f"  compacting: {len(starts) - keep_recent_exchanges} iteration(s) folded into a summary")

    cutoff = starts[-keep_recent_exchanges]
    old_messages = messages[2:cutoff]
    recent_messages = messages[cutoff:]

    lines: list[str] = []
    for message in old_messages:
        role = message.get("role")
        content = message.get("content")
        if role == "user" and isinstance(content, str) and content.startswith(_COMPACTED_HEADER):
            lines.extend(content.split("\n")[1:])
        elif role == "assistant":
            calls = message.get("tool_calls")
            if isinstance(calls, list):
                for call in calls:
                    function = call["function"]
                    lines.append(
                        f"- called {function['name']}({_truncate_middle(function['arguments'], 80)})"
                    )
        elif role == "tool":
            lines.append(f"  -> {_truncate_middle(str(content or ''), 120)}")
    compacted: ChatCompletionMessageParam = {
        "role": "user",
        "content": _COMPACTED_HEADER + "\n" + "\n".join(lines),
    }
    return [*messages[:2], compacted, *recent_messages]


def run_agent(
    task: str,
    *,
    model: ModelFn | None = None,
    max_iterations: int = MAX_ITERATIONS,
    max_consecutive_repeats: int = MAX_CONSECUTIVE_REPEATS,
    keep_recent_exchanges: int = KEEP_RECENT_EXCHANGES,
    mode: permissions.Mode = permissions.Mode.INTERACTIVE,
) -> str:
    """Run the ReAct loop until the model answers or the budget runs out.

    Args:
        task: The user's request.
        model: Function that turns a conversation into an assistant message.
            Defaults to the provider selected by configure() or
            AGENT_PROVIDER/AGENT_MODEL (groq, gemini or openrouter); tests
            pass a fake.
        max_iterations: Hard cap on model round-trips.
        max_consecutive_repeats: Stop early if the same tool is called with
            identical arguments this many times in a row. Catches the
            narrow case of the model retrying one exact failing call; does
            not catch broader thrash across different calls with no
            repeated signature.
        keep_recent_exchanges: Once more than this many ReAct iterations
            have accumulated, fold everything older into one deterministic
            summary appended as a user-role message. Never role="system" -
            see _compact_if_needed's docstring for why. Default sits above
            what any current eval task produces, so it does not fire on
            the existing suite.
        mode: Which permissions.Mode governs ASK_FIRST tool calls.
            INTERACTIVE (the default) prompts at the terminal and waits for
            a yes/no - correct when a human is actually present, as in the
            __main__ block below. AUTO_DENY denies every ASK_FIRST call
            without prompting - correct for unattended eval runs, which
            have no human available to ask, and which must not silently
            grant themselves consequential actions just because nobody is
            watching.

    Returns:
        The model's final answer, or a message saying the cap was reached.

    Before the first iteration, non-expired entries from memory.jsonl are
    loaded and folded into the system message as a labelled block (see
    memory.py). This happens once per run, not per iteration - memory
    written by save_memory during this run is visible to future runs, not
    retroactively to this one.
    """
    if model is not None:
        complete = model
    else:
        complete = {
            "gemini": _gemini_completion,
            "openrouter": _openrouter_completion,
        }.get(active_config().provider, _groq_completion)

    system_content = SYSTEM_PROMPT
    memory_block = memory.format_memory_for_prompt(memory.load_memory())
    if memory_block:
        system_content = f"{SYSTEM_PROMPT}\n\n{memory_block}"

    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": task},
    ]

    recent_calls: list[tuple[str, str]] = []

    for iteration in range(1, max_iterations + 1):
        print(f"\n=== iteration {iteration}/{max_iterations} ===")
        message = complete(messages)
        tool_calls = message.tool_calls

        if not tool_calls:
            answer = message.content or ""
            print(f"final answer: {answer}")
            return answer

        messages.append(_as_assistant_param(message, tool_calls))
        for call in tool_calls:
            print(f"  call   {call.function.name}({_truncate(call.function.arguments)})")
            result = _execute_tool(call, mode)
            print(f"  result {_truncate(result)}")
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": result}
            )
            signature = _call_signature(call)
            recent_calls.append(signature)
            if (
                len(recent_calls) >= max_consecutive_repeats
                and len(set(recent_calls[-max_consecutive_repeats:])) == 1
            ):
                print(
                    f"  loop detected: {call.function.name} called "
                    f"{max_consecutive_repeats}x in a row with identical "
                    "arguments"
                )
                return (
                    f"Stopped after detecting a loop: {call.function.name} "
                    f"was called {max_consecutive_repeats} times in a row "
                    "with identical arguments, with no progress toward an "
                    "answer."
                )
        messages = _compact_if_needed(messages, keep_recent_exchanges)

    return (
        f"Max iterations reached ({max_iterations}) without a final answer. "
        "The task may be too complex, or the model may be looping."
    )


if __name__ == "__main__":
    import argparse

    _parser = argparse.ArgumentParser(description="Run the demo task.")
    _parser.add_argument("--provider", choices=PROVIDERS, help="default: groq")
    _parser.add_argument("--model", help="exact model slug")
    _args = _parser.parse_args()
    try:
        configure(_args.provider, _args.model)
    except ProviderError as _exc:
        raise SystemExit(str(_exc)) from None

    tools.WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    (tools.WORKSPACE_DIR / "inventory.txt").write_text(
        "Warehouse inventory report\nwidgets in stock: 47\nunit price: 12.50\n",
        encoding="utf-8",
    )

    try:
        final = run_agent(
            "Read inventory.txt from the workspace. It lists how many widgets are "
            "in stock and their unit price. Work out the total value of the widget "
            "stock, and tell me the number."
        )
    except ProviderError as _exc:
        raise SystemExit(str(_exc)) from None
    print(f"\n{'=' * 40}\n{final}")
