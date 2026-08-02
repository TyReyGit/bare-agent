"""A bare ReAct loop over the Groq or Gemini API — no agent framework.

The loop is deliberately tiny: send the conversation to the model, let it
either call a tool or answer, run the tool, append the result, repeat. The
model never executes anything; it only names a function and supplies JSON
arguments, and this module decides whether to honour that.
"""

from __future__ import annotations

import json
import os
import time
from typing import Callable

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

from bare_agent import tools

load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"
GEMINI_MODEL = "gemini-flash-latest"
PROVIDER = os.getenv("AGENT_PROVIDER", "groq").lower()
MAX_ITERATIONS = 10
MAX_TOOL_CALL_RETRIES = 8
_RETRY_BACKOFF_SECONDS = 1.0

SYSTEM_PROMPT = (
    "You are a concise assistant that solves tasks by calling tools. Files "
    "live in a sandboxed workspace/ directory; read them with read_file using "
    "paths relative to that directory. Use calculate for arithmetic rather "
    "than doing it in your head. Call one tool at a time, then use its result "
    "to decide the next step. When you have enough information, reply with "
    "the final answer as plain text and no tool call."
)

# Maps the tool names the model may emit to the real Python functions.
# Anything not in this dict is refused in _execute_tool.
TOOL_FUNCTIONS: dict[str, Callable[..., str]] = {
    "read_file": tools.read_file,
    "run_bash": tools.run_bash,
    "calculate": tools.calculate,
}


def _schema(
    name: str, description: str, arg: str, arg_description: str
) -> ChatCompletionToolParam:
    """Build the tool schema the API expects, for a single-string-argument tool.

    Produces, in JSON terms:
        {"type": "function", "function": {"name": ..., "description": ...,
         "parameters": {"type": "object", "properties": {arg: {...}},
                        "required": [arg]}}}
    """
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {arg: {"type": "string", "description": arg_description}},
                "required": [arg],
            },
        },
    }


# Transcribed from the "Tool schema notes" section of each docstring in tools.py.
TOOL_SCHEMAS: list[ChatCompletionToolParam] = [
    _schema(
        "read_file",
        "Read the contents of a file from the workspace directory.",
        "path",
        'Relative path to the file within workspace/, e.g. "notes.txt" or "data/input.csv".',
    ),
    _schema(
        "run_bash",
        "Run a shell command and return its stdout. Times out after 10 seconds.",
        "command",
        'The command to execute, e.g. "ls -la" or "python script.py".',
    ),
    _schema(
        "calculate",
        "Evaluate a basic arithmetic expression (+, -, *, /, //, %, **, "
        "parentheses) and return the numeric result.",
        "expression",
        'The arithmetic expression to evaluate, e.g. "(2 + 3) * 4".',
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

# Gemini attaches an opaque thought_signature to each functionCall part, and
# rejects a replayed turn that is missing it. That signature cannot survive
# the Groq-shaped message format the loop stores, so the original Content is
# kept here and reused verbatim, keyed by the call ids minted for it.
_gemini_turns: dict[str, genai_types.Content] = {}
_gemini_call_counter = 0


def _groq_completion(messages: list[ChatCompletionMessageParam]) -> ChatCompletionMessage:
    """Send the conversation to Groq and return the assistant's reply.

    Retries a bounded number of times if Groq rejects the request with a
    'tool_use_failed' error. This happens when the model emits a malformed
    tool call (a pseudo-XML wrapper instead of structured JSON) rather than
    a valid one — a sampling issue, not a problem with our request. Any
    other BadRequestError is a real problem and is raised immediately.
    """
    global _client
    if _client is None:
        _client = Groq()  # reads GROQ_API_KEY from the environment

    for attempt in range(1, MAX_TOOL_CALL_RETRIES + 1):
        try:
            response = _client.chat.completions.create(
                model=GROQ_MODEL, messages=messages, tools=TOOL_SCHEMAS
            )
            return response.choices[0].message
        except BadRequestError as exc:
            if "tool_use_failed" not in str(exc):
                raise
            if attempt == MAX_TOOL_CALL_RETRIES:
                raise
            wait = _RETRY_BACKOFF_SECONDS * attempt
            print(
                f"  retry {attempt}/{MAX_TOOL_CALL_RETRIES}: malformed tool call "
                f"from model, retrying in {wait}s..."
            )
            time.sleep(wait)

    raise RuntimeError("Unreachable: retry loop exited without returning or raising.")


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
    global _gemini_client, _gemini_call_counter
    if _gemini_client is None:
        _gemini_client = genai.Client()  # reads GOOGLE_API_KEY from the environment

    system_instruction, contents = _to_gemini_contents(messages)
    response = _gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=genai_types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=[_gemini_tool()],
            automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        ),
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


def _execute_tool(tool_call: ChatCompletionMessageToolCall) -> str:
    """Run one requested tool, returning the model-visible result string.

    Every failure becomes a returned string rather than an exception: the
    model reads this text and gets a chance to correct itself next turn.
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


def run_agent(
    task: str,
    *,
    model: ModelFn | None = None,
    max_iterations: int = MAX_ITERATIONS,
) -> str:
    """Run the ReAct loop until the model answers or the budget runs out.

    Args:
        task: The user's request.
        model: Function that turns a conversation into an assistant message.
            Defaults to the provider named by AGENT_PROVIDER (groq or
            gemini); tests pass a fake.
        max_iterations: Hard cap on model round-trips.

    Returns:
        The model's final answer, or a message saying the cap was reached.
    """
    if model is not None:
        complete = model
    elif PROVIDER == "gemini":
        complete = _gemini_completion
    else:
        complete = _groq_completion
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]

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
            result = _execute_tool(call)
            print(f"  result {_truncate(result)}")
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": result}
            )

    return (
        f"Max iterations reached ({max_iterations}) without a final answer. "
        "The task may be too complex, or the model may be looping."
    )


if __name__ == "__main__":
    tools.WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    (tools.WORKSPACE_DIR / "inventory.txt").write_text(
        "Warehouse inventory report\nwidgets in stock: 47\nunit price: 12.50\n",
        encoding="utf-8",
    )

    final = run_agent(
        "Read inventory.txt from the workspace. It lists how many widgets are "
        "in stock and their unit price. Work out the total value of the widget "
        "stock, and tell me the number."
    )
    print(f"\n{'=' * 40}\n{final}")
