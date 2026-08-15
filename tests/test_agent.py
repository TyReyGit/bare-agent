from groq.types.chat import ChatCompletionMessage, ChatCompletionMessageParam
from groq.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

from bare_agent import agent


def make_tool_call(name: str, arguments: str, call_id: str = "call_1") -> ChatCompletionMessageToolCall:
    return ChatCompletionMessageToolCall(
        id=call_id, type="function", function=Function(name=name, arguments=arguments)
    )


def final_answer(content: str) -> ChatCompletionMessage:
    return ChatCompletionMessage(role="assistant", content=content)


def tool_request(name: str, arguments: str) -> ChatCompletionMessage:
    return ChatCompletionMessage(
        role="assistant", content=None, tool_calls=[make_tool_call(name, arguments)]
    )


class FakeModel:
    """Stands in for a real Groq call: replays scripted assistant messages.

    Once the script runs out it repeats the last message forever, which is
    what makes the runaway-loop test possible.
    """

    def __init__(self, *responses: ChatCompletionMessage) -> None:
        self.responses = list(responses)
        self.seen: list[list[ChatCompletionMessageParam]] = []

    def __call__(
        self, messages: list[ChatCompletionMessageParam]
    ) -> ChatCompletionMessage:
        self.seen.append(list(messages))
        index = min(len(self.seen) - 1, len(self.responses) - 1)
        return self.responses[index]

    @property
    def call_count(self) -> int:
        return len(self.seen)


def test_final_answer_without_tool_call_terminates() -> None:
    model = FakeModel(final_answer("The answer is 42."))

    result = agent.run_agent("What is the answer?", model=model)

    assert result == "The answer is 42."
    assert model.call_count == 1


def test_max_iterations_terminates_the_loop() -> None:
    model = FakeModel(tool_request("calculate", '{"expression": "1 + 1"}'))

    # max_consecutive_repeats is set above max_iterations so loop
    # detection (added for the identical-repeated-call case) can't
    # trigger first and mask what this test is actually checking - the
    # iteration cap itself.
    result = agent.run_agent(
        "Loop forever", model=model, max_iterations=3, max_consecutive_repeats=4
    )
    assert "Max iterations reached (3)" in result
    assert model.call_count == 3


def test_identical_repeated_call_triggers_loop_detection() -> None:
    model = FakeModel(tool_request("calculate", '{"expression": "1 + 1"}'))

    result = agent.run_agent(
        "Loop forever", model=model, max_iterations=10, max_consecutive_repeats=3
    )

    assert "Stopped after detecting a loop" in result
    assert "calculate" in result
    # Detection fires on the 3rd identical call, not after max_iterations
    # is exhausted - that's the whole point of stopping early.
    assert model.call_count == 3


def test_varying_calls_do_not_trigger_loop_detection() -> None:
    """Three different calls in a row must NOT look like a repeated loop.

    Guards against a false positive on ordinary multi-step tasks, where
    calling the same tool several times with different arguments (reading
    three different files, say) is normal progress, not thrash.
    """
    model = FakeModel(
        tool_request("calculate", '{"expression": "1 + 1"}'),
        tool_request("calculate", '{"expression": "2 + 2"}'),
        tool_request("calculate", '{"expression": "3 + 3"}'),
        final_answer("done"),
    )

    result = agent.run_agent(
        "Do three different sums", model=model, max_consecutive_repeats=3
    )

    assert result == "done"
    assert "Stopped after detecting a loop" not in result


def test_compaction_folds_old_iterations_and_preserves_a_record() -> None:
    """FakeModel replays a fixed script regardless of what it's shown, so
    this can't prove the model still reasons correctly over compacted
    context - that needs a live model. What IS checkable: compaction
    fires at the right point, keeps the conversation bounded instead of
    growing forever, preserves a record of what was folded even when a
    second compaction re-folds a summary that already exists, and the
    loop still completes normally.
    """
    model = FakeModel(
        tool_request("calculate", '{"expression": "1 + 1"}'),
        tool_request("calculate", '{"expression": "2 + 2"}'),
        tool_request("calculate", '{"expression": "3 + 3"}'),
        tool_request("calculate", '{"expression": "4 + 4"}'),
        final_answer("done"),
    )

    result = agent.run_agent("Do four sums", model=model, keep_recent_exchanges=2)

    assert result == "done"

    final_call_messages = model.seen[-1]
    # Bounded: system + user + one compacted summary + 2 verbatim
    # iterations (assistant+tool each) = 7, not the 10 it would be if
    # every iteration were kept.
    assert len(final_call_messages) == 7

    compacted = final_call_messages[2]
    assert compacted["role"] == "user"
    assert str(compacted["content"]).startswith(agent._COMPACTED_HEADER)
    # The very first call survives into the SECOND compaction's
    # merge-forward - proving folding a summary re-uses its contents
    # rather than silently discarding them.
    assert "1 + 1" in str(compacted["content"])

    # The two most recent iterations stay verbatim, not summarised.
    assert final_call_messages[3]["role"] == "assistant"
    assert final_call_messages[3].get("tool_calls")
    assert final_call_messages[5]["role"] == "assistant"
    assert final_call_messages[5].get("tool_calls")


def test_short_task_does_not_trigger_compaction() -> None:
    """A task within the default keep_recent_exchanges=5 window - like
    every task in the current 14-task eval suite - must never compact.
    """
    model = FakeModel(
        tool_request("calculate", '{"expression": "1 + 1"}'),
        tool_request("calculate", '{"expression": "2 + 2"}'),
        tool_request("calculate", '{"expression": "3 + 3"}'),
        final_answer("done"),
    )

    result = agent.run_agent("Do three sums", model=model)

    assert result == "done"
    final_call_messages = model.seen[-1]
    # No folding: system + user + 3 full iterations (assistant+tool) = 8.
    assert len(final_call_messages) == 8
    assert not any(
        agent._COMPACTED_HEADER in str(m.get("content", "")) for m in final_call_messages
    )

def test_tool_result_is_fed_back_into_the_conversation() -> None:
    model = FakeModel(
        tool_request("calculate", '{"expression": "6 * 7"}'),
        final_answer("42"),
    )

    result = agent.run_agent("Multiply 6 by 7", model=model)

    assert result == "42"
    second_conversation = model.seen[1]
    assert second_conversation[-1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "42",
    }
    assert second_conversation[-2]["role"] == "assistant"


def test_unknown_tool_is_reported_back_to_the_model() -> None:
    model = FakeModel(tool_request("delete_everything", "{}"), final_answer("ok"))

    agent.run_agent("Do something odd", model=model)

    tool_message = model.seen[1][-1]
    assert "Unknown tool 'delete_everything'" in str(tool_message["content"])


def test_malformed_json_arguments_are_reported_back_to_the_model() -> None:
    model = FakeModel(tool_request("calculate", "{not json"), final_answer("ok"))

    agent.run_agent("Break the arguments", model=model)

    tool_message = model.seen[1][-1]
    assert "not valid JSON" in str(tool_message["content"])


def test_wrong_argument_name_is_reported_back_to_the_model() -> None:
    model = FakeModel(tool_request("calculate", '{"formula": "1 + 1"}'), final_answer("ok"))

    agent.run_agent("Use the wrong keyword", model=model)

    tool_message = model.seen[1][-1]
    assert "Invalid arguments for 'calculate'" in str(tool_message["content"])


def test_execute_tool_runs_the_real_function() -> None:
    result = agent._execute_tool(make_tool_call("calculate", '{"expression": "(2+3)*4"}'))

    assert result == "20"
