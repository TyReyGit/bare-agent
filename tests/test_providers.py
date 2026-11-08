"""Provider/model selection, OpenRouter error mapping, and eval-runner flags.

Every network call is mocked; nothing here touches a real API.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import groq
import httpx
import httpx2
import openai
from groq.types.chat import ChatCompletionMessage
import pytest

from bare_agent import agent

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
import run_eval  # noqa: E402
import run_passk  # noqa: E402

FAKE_KEY = "sk-or-test-SECRET-123"


@pytest.fixture(autouse=True)
def clean_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("AGENT_PROVIDER", raising=False)
    monkeypatch.delenv("AGENT_MODEL", raising=False)
    monkeypatch.setattr(agent, "_config", None)
    monkeypatch.setattr(agent, "_openrouter_client", None)
    agent.reset_retry_count()
    yield


# --- selection --------------------------------------------------------------


def test_default_is_groq_with_its_default_model() -> None:
    cfg = agent.resolve_provider()
    assert (cfg.provider, cfg.model) == ("groq", agent.GROQ_MODEL)


def test_gemini_default_model_unchanged() -> None:
    assert agent.resolve_provider("gemini").model == "gemini-3.6-flash"


def test_env_vars_select_provider_and_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_PROVIDER", "OpenRouter")
    monkeypatch.setenv("AGENT_MODEL", "anthropic/claude-haiku-5.5")
    cfg = agent.resolve_provider()
    assert (cfg.provider, cfg.model) == ("openrouter", "anthropic/claude-haiku-5.5")


def test_explicit_args_beat_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_PROVIDER", "groq")
    monkeypatch.setenv("AGENT_MODEL", "x")
    cfg = agent.resolve_provider("gemini", "gemini-3.8-flash")
    assert (cfg.provider, cfg.model) == ("gemini", "gemini-3.8-flash")


@pytest.mark.parametrize(
    ("provider", "slug"),
    [
        ("openrouter", "openai/gpt-6-luna"),
        ("openrouter", "anthropic/claude-haiku-5.5"),
        ("groq", "openai/gpt-oss-120b"),
        ("gemini", "gemini-3.8-flash"),
    ],
)
def test_slugs_pass_through_unchanged(provider: str, slug: str) -> None:
    assert agent.resolve_provider(provider, slug).model == slug


def test_openrouter_requires_a_model() -> None:
    with pytest.raises(agent.ProviderError, match="no default model"):
        agent.resolve_provider("openrouter")


def test_unknown_provider_rejected() -> None:
    with pytest.raises(agent.ProviderError, match="Unknown provider"):
        agent.resolve_provider("bogus")


def test_run_agent_dispatches_by_configured_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []

    def fake(messages: Any) -> Any:
        seen.append(agent.active_config().model)
        return ChatCompletionMessage(role="assistant", content="done")

    monkeypatch.setattr(agent, "_openrouter_completion", fake)
    agent.configure("openrouter", "openai/gpt-6-luna")
    assert agent.run_agent("hi") == "done"
    assert seen == ["openai/gpt-6-luna"]


# --- OpenRouter ---------------------------------------------------------------


def _status_error(code: int) -> openai.APIStatusError:
    # openai 3.x types its errors against httpx2, groq's against httpx.
    request = httpx2.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx2.Response(code, request=request, json={"error": {"message": "x"}})
    return openai.APIStatusError("boom", response=response, body=None)


class _FakeClient:
    def __init__(self, behaviour: Any) -> None:
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=behaviour)
        )


def _use_fake_client(monkeypatch: pytest.MonkeyPatch, behaviour: Any) -> None:
    monkeypatch.setattr(agent, "_openrouter_client", _FakeClient(behaviour))
    agent.configure("openrouter", "openai/gpt-6-luna")


@pytest.mark.parametrize(
    ("code", "hint"),
    [(401, "OPENROUTER_API_KEY"), (402, "credit"), (429, "rate limit")],
)
def test_openrouter_errors_become_actionable(
    monkeypatch: pytest.MonkeyPatch, code: int, hint: str
) -> None:
    def create(**_: Any) -> Any:
        raise _status_error(code)

    _use_fake_client(monkeypatch, create)
    with pytest.raises(agent.ProviderError) as info:
        agent._openrouter_completion([])
    assert hint in str(info.value)
    assert str(code) in str(info.value)
    assert FAKE_KEY not in str(info.value)


@pytest.mark.parametrize(("code", "fatal"), [(401, True), (402, True), (429, False)])
def test_only_401_and_402_are_fatal(
    monkeypatch: pytest.MonkeyPatch, code: int, fatal: bool
) -> None:
    def create(**_: Any) -> Any:
        raise _status_error(code)

    _use_fake_client(monkeypatch, create)
    with pytest.raises(agent.ProviderError) as info:
        agent._openrouter_completion([])
    assert isinstance(info.value, agent.FatalProviderError) is fatal


def test_run_task_reraises_fatal_error_and_keeps_trace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(run_eval, "TRACES_DIR", tmp_path)
    agent.configure("openrouter", "openai/gpt-6-luna")

    def boom(task: str, *, mode: Any) -> str:
        raise agent.FatalProviderError("OpenRouter returned 402")

    monkeypatch.setattr(run_eval, "run_agent", boom)
    with pytest.raises(agent.FatalProviderError):
        run_eval.run_task({"id": "t", "category": "c", "task": "q", "expect": "1"})
    assert "402" in (tmp_path / "t.log").read_text(encoding="utf-8")


def test_run_task_still_records_nonfatal_provider_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(run_eval, "TRACES_DIR", tmp_path)
    agent.configure("openrouter", "openai/gpt-6-luna")

    def limited(task: str, *, mode: Any) -> str:
        raise agent.ProviderError("OpenRouter returned 429")

    monkeypatch.setattr(run_eval, "run_agent", limited)
    result = run_eval.run_task({"id": "t", "category": "c", "task": "q", "expect": "1"})
    assert result["status"] == "ERROR"


def test_abort_run_exits_nonzero_without_writing_results(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as info:
        run_eval.abort_run(agent.FatalProviderError("OpenRouter returned 401"))
    assert info.value.code == 1
    assert "Aborting run" in capsys.readouterr().err


def test_other_openrouter_status_is_not_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def create(**_: Any) -> Any:
        raise _status_error(500)

    _use_fake_client(monkeypatch, create)
    with pytest.raises(openai.APIStatusError):
        agent._openrouter_completion([])


def test_missing_openrouter_key_is_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    agent.configure("openrouter", "openai/gpt-6-luna")
    with pytest.raises(agent.ProviderError, match="OPENROUTER_API_KEY is not set"):
        agent._openrouter_completion([])


def test_client_built_with_openrouter_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_KEY)
    captured: dict[str, Any] = {}

    class FakeOpenAI(_FakeClient):
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)
            super().__init__(lambda **_: _reply("ok"))

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    agent.configure("openrouter", "openai/gpt-6-luna")
    agent._openrouter_completion([])
    assert captured["base_url"] == "https://openrouter.ai/api/v1"
    assert captured["max_retries"] == 0


def _reply(content: str | None, calls: list[Any] | None = None) -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=calls))]
    )


def test_openrouter_tool_calls_normalised_to_loop_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call = SimpleNamespace(
        id="c1",
        type="function",
        function=SimpleNamespace(name="read_file", arguments='{"path": "a.txt"}'),
    )
    sent: dict[str, Any] = {}

    def create(**kwargs: Any) -> Any:
        sent.update(kwargs)
        return _reply(None, [call])

    _use_fake_client(monkeypatch, create)
    message = agent._openrouter_completion([])
    assert sent["model"] == "openai/gpt-6-luna"
    assert message.tool_calls is not None
    assert message.tool_calls[0].id == "c1"
    assert message.tool_calls[0].function.name == "read_file"
    assert message.tool_calls[0].function.arguments == '{"path": "a.txt"}'


# --- retries ------------------------------------------------------------------


def _groq_tool_use_failed() -> groq.BadRequestError:
    request = httpx.Request("POST", "https://api.groq.com/x")
    response = httpx.Response(400, request=request)
    return groq.BadRequestError(
        "tool_use_failed: bad call", response=response, body=None
    )


def test_groq_retry_counted_and_recovers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "sleep", lambda _: None)
    attempts = iter([_groq_tool_use_failed(), _groq_tool_use_failed(), None])

    def create(**_: Any) -> Any:
        err = next(attempts)
        if err is not None:
            raise err
        return _reply("fine")

    message = agent._openai_compat_completion(
        create, "m", [], retry_tool_use_failed=True
    )
    assert message.content == "fine"
    assert agent.get_retry_count() == 2


def test_tool_use_failed_not_retried_when_flag_off() -> None:
    def create(**_: Any) -> Any:
        raise _groq_tool_use_failed()

    with pytest.raises(groq.BadRequestError):
        agent._openai_compat_completion(create, "m", [], retry_tool_use_failed=False)
    assert agent.get_retry_count() == 0


# --- eval runners ---------------------------------------------------------------


def test_run_eval_requires_provider_and_model(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["run_eval.py"])
    with pytest.raises(SystemExit) as info:
        run_eval.parse_args()
    assert info.value.code == 2
    err = capsys.readouterr().err
    assert "--provider" in err and "--model" in err


def test_run_passk_requires_provider_and_model(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["run_passk.py", "-k", "3", "--provider", "groq"])
    with pytest.raises(SystemExit):
        run_passk.parse_args()
    assert "--model" in capsys.readouterr().err


def test_run_task_records_provider_model_retries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(run_eval, "TRACES_DIR", tmp_path)
    agent.configure("openrouter", "anthropic/claude-haiku-5.5")

    def fake_run_agent(task: str, *, mode: Any) -> str:
        agent._retry_count += 3
        return "42"

    monkeypatch.setattr(run_eval, "run_agent", fake_run_agent)
    result = run_eval.run_task(
        {"id": "t", "category": "c", "task": "q", "expect": "42"}
    )
    assert result["status"] == "PASS"
    assert result["provider"] == "openrouter"
    assert result["model"] == "anthropic/claude-haiku-5.5"
    assert result["retries"] == 3


def test_passk_results_file_records_provider_and_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(run_passk, "RESULTS_DIR", tmp_path)
    runs: list[list[run_eval.Result]] = [[], []]
    record: run_passk.TaskPassK = {
        "id": "t", "category": "c", "task": "q", "k": 2, "passes": 2,
        "fails": 0, "errors": 0, "all_passed": True, "error_limited": False,
        "statuses": ["PASS", "PASS"],
    }
    written = run_passk.write_passk_results(
        2, runs, [record], [1.0, 1.0], "openrouter", "openai/gpt-6-luna"
    )
    data = json.loads(written.read_text(encoding="utf-8"))
    assert data["provider"] == "openrouter"
    assert data["model"] == "openai/gpt-6-luna"


# --- 429 retry ----------------------------------------------------------------


def test_429_retries_twice_with_backoff_and_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(agent.time, "sleep", sleeps.append)
    calls = 0

    def create(**_: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise _status_error(429)
        msg = SimpleNamespace(content="ok", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    _use_fake_client(monkeypatch, create)
    agent.reset_retry_count()
    assert agent._openrouter_completion([]).content == "ok"
    assert sleeps == [2.0, 5.0]
    assert agent.get_retry_count() == 2


def test_persistent_429_gives_up_after_two_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent.time, "sleep", lambda _s: None)
    calls = 0

    def create(**_: Any) -> Any:
        nonlocal calls
        calls += 1
        raise _status_error(429)

    _use_fake_client(monkeypatch, create)
    agent.reset_retry_count()
    with pytest.raises(agent.ProviderError):
        agent._openrouter_completion([])
    assert calls == 3
    assert agent.get_retry_count() == 2


@pytest.mark.parametrize("code", [401, 402])
def test_401_402_not_retried(monkeypatch: pytest.MonkeyPatch, code: int) -> None:
    monkeypatch.setattr(agent.time, "sleep", lambda _s: pytest.fail("slept"))
    calls = 0

    def create(**_: Any) -> Any:
        nonlocal calls
        calls += 1
        raise _status_error(code)

    _use_fake_client(monkeypatch, create)
    with pytest.raises(agent.FatalProviderError):
        agent._openrouter_completion([])
    assert calls == 1


def test_gemini_style_429_code_is_recognised() -> None:
    assert agent._is_rate_limited(SimpleNamespace(code=429))  # type: ignore[arg-type]
    assert not agent._is_rate_limited(SimpleNamespace(code=500))  # type: ignore[arg-type]
