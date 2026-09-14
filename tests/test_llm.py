from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from openai import APIConnectionError, BadRequestError
from pytest import MonkeyPatch

from voxis.config import VoxisConfig
from voxis.llm import call_json_schema_llm, call_llm, no_thinking_extra_body


class FakeCompletions:
    def __init__(self, errors: list[Exception]) -> None:
        self.errors = errors
        self.calls = 0
        self.call_kwargs: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:  # noqa: ANN401 - mimics OpenAI response object.
        self.calls += 1
        self.call_kwargs.append(kwargs)
        if self.errors:
            raise self.errors.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=4),
        )


class FakeOpenAIClient:
    def __init__(self, completions: FakeCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)


def test_call_llm_retries_retryable_errors(monkeypatch: MonkeyPatch) -> None:
    request = httpx.Request("POST", "https://example.test/chat/completions")
    completions = FakeCompletions([APIConnectionError(message="temporary", request=request)])
    monkeypatch.setattr("voxis.llm.OpenAI", lambda **kwargs: FakeOpenAIClient(completions))
    monkeypatch.setattr("voxis.llm.time.sleep", lambda seconds: None)

    response = call_llm("hello", config=make_config())

    assert response.text == "ok"
    assert response.prompt_tokens == 3
    assert completions.calls == 2


def test_call_llm_does_not_retry_bad_requests(monkeypatch: MonkeyPatch) -> None:
    request = httpx.Request("POST", "https://example.test/chat/completions")
    response = httpx.Response(400, request=request)
    completions = FakeCompletions([BadRequestError("bad request", response=response, body=None)])
    monkeypatch.setattr("voxis.llm.OpenAI", lambda **kwargs: FakeOpenAIClient(completions))
    monkeypatch.setattr("voxis.llm.time.sleep", lambda seconds: None)

    with pytest.raises(BadRequestError):
        call_llm("hello", config=make_config())

    assert completions.calls == 1


def test_call_json_schema_llm_passes_strict_schema_wrapper(monkeypatch: MonkeyPatch) -> None:
    completions = FakeCompletions([])
    monkeypatch.setattr("voxis.llm.OpenAI", lambda **kwargs: FakeOpenAIClient(completions))
    schema: dict[str, object] = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }

    response = call_json_schema_llm(
        "return json",
        config=make_config(),
        model="schema-model",
        schema_name="answer_schema",
        schema=schema,
        max_tokens=123,
        extra_body={"enable_thinking": False},
    )

    assert response.text == "ok"
    assert completions.calls == 1
    kwargs = completions.call_kwargs[0]
    assert kwargs["model"] == "schema-model"
    assert kwargs["messages"] == [{"role": "user", "content": "return json"}]
    assert kwargs["max_tokens"] == 123
    assert kwargs["extra_body"] == {"enable_thinking": False}
    assert kwargs["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "answer_schema",
            "strict": True,
            "schema": schema,
        },
    }


def make_config() -> VoxisConfig:
    return VoxisConfig(
        llm_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        llm_api_key="key",
        polish_model="deepseek-v4-flash",
        speaker_map_model="deepseek-v4-flash",
    )


def test_no_thinking_extra_body_respects_config() -> None:
    assert no_thinking_extra_body(make_config()) == {"enable_thinking": False}

    enabled = VoxisConfig(
        llm_base_url="https://api.openai.com/v1",
        llm_api_key="key",
        polish_model="gpt-5",
        speaker_map_model="gpt-5-mini",
        llm_disable_thinking=False,
    )
    assert no_thinking_extra_body(enabled) is None


def test_deepseek_json_object_and_omitted_completion_cap(monkeypatch: MonkeyPatch) -> None:
    from dataclasses import replace

    completions = FakeCompletions([])
    monkeypatch.setattr("voxis.llm.OpenAI", lambda **kwargs: FakeOpenAIClient(completions))
    config = replace(make_config(), llm_base_url="https://api.deepseek.com")
    call_json_schema_llm(
        "return JSON",
        config=config,
        model="deepseek-flash",
        schema_name="map",
        schema={},
        max_tokens=None,
        extra_body={"thinking": {"type": "enabled"}, "reasoning_effort": "low"},
    )
    kwargs = completions.call_kwargs[0]
    assert "max_tokens" not in kwargs
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["extra_body"]["reasoning_effort"] == "low"
