from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import urlsplit

from openai import APIConnectionError, APIStatusError, APITimeoutError, InternalServerError, OpenAI, RateLimitError

from voxis.config import VoxisConfig

DEFAULT_MAX_ATTEMPTS = 3
RETRY_BASE_DELAY_SECONDS = 0.5
NO_THINKING_EXTRA_BODY: dict[str, object] = {"enable_thinking": False}


def no_thinking_extra_body(config: VoxisConfig) -> dict[str, object] | None:
    if config.llm_disable_thinking:
        return dict(NO_THINKING_EXTRA_BODY)
    return None


@dataclass(frozen=True)
class LLMResponse:
    text: str
    elapsed_seconds: float
    prompt_tokens: int
    completion_tokens: int


def call_llm(
    prompt: str,
    *,
    config: VoxisConfig,
    model: str | None = None,
    max_tokens: int | None = 65_536,
    response_format: Mapping[str, object] | None = None,
    extra_body: Mapping[str, object] | None = None,
) -> LLMResponse:
    if config.llm_api_key is None:
        raise ValueError("VOXIS_LLM_API_KEY (or BAILIAN_API_KEY) is required")
    client = OpenAI(api_key=config.llm_api_key, base_url=config.llm_base_url)
    started = time.monotonic()
    kwargs: dict[str, Any] = {
        "model": model or config.polish_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if response_format is not None:
        kwargs["response_format"] = dict(response_format)
    if extra_body is not None:
        kwargs["extra_body"] = dict(extra_body)
    response = _create_completion_with_retry(client, kwargs)
    elapsed = time.monotonic() - started
    message = response.choices[0].message.content or ""
    usage: Any = response.usage
    return LLMResponse(
        text=message,
        elapsed_seconds=elapsed,
        prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
        completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
    )


def _create_completion_with_retry(client: OpenAI, kwargs: dict[str, Any]) -> Any:
    for attempt in range(1, DEFAULT_MAX_ATTEMPTS + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            if attempt >= DEFAULT_MAX_ATTEMPTS or not _retryable_llm_error(exc):
                raise
            time.sleep(RETRY_BASE_DELAY_SECONDS * attempt)
    raise RuntimeError("unreachable LLM retry state")


def _retryable_llm_error(exc: Exception) -> bool:
    if isinstance(exc, APIConnectionError | APITimeoutError | RateLimitError | InternalServerError):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in {408, 409, 425, 429} or exc.status_code >= 500
    return False


def call_json_schema_llm(
    prompt: str,
    *,
    config: VoxisConfig,
    model: str,
    schema_name: str,
    schema: Mapping[str, object],
    max_tokens: int | None = 4096,
    extra_body: Mapping[str, object] | None = None,
) -> LLMResponse:
    # Official DeepSeek Chat Completions supports JSON-object output, not JSON schema.
    response_format: dict[str, object] = (
        {"type": "json_object"}
        if urlsplit(config.llm_base_url).hostname == "api.deepseek.com"
        else {"type": "json_schema", "json_schema": {"name": schema_name, "strict": True, "schema": dict(schema)}}
    )
    return call_llm(
        prompt,
        config=config,
        model=model,
        max_tokens=max_tokens,
        extra_body=extra_body,
        response_format=response_format,
    )
