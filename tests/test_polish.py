from __future__ import annotations

import json
import time
from typing import Any

from pytest import MonkeyPatch

from voxis.config import VoxisConfig
from voxis.diarization_types import Turn
from voxis.llm import LLMResponse
from voxis.polish import POLISH_CHUNK_TARGET_CHARS, _strip_code_fence, polish_turns


def _config() -> VoxisConfig:
    return VoxisConfig(
        llm_base_url="https://api.example/v1",
        llm_api_key="key",
        polish_model="model",
        speaker_map_model="speaker-map-model",
    )


def _json_response(texts: list[str]) -> str:
    return json.dumps(
        {"texts": {str(index): text for index, text in enumerate(texts)}},
        ensure_ascii=False,
    )


def test_strip_code_fence_handles_markdown_fence() -> None:
    assert _strip_code_fence("```text\nSPEAKER_00: hello\n```") == "SPEAKER_00: hello"


def test_polish_turns_uses_schema_and_grafts_original_metadata(monkeypatch: MonkeyPatch) -> None:
    calls: dict[str, Any] = {}

    def fake_call_json_schema_llm(
        prompt: str,
        *,
        config: VoxisConfig,
        model: str,
        schema_name: str,
        schema: dict[str, object],
        max_tokens: int = 4096,
        extra_body: dict[str, object] | None = None,
    ) -> LLMResponse:
        calls.update(
            {
                "prompt": prompt,
                "config": config,
                "model": model,
                "schema_name": schema_name,
                "schema": schema,
                "max_tokens": max_tokens,
                "extra_body": extra_body,
            }
        )
        return LLMResponse("```json\n" + _json_response(["Hello."]) + "\n```", 1.0, 7, 3)

    monkeypatch.setattr("voxis.polish.call_json_schema_llm", fake_call_json_schema_llm)
    config = _config()

    result = polish_turns([Turn("SPEAKER_00", 0.0, 1.0, "hello")], context="Claude Code", config=config)

    assert calls["config"] == config
    assert calls["model"] == "model"
    assert calls["schema_name"] == "polished_turns"
    assert calls["schema"]["required"] == ["texts"]
    assert calls["schema"]["properties"]["texts"]["required"] == ["0"]  # type: ignore[index]
    assert calls["schema"]["properties"]["texts"]["additionalProperties"] is False  # type: ignore[index]
    assert calls["schema"]["properties"]["texts"]["properties"] == {"0": {"type": "string"}}  # type: ignore[index]
    assert calls["extra_body"] == {"enable_thinking": False}
    assert result.text == "[00:00:00.00 - 00:00:01.00] SPEAKER_00: Hello."
    assert result.status == "succeeded"
    assert result.prompt_tokens == 7


def test_polish_turns_passes_context_for_term_correction(monkeypatch: MonkeyPatch) -> None:
    calls: dict[str, Any] = {}

    def fake_call_json_schema_llm(
        prompt: str,
        *,
        config: VoxisConfig,
        model: str,
        schema_name: str,
        schema: dict[str, object],
        max_tokens: int = 4096,
        extra_body: dict[str, object] | None = None,
    ) -> LLMResponse:
        calls["saw_context"] = "Product terms: Supabase." in prompt
        return LLMResponse(_json_response(["We built the backend on Supabase."]), 1.0, 7, 3)

    monkeypatch.setattr("voxis.polish.call_json_schema_llm", fake_call_json_schema_llm)

    result = polish_turns(
        [Turn("SPEAKER_00", 0.0, 2.0, "we built the backend on super base")],
        context="Product terms: Supabase.",
        config=_config(),
    )

    assert calls["saw_context"] is True
    assert result.text == "[00:00:00.00 - 00:00:02.00] SPEAKER_00: We built the backend on Supabase."
    assert result.status == "succeeded"


def test_polish_turns_rejects_malformed_json(monkeypatch: MonkeyPatch) -> None:
    def fake_call_json_schema_llm(*args: Any, **kwargs: Any) -> LLMResponse:
        return LLMResponse("SPEAKER_00: Hello.", 1.0, 7, 3)

    monkeypatch.setattr("voxis.polish.call_json_schema_llm", fake_call_json_schema_llm)

    result = polish_turns([Turn("SPEAKER_00", 0.0, 1.0, "hello")], context=None, config=_config())

    assert result.status == "succeeded_with_fallback"
    assert result.text == "[00:00:00.00 - 00:00:01.00] SPEAKER_00: hello"
    assert result.issues == ("error:MALFORMED_JSON",)
    assert result.chunk_count == 1
    assert result.chunks[0]["status"] == "failed_validation"
    assert result.chunks[0]["issues"][0]["code"] == "MALFORMED_JSON"


def test_polish_turns_rejects_missing_text_key(monkeypatch: MonkeyPatch) -> None:
    def fake_call_json_schema_llm(*args: Any, **kwargs: Any) -> LLMResponse:
        return LLMResponse(_json_response(["Only one."]), 1.0, 7, 3)

    monkeypatch.setattr("voxis.polish.call_json_schema_llm", fake_call_json_schema_llm)
    turns = [
        Turn("SPEAKER_00", 0.0, 1.0, "first"),
        Turn("SPEAKER_00", 1.0, 2.0, "second"),
    ]

    result = polish_turns(turns, context=None, config=_config())

    assert result.status == "succeeded_with_fallback"
    assert result.text == "\n".join(
        [
            "[00:00:00.00 - 00:00:01.00] SPEAKER_00: first",
            "[00:00:01.00 - 00:00:02.00] SPEAKER_00: second",
        ]
    )
    assert "MISSING_TURN_KEY" in {issue["code"] for issue in result.chunks[0]["issues"]}


def test_polish_turns_rejects_extra_text_key(monkeypatch: MonkeyPatch) -> None:
    def fake_call_json_schema_llm(*args: Any, **kwargs: Any) -> LLMResponse:
        return LLMResponse(json.dumps({"texts": {"0": "Hello.", "9": "Extra."}}), 1.0, 7, 3)

    monkeypatch.setattr("voxis.polish.call_json_schema_llm", fake_call_json_schema_llm)

    result = polish_turns([Turn("SPEAKER_00", 0.0, 1.0, "hello")], context=None, config=_config())

    assert result.status == "succeeded_with_fallback"
    assert result.text == "[00:00:00.00 - 00:00:01.00] SPEAKER_00: hello"
    assert "EXTRA_TURN_KEY" in {issue["code"] for issue in result.chunks[0]["issues"]}


def test_polish_turns_rejects_structural_text_leak(monkeypatch: MonkeyPatch) -> None:
    def fake_call_json_schema_llm(*args: Any, **kwargs: Any) -> LLMResponse:
        return LLMResponse(_json_response(["Alice: Hello."]), 1.0, 7, 3)

    monkeypatch.setattr("voxis.polish.call_json_schema_llm", fake_call_json_schema_llm)

    result = polish_turns([Turn("Alice", 0.0, 1.0, "hello")], context=None, config=_config())

    assert result.status == "succeeded_with_fallback"
    assert result.text == "[00:00:00.00 - 00:00:01.00] Alice: hello"
    assert "STRUCTURAL_TEXT_LEAK" in {issue["code"] for issue in result.chunks[0]["issues"]}


def test_polish_turns_allows_ordinary_colon_text(monkeypatch: MonkeyPatch) -> None:
    def fake_call_json_schema_llm(*args: Any, **kwargs: Any) -> LLMResponse:
        return LLMResponse(_json_response(["Question: what should we do next?"]), 1.0, 7, 3)

    monkeypatch.setattr("voxis.polish.call_json_schema_llm", fake_call_json_schema_llm)

    result = polish_turns([Turn("Alice", 0.0, 1.0, "question what should we do next")], context=None, config=_config())

    assert result.status == "succeeded"
    assert result.text == "[00:00:00.00 - 00:00:01.00] Alice: Question: what should we do next?"


def test_polish_turns_rejects_heading_text(monkeypatch: MonkeyPatch) -> None:
    def fake_call_json_schema_llm(*args: Any, **kwargs: Any) -> LLMResponse:
        return LLMResponse(_json_response(["## Transcript"]), 1.0, 7, 3)

    monkeypatch.setattr("voxis.polish.call_json_schema_llm", fake_call_json_schema_llm)

    result = polish_turns([Turn("SPEAKER_00", 0.0, 1.0, "hello")], context=None, config=_config())

    assert result.status == "succeeded_with_fallback"
    assert result.chunks[0]["issues"][0]["code"] == "MARKDOWN_HEADING"


def test_polish_turns_chunks_and_keeps_valid_chunk_when_later_chunk_fails(monkeypatch: MonkeyPatch) -> None:
    prompts: list[str] = []

    def fake_call_json_schema_llm(prompt: str, *args: Any, **kwargs: Any) -> LLMResponse:
        prompts.append(prompt)
        response = _json_response(["Clean first."]) if '"text": "first"' in prompt else "not json"
        return LLMResponse(response, 1.0, 7, 3)

    monkeypatch.setattr("voxis.polish.call_json_schema_llm", fake_call_json_schema_llm)
    turns = [
        Turn("SPEAKER_00", 0.0, 1.0, "first"),
        Turn("SPEAKER_01", 2.0, 3.0, "x" * POLISH_CHUNK_TARGET_CHARS),
    ]

    result = polish_turns(turns, context=None, config=_config())

    assert len(prompts) == 2
    assert result.status == "succeeded_with_fallback"
    assert result.chunk_count == 2
    assert result.text.startswith("[00:00:00.00 - 00:00:01.00] SPEAKER_00: Clean first.")
    assert f"[00:00:02.00 - 00:00:03.00] SPEAKER_01: {'x' * POLISH_CHUNK_TARGET_CHARS}" in result.text
    assert result.chunks[0]["status"] == "succeeded"
    assert result.chunks[1]["status"] == "failed_validation"


def test_polish_turns_preserves_chunk_order_when_parallel_calls_finish_out_of_order(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr("voxis.polish.POLISH_CHUNK_TARGET_CHARS", 20)

    def fake_call_json_schema_llm(prompt: str, *args: Any, **kwargs: Any) -> LLMResponse:
        if "chunk-0 original" in prompt:
            time.sleep(0.03)
            return LLMResponse(_json_response(["Clean zero."]), 0.03, 7, 3)
        if "chunk-1 original" in prompt:
            time.sleep(0.01)
            return LLMResponse(_json_response(["Clean one."]), 0.01, 7, 3)
        return LLMResponse(_json_response(["Clean two."]), 0.0, 7, 3)

    monkeypatch.setattr("voxis.polish.call_json_schema_llm", fake_call_json_schema_llm)
    config = VoxisConfig(
        llm_base_url="https://api.example/v1",
        llm_api_key="key",
        polish_model="model",
        speaker_map_model="speaker-map-model",
        polish_max_workers=3,
    )
    turns = [
        Turn("SPEAKER_00", 0.0, 1.0, "chunk-0 original"),
        Turn("SPEAKER_00", 2.0, 3.0, "chunk-1 original"),
        Turn("SPEAKER_00", 4.0, 5.0, "chunk-2 original"),
    ]

    result = polish_turns(turns, context=None, config=config)

    assert result.status == "succeeded"
    assert result.chunk_count == 3
    assert result.max_workers == 3
    assert result.effective_workers == 3
    assert [chunk["index"] for chunk in result.chunks] == [0, 1, 2]
    assert result.text.index("Clean zero.") < result.text.index("Clean one.") < result.text.index("Clean two.")


def test_polish_turns_keeps_parallel_fallback_isolated(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr("voxis.polish.POLISH_CHUNK_TARGET_CHARS", 20)

    def fake_call_json_schema_llm(prompt: str, *args: Any, **kwargs: Any) -> LLMResponse:
        if "chunk-1 original" in prompt:
            return LLMResponse("not json", 0.0, 7, 3)
        if "chunk-0 original" in prompt:
            return LLMResponse(_json_response(["Clean zero."]), 0.0, 7, 3)
        return LLMResponse(_json_response(["Clean two."]), 0.0, 7, 3)

    monkeypatch.setattr("voxis.polish.call_json_schema_llm", fake_call_json_schema_llm)
    config = VoxisConfig(
        llm_base_url="https://api.example/v1",
        llm_api_key="key",
        polish_model="model",
        speaker_map_model="speaker-map-model",
        polish_max_workers=3,
    )
    turns = [
        Turn("SPEAKER_00", 0.0, 1.0, "chunk-0 original"),
        Turn("SPEAKER_00", 2.0, 3.0, "chunk-1 original"),
        Turn("SPEAKER_00", 4.0, 5.0, "chunk-2 original"),
    ]

    result = polish_turns(turns, context=None, config=config)

    assert result.status == "succeeded_with_fallback"
    assert [chunk["status"] for chunk in result.chunks] == ["succeeded", "failed_validation", "succeeded"]
    assert "Clean zero." in result.text
    assert "[00:00:02.00 - 00:00:03.00] SPEAKER_00: chunk-1 original" in result.text
    assert "Clean two." in result.text
