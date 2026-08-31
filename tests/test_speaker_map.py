from __future__ import annotations

import pytest
from pytest import MonkeyPatch

from voxis.config import VoxisConfig
from voxis.diarization_types import Turn
from voxis.llm import LLMResponse
from voxis.speaker_map import (
    ROLE_INTRODUCTION_FAILURE_KIND,
    apply_speaker_map_to_turns,
    build_speaker_map,
    detect_role_introductions,
    speaker_map_payload,
)


@pytest.mark.parametrize(
    ("context", "welcome", "ack", "addressed_name", "target_cluster"),
    [
        (
            "Title: Martin Casado on Where the Value Is Going in AI",
            "Hello everyone. Martin, welcome to MTS",
            "Super happy to be here. Thanks for having me.",
            "Martin Casado",
            "SPEAKER_02",
        ),
        (
            "Title: Michael Kratsios: Science Agenda",
            "Michael Kratsios, welcome to the interview",
            "Thank you for having me. Excited to be here.",
            "Michael Kratsios",
            "SPEAKER_01",
        ),
    ],
)
def test_detect_role_introductions_finds_audited_production_cases(
    context: str,
    welcome: str,
    ack: str,
    addressed_name: str,
    target_cluster: str,
) -> None:
    evidence = detect_role_introductions(
        [
            Turn("SPEAKER_00", 0.0, 4.0, welcome),
            Turn(target_cluster, 4.1, 6.0, ack),
        ],
        context=context,
    )

    assert evidence == [
        {
            "welcome_turn_index": 0,
            "ack_turn_index": 1,
            "addressed_name": addressed_name,
            "welcoming_cluster": "SPEAKER_00",
            "target_cluster": target_cluster,
            "welcome_phrase": "welcome to",
            "ack_phrase": "happy to be here" if addressed_name == "Martin Casado" else "Thank you for having me",
            "gap_seconds": 0.1,
            "welcome_excerpt": welcome,
            "ack_excerpt": ack,
        }
    ]


@pytest.mark.parametrize(
    ("context", "turns"),
    [
        (
            None,
            [
                Turn("SPEAKER_00", 0.0, 1.0, "Alice, welcome to the show"),
                Turn("SPEAKER_01", 1.0, 2.0, "Thanks for having me"),
            ],
        ),
        (
            "Guests: Alice Smith and Alice Jones",
            [
                Turn("SPEAKER_00", 0.0, 1.0, "Alice, welcome to the show"),
                Turn("SPEAKER_01", 1.0, 2.0, "Thanks for having me"),
            ],
        ),
        (
            "Guest: Alice Smith",
            [
                Turn("SPEAKER_00", 0.0, 1.0, "In this clip Alice says welcome to the show"),
                Turn("SPEAKER_01", 1.0, 2.0, "Thanks for having me"),
            ],
        ),
        (
            "Guest: Alice Smith",
            [
                Turn("SPEAKER_00", 0.0, 1.0, "Alice, welcome to the show"),
                Turn(
                    "SPEAKER_01",
                    1.0,
                    2.0,
                    "Before we start, the trailer ends with someone saying thank you for having me, then fades out",
                ),
            ],
        ),
        (
            "Guest: Alice Smith",
            [
                Turn("SPEAKER_00", 0.0, 1.0, "Alice, welcome to the show"),
                Turn("SPEAKER_00", 1.0, 2.0, "Thanks for having me"),
            ],
        ),
        (
            "Guest: Alice Smith",
            [
                Turn("SPEAKER_00", 0.0, 1.0, "Alice, welcome to the show"),
                Turn("SPEAKER_01", 4.0, 5.0, "Thanks for having me"),
            ],
        ),
        (
            "Guest: Alice Smith",
            [
                Turn("SPEAKER_00", 0.0, 1.0, "Alice, welcome to the show"),
                Turn("SPEAKER_01", 1.0, 2.0, "Let us begin with the topic"),
            ],
        ),
        (
            "Newsletter: https://creatoreconomy.so/p/vibe-coding. Guest: Amol Jain.",
            [
                Turn("SPEAKER_00", 0.0, 1.0, "So welcome sir"),
                Turn("SPEAKER_01", 1.1, 2.0, "Thank Peter. Great to be here."),
            ],
        ),
        (
            "The summary says so in ordinary prose. Guest: Amol Jain.",
            [
                Turn("SPEAKER_00", 0.0, 1.0, "So welcome sir"),
                Turn("SPEAKER_01", 1.1, 2.0, "Thank Peter. Great to be here."),
            ],
        ),
    ],
)
def test_detect_role_introductions_rejects_ambiguous_or_weak_evidence(
    context: str | None,
    turns: list[Turn],
) -> None:
    assert detect_role_introductions(turns, context=context) == []


def test_build_speaker_map_fails_closed_on_role_introduction_conflict(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        "voxis.speaker_map.call_json_schema_llm",
        lambda *args, **kwargs: LLMResponse(
            '{"mapping":{"SPEAKER_00":"Michael Kratsios","SPEAKER_01":null}}',
            elapsed_seconds=0.2,
            prompt_tokens=11,
            completion_tokens=7,
        ),
    )
    turns = [
        Turn("SPEAKER_00", 0.0, 4.0, "Michael Kratsios, welcome to the interview"),
        Turn("SPEAKER_01", 4.1, 8.0, "Thank you for having me. Excited to be here."),
    ]

    result = build_speaker_map(turns, context="Title: Michael Kratsios: Science Agenda", config=make_config())

    assert result.status == "failed"
    assert result.failure_kind == ROLE_INTRODUCTION_FAILURE_KIND
    assert result.mapping == {
        "SPEAKER_00": "Michael Kratsios",
        "SPEAKER_01": "Unknown (SPEAKER_01)",
    }
    assert [issue["code"] for issue in result.issues] == ["ROLE_INTRODUCTION_CONFLICT"]
    assert result.issues[0]["value"]["reasons"] == [
        "target_cluster_name_mismatch",
        "welcoming_cluster_claims_guest",
    ]
    payload = speaker_map_payload(result)
    assert payload["role_introduction"]["conflict_count"] == 1
    assert payload["role_introduction"]["evidence"] == result.role_introduction_evidence


def test_build_speaker_map_accepts_consistent_role_introduction(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        "voxis.speaker_map.call_json_schema_llm",
        lambda *args, **kwargs: LLMResponse(
            '{"mapping":{"SPEAKER_00":"Brett Winton","SPEAKER_01":"Dylan Robbins"}}',
            elapsed_seconds=0.2,
            prompt_tokens=11,
            completion_tokens=7,
        ),
    )
    turns = [
        Turn("SPEAKER_00", 0.0, 4.0, "Dylan, welcome to the pod"),
        Turn("SPEAKER_01", 4.1, 8.0, "Doing great. Thanks for having me."),
    ]

    result = build_speaker_map(
        turns,
        context="Host: Brett Winton. Guest: Dylan Robbins.",
        config=make_config(),
    )

    assert result.status == "succeeded"
    assert result.failure_kind is None
    assert result.mapping == {"SPEAKER_00": "Brett Winton", "SPEAKER_01": "Dylan Robbins"}
    assert result.role_introduction_evidence[0]["addressed_name"] == "Dylan Robbins"


def test_build_speaker_map_fails_when_probe_speaker_is_omitted(monkeypatch: MonkeyPatch) -> None:
    def fake_call_json_schema_llm(
        prompt: str,
        *,
        config: VoxisConfig,
        model: str,
        schema_name: str,
        schema: dict[str, object],
        max_tokens: int = 2048,
        extra_body: dict[str, object] | None = None,
    ) -> LLMResponse:
        assert model == "speaker-map-model"
        assert schema_name == "speaker_map_response"
        assert schema["required"] == ["mapping"]
        assert isinstance(schema["properties"], dict)
        mapping_schema = schema["properties"]["mapping"]
        assert isinstance(mapping_schema, dict)
        assert mapping_schema["required"] == ["SPEAKER_00", "SPEAKER_01"]
        assert mapping_schema["additionalProperties"] is False
        assert mapping_schema["properties"] == {
            "SPEAKER_00": {"type": ["string", "null"]},
            "SPEAKER_01": {"type": ["string", "null"]},
        }
        assert max_tokens == 256
        assert extra_body == {"enable_thinking": False}
        return LLMResponse(
            '{"mapping":{"SPEAKER_00": "Alice", "SPEAKER_99": "Extra"}}',
            elapsed_seconds=0.2,
            prompt_tokens=11,
            completion_tokens=7,
        )

    monkeypatch.setattr("voxis.speaker_map.call_json_schema_llm", fake_call_json_schema_llm)
    config = make_config()
    turns = [
        Turn("SPEAKER_00", 0.0, 1.0, "hello"),
        Turn("SPEAKER_01", 1.0, 2.0, "world"),
    ]

    result = build_speaker_map(turns, context="Host: Alice", config=config)

    assert result.status == "failed"
    assert result.failure_kind == "invalid_output"
    assert result.model == "speaker-map-model"
    assert result.thinking_enabled is False
    assert result.mapping == {"SPEAKER_00": "Alice"}
    assert {issue["code"] for issue in result.issues} == {"EXTRA_SPEAKER_KEY", "MISSING_SPEAKER_KEY"}
    payload = speaker_map_payload(result)
    assert payload["schema_version"] == "speaker_map_v1"
    assert payload["failure_kind"] == "invalid_output"
    assert payload["model"] == "speaker-map-model"
    assert payload["thinking_enabled"] is False
    assert payload["prompt_tokens"] == 11


def test_build_speaker_map_runs_probe_only_without_context(monkeypatch: MonkeyPatch) -> None:
    def fake_call_json_schema_llm(
        prompt: str,
        *,
        config: VoxisConfig,
        model: str,
        schema_name: str,
        schema: dict[str, object],
        max_tokens: int = 2048,
        extra_body: dict[str, object] | None = None,
    ) -> LLMResponse:
        return LLMResponse(
            '{"mapping":{"SPEAKER_00":"Alice"}}',
            elapsed_seconds=0.2,
            prompt_tokens=11,
            completion_tokens=7,
        )

    monkeypatch.setattr("voxis.speaker_map.call_json_schema_llm", fake_call_json_schema_llm)

    result = build_speaker_map([Turn("SPEAKER_00", 0.0, 1.0, "I'm Alice")], context=None, config=make_config())

    assert result.status == "succeeded"
    assert result.evidence_mode == "probe_only"
    assert result.mapping == {"SPEAKER_00": "Alice"}
    assert result.probe_turns[0]["speaker"] == "SPEAKER_00"


def test_build_speaker_map_skips_without_llm_key() -> None:
    turns = [Turn("SPEAKER_00", 0.0, 1.0, "hello")]

    result = build_speaker_map(turns, context=None, config=make_config(api_key=None))

    assert result.status == "skipped_no_llm_key"
    assert result.evidence_mode == "probe_only"


def test_probe_only_speaker_map_falls_back_for_names_not_in_probe(monkeypatch: MonkeyPatch) -> None:
    def fake_call_json_schema_llm(
        prompt: str,
        *,
        config: VoxisConfig,
        model: str,
        schema_name: str,
        schema: dict[str, object],
        max_tokens: int = 2048,
        extra_body: dict[str, object] | None = None,
    ) -> LLMResponse:
        return LLMResponse(
            '{"mapping":{"SPEAKER_00":"Alice"}}',
            elapsed_seconds=0.2,
            prompt_tokens=11,
            completion_tokens=7,
        )

    monkeypatch.setattr("voxis.speaker_map.call_json_schema_llm", fake_call_json_schema_llm)

    result = build_speaker_map([Turn("SPEAKER_00", 0.0, 1.0, "hello")], context=None, config=make_config())

    assert result.status == "succeeded_with_issues"
    assert result.evidence_mode == "probe_only"
    assert result.mapping == {"SPEAKER_00": "Unknown (SPEAKER_00)"}
    assert [issue["code"] for issue in result.issues] == ["UNSUPPORTED_NAME"]
    assert {issue["severity"] for issue in result.issues} == {"warning"}


def test_build_speaker_map_probe_includes_first_turns_and_late_speaker_windows(monkeypatch: MonkeyPatch) -> None:
    captured_prompt = ""

    def fake_call_json_schema_llm(
        prompt: str,
        *,
        config: VoxisConfig,
        model: str,
        schema_name: str,
        schema: dict[str, object],
        max_tokens: int = 2048,
        extra_body: dict[str, object] | None = None,
    ) -> LLMResponse:
        nonlocal captured_prompt
        captured_prompt = prompt
        return LLMResponse(
            '{"mapping":{"SPEAKER_00":"Alice","SPEAKER_01":"Bob","SPEAKER_02":"Carol"}}',
            elapsed_seconds=0.2,
            prompt_tokens=11,
            completion_tokens=7,
        )

    monkeypatch.setattr("voxis.speaker_map.call_json_schema_llm", fake_call_json_schema_llm)
    turns = [Turn("SPEAKER_00", float(index), float(index) + 0.5, f"turn-{index:02d}") for index in range(10)] + [
        Turn("SPEAKER_00", 10.0, 10.5, "turn-10-before-b"),
        Turn("SPEAKER_00", 11.0, 11.5, "turn-11-before-b"),
        Turn("SPEAKER_01", 12.0, 12.5, "turn-12-first-b"),
        Turn("SPEAKER_01", 13.0, 13.5, "turn-13-after-b"),
        Turn("SPEAKER_01", 14.0, 14.5, "turn-14-after-b"),
        Turn("SPEAKER_00", 15.0, 15.5, "turn-15-excluded"),
        Turn("SPEAKER_00", 16.0, 16.5, "turn-16-before-c"),
        Turn("SPEAKER_01", 17.0, 17.5, "turn-17-before-c"),
        Turn("SPEAKER_02", 18.0, 18.5, "turn-18-first-c"),
        Turn("SPEAKER_02", 19.0, 19.5, "turn-19-after-c"),
    ]

    result = build_speaker_map(turns, context="Host: Alice. Guests: Bob and Carol.", config=make_config())

    assert result.status == "succeeded"
    assert [turn["index"] for turn in result.probe_turns] == [*range(15), 16, 17, 18, 19]
    assert result.probe_turns[0]["reasons"] == ["initial_turn"]
    assert result.probe_turns[10]["reasons"] == ["first_speaker_appearance_window:SPEAKER_01"]
    assert result.probe_turns[-1]["reasons"] == ["first_speaker_appearance_window:SPEAKER_02"]
    assert "turn-00" in captured_prompt
    assert "turn-09" in captured_prompt
    assert "turn-10-before-b" in captured_prompt
    assert "turn-12-first-b" in captured_prompt
    assert "turn-14-after-b" in captured_prompt
    assert "turn-15-excluded" not in captured_prompt
    assert "turn-16-before-c" in captured_prompt
    assert "turn-18-first-c" in captured_prompt


def test_build_speaker_map_fails_when_llm_fails(monkeypatch: MonkeyPatch) -> None:
    def fake_call_json_schema_llm(
        prompt: str,
        *,
        config: VoxisConfig,
        model: str,
        schema_name: str,
        schema: dict[str, object],
        max_tokens: int = 2048,
        extra_body: dict[str, object] | None = None,
    ) -> LLMResponse:
        raise RuntimeError("network down")

    monkeypatch.setattr("voxis.speaker_map.call_json_schema_llm", fake_call_json_schema_llm)

    result = build_speaker_map([Turn("SPEAKER_00", 0.0, 1.0, "hello")], context="Host: Alice", config=make_config())

    assert result.status == "failed"
    assert result.failure_kind == "llm_error"
    assert result.model == "speaker-map-model"
    assert result.mapping == {}
    assert result.issues[0]["code"] == "LLM_ERROR"
    assert "preserving raw speaker labels" not in result.issues[0]["detail"]


def test_apply_speaker_map_to_turns_uses_unknown_values() -> None:
    turns = [
        Turn("SPEAKER_00", 0.0, 1.0, "hello"),
        Turn("SPEAKER_01", 1.0, 2.0, "world"),
    ]

    mapped = apply_speaker_map_to_turns(turns, {"SPEAKER_00": "Alice", "SPEAKER_01": "Unknown (SPEAKER_01)"})

    assert [(turn.speaker, turn.text) for turn in mapped] == [
        ("Alice", "hello"),
        ("Unknown (SPEAKER_01)", "world"),
    ]


def test_malformed_speaker_map_response_fails(monkeypatch: MonkeyPatch) -> None:
    def fake_call_json_schema_llm(
        prompt: str,
        *,
        config: VoxisConfig,
        model: str,
        schema_name: str,
        schema: dict[str, object],
        max_tokens: int = 2048,
        extra_body: dict[str, object] | None = None,
    ) -> LLMResponse:
        return LLMResponse("not json", 0.1, 4, 2)

    monkeypatch.setattr("voxis.speaker_map.call_json_schema_llm", fake_call_json_schema_llm)

    result = build_speaker_map([Turn("SPEAKER_00", 0.0, 1.0, "hello")], context="Host: Alice", config=make_config())

    assert result.status == "failed"
    assert result.failure_kind == "invalid_output"
    assert result.mapping == {}
    assert result.issues[0]["code"] == "MALFORMED_JSON"
    assert result.issues[0]["response_excerpt"] == "not json"
    assert result.issues[0]["response_length"] == 8
    assert "preserving raw speaker labels" not in result.issues[0]["detail"]


def test_speaker_map_thinking_receipt_is_unknown_when_not_explicitly_disabled(monkeypatch: MonkeyPatch) -> None:
    def fake_call_json_schema_llm(
        prompt: str,
        *,
        config: VoxisConfig,
        model: str,
        schema_name: str,
        schema: dict[str, object],
        max_tokens: int = 2048,
        extra_body: dict[str, object] | None = None,
    ) -> LLMResponse:
        assert extra_body is None
        return LLMResponse('{"mapping":{"SPEAKER_00":null}}', 0.1, 4, 2)

    monkeypatch.setattr("voxis.speaker_map.call_json_schema_llm", fake_call_json_schema_llm)
    config = make_config(llm_disable_thinking=False)

    result = build_speaker_map([Turn("SPEAKER_00", 0.0, 1.0, "hello")], context=None, config=config)

    assert result.status == "succeeded"
    assert result.thinking_enabled is None


def test_malformed_speaker_map_response_retries_once(monkeypatch: MonkeyPatch) -> None:
    responses = iter(
        [
            LLMResponse("not json", 0.1, 4, 256),
            LLMResponse('{"mapping":{"SPEAKER_00":"Alice"}}', 0.2, 4, 6),
        ]
    )
    calls = 0

    def fake_call_json_schema_llm(
        prompt: str,
        *,
        config: VoxisConfig,
        model: str,
        schema_name: str,
        schema: dict[str, object],
        max_tokens: int = 2048,
        extra_body: dict[str, object] | None = None,
    ) -> LLMResponse:
        nonlocal calls
        calls += 1
        return next(responses)

    monkeypatch.setattr("voxis.speaker_map.call_json_schema_llm", fake_call_json_schema_llm)

    result = build_speaker_map([Turn("SPEAKER_00", 0.0, 1.0, "I'm Alice")], context=None, config=make_config())

    assert calls == 2
    assert result.status == "succeeded"
    assert result.mapping == {"SPEAKER_00": "Alice"}
    assert result.prompt_tokens == 8
    assert result.completion_tokens == 262


def test_missing_speaker_map_shape_fails(monkeypatch: MonkeyPatch) -> None:
    def fake_call_json_schema_llm(
        prompt: str,
        *,
        config: VoxisConfig,
        model: str,
        schema_name: str,
        schema: dict[str, object],
        max_tokens: int = 2048,
        extra_body: dict[str, object] | None = None,
    ) -> LLMResponse:
        return LLMResponse("{}", 0.1, 4, 2)

    monkeypatch.setattr("voxis.speaker_map.call_json_schema_llm", fake_call_json_schema_llm)

    result = build_speaker_map([Turn("SPEAKER_00", 0.0, 1.0, "hello")], context="Host: Alice", config=make_config())

    assert result.status == "failed"
    assert result.mapping == {}
    assert result.issues[0]["code"] == "BAD_SHAPE"
    assert "preserving raw speaker labels" not in result.issues[0]["detail"]


def test_semantic_speaker_map_errors_do_not_retry(monkeypatch: MonkeyPatch) -> None:
    calls = 0

    def fake_call_json_schema_llm(
        prompt: str,
        *,
        config: VoxisConfig,
        model: str,
        schema_name: str,
        schema: dict[str, object],
        max_tokens: int = 2048,
        extra_body: dict[str, object] | None = None,
    ) -> LLMResponse:
        nonlocal calls
        calls += 1
        return LLMResponse('{"mapping":{"SPEAKER_00":"Alice"}}', 0.1, 4, 2)

    monkeypatch.setattr("voxis.speaker_map.call_json_schema_llm", fake_call_json_schema_llm)

    result = build_speaker_map([Turn("SPEAKER_00", 0.0, 1.0, "hello")], context=None, config=make_config())

    assert calls == 1
    assert result.status == "succeeded_with_issues"
    assert result.mapping == {"SPEAKER_00": "Unknown (SPEAKER_00)"}
    assert [issue["code"] for issue in result.issues] == ["UNSUPPORTED_NAME"]
    assert {issue["severity"] for issue in result.issues} == {"warning"}


def test_duplicate_real_name_mappings_succeed_with_warning(monkeypatch: MonkeyPatch) -> None:
    def fake_call_json_schema_llm(
        prompt: str,
        *,
        config: VoxisConfig,
        model: str,
        schema_name: str,
        schema: dict[str, object],
        max_tokens: int = 2048,
        extra_body: dict[str, object] | None = None,
    ) -> LLMResponse:
        return LLMResponse(
            '{"mapping":{"SPEAKER_00":"Alice","SPEAKER_01":"Jared Friedman","SPEAKER_02":"Jared Friedman"}}',
            0.1,
            4,
            2,
        )

    monkeypatch.setattr("voxis.speaker_map.call_json_schema_llm", fake_call_json_schema_llm)
    turns = [
        Turn("SPEAKER_00", 0.0, 1.0, "hello"),
        Turn("SPEAKER_01", 1.0, 2.0, "first"),
        Turn("SPEAKER_02", 2.0, 3.0, "second"),
    ]

    result = build_speaker_map(turns, context="Host: Alice. Guests: Jared Friedman.", config=make_config())
    mapped = apply_speaker_map_to_turns(turns, result.mapping)

    assert result.status == "succeeded_with_issues"
    assert result.mapping == {
        "SPEAKER_00": "Alice",
        "SPEAKER_01": "Jared Friedman",
        "SPEAKER_02": "Jared Friedman",
    }
    assert [issue["code"] for issue in result.issues] == ["DUPLICATE_REAL_NAME", "DUPLICATE_REAL_NAME"]
    assert {issue["severity"] for issue in result.issues} == {"warning"}
    assert [(turn.speaker, turn.text) for turn in mapped] == [
        ("Alice", "hello"),
        ("Jared Friedman", "first"),
        ("Jared Friedman", "second"),
    ]


def test_duplicate_unsupported_real_names_fall_back_to_unknown(monkeypatch: MonkeyPatch) -> None:
    def fake_call_json_schema_llm(
        prompt: str,
        *,
        config: VoxisConfig,
        model: str,
        schema_name: str,
        schema: dict[str, object],
        max_tokens: int = 2048,
        extra_body: dict[str, object] | None = None,
    ) -> LLMResponse:
        return LLMResponse(
            '{"mapping":{"SPEAKER_01":"Diane","SPEAKER_02":"Diane"}}',
            0.1,
            4,
            2,
        )

    monkeypatch.setattr("voxis.speaker_map.call_json_schema_llm", fake_call_json_schema_llm)
    turns = [
        Turn("SPEAKER_01", 0.0, 20.0, "first speaker"),
        Turn("SPEAKER_02", 20.0, 22.0, "second speaker"),
    ]

    result = build_speaker_map(turns, context="", config=make_config())

    assert result.status == "succeeded_with_issues"
    assert result.mapping == {
        "SPEAKER_01": "Unknown (SPEAKER_01)",
        "SPEAKER_02": "Unknown (SPEAKER_02)",
    }
    assert [issue["code"] for issue in result.issues] == ["UNSUPPORTED_NAME", "UNSUPPORTED_NAME"]
    assert {issue["severity"] for issue in result.issues} == {"warning"}


def test_unknown_value_with_raw_speaker_label_is_rejected(monkeypatch: MonkeyPatch) -> None:
    def fake_call_json_schema_llm(
        prompt: str,
        *,
        config: VoxisConfig,
        model: str,
        schema_name: str,
        schema: dict[str, object],
        max_tokens: int = 2048,
        extra_body: dict[str, object] | None = None,
    ) -> LLMResponse:
        return LLMResponse(
            '{"mapping":{"SPEAKER_00":"Alice","SPEAKER_01":"unmapped (SPEAKER_01)"}}',
            0.1,
            4,
            2,
        )

    monkeypatch.setattr("voxis.speaker_map.call_json_schema_llm", fake_call_json_schema_llm)
    turns = [
        Turn("SPEAKER_00", 0.0, 1.0, "hello"),
        Turn("SPEAKER_01", 1.0, 2.0, "world"),
    ]

    result = build_speaker_map(turns, context="Host: Alice.", config=make_config())
    mapped = apply_speaker_map_to_turns(turns, result.mapping)

    assert result.status == "failed"
    assert result.mapping == {"SPEAKER_00": "Alice"}
    assert [issue["code"] for issue in result.issues] == ["INVALID_UNKNOWN_VALUE"]
    assert [(turn.speaker, turn.text) for turn in mapped] == [
        ("Alice", "hello"),
        ("SPEAKER_01", "world"),
    ]


def test_null_unknown_value_is_allowed(monkeypatch: MonkeyPatch) -> None:
    def fake_call_json_schema_llm(
        prompt: str,
        *,
        config: VoxisConfig,
        model: str,
        schema_name: str,
        schema: dict[str, object],
        max_tokens: int = 2048,
        extra_body: dict[str, object] | None = None,
    ) -> LLMResponse:
        return LLMResponse(
            '{"mapping":{"SPEAKER_00":null}}',
            0.1,
            4,
            2,
        )

    monkeypatch.setattr("voxis.speaker_map.call_json_schema_llm", fake_call_json_schema_llm)

    result = build_speaker_map([Turn("SPEAKER_00", 0.0, 1.0, "hello")], context="Host: Alice.", config=make_config())

    assert result.status == "succeeded"
    assert result.mapping == {"SPEAKER_00": "Unknown (SPEAKER_00)"}
    assert result.issues == []


def test_generic_unknown_placeholder_value_is_rejected(monkeypatch: MonkeyPatch) -> None:
    def fake_call_json_schema_llm(
        prompt: str,
        *,
        config: VoxisConfig,
        model: str,
        schema_name: str,
        schema: dict[str, object],
        max_tokens: int = 2048,
        extra_body: dict[str, object] | None = None,
    ) -> LLMResponse:
        return LLMResponse(
            '{"mapping":{"SPEAKER_00":"unknown speaker"}}',
            0.1,
            4,
            2,
        )

    monkeypatch.setattr("voxis.speaker_map.call_json_schema_llm", fake_call_json_schema_llm)

    result = build_speaker_map([Turn("SPEAKER_00", 0.0, 1.0, "hello")], context="Host: Alice.", config=make_config())

    assert result.status == "failed"
    assert result.mapping == {}
    assert [issue["code"] for issue in result.issues] == ["INVALID_UNKNOWN_VALUE"]


def test_unsupported_speaker_name_falls_back_to_unknown(monkeypatch: MonkeyPatch) -> None:
    def fake_call_json_schema_llm(
        prompt: str,
        *,
        config: VoxisConfig,
        model: str,
        schema_name: str,
        schema: dict[str, object],
        max_tokens: int = 2048,
        extra_body: dict[str, object] | None = None,
    ) -> LLMResponse:
        return LLMResponse(
            '{"mapping":{"SPEAKER_00":"未知"}}',
            0.1,
            4,
            2,
        )

    monkeypatch.setattr("voxis.speaker_map.call_json_schema_llm", fake_call_json_schema_llm)

    result = build_speaker_map([Turn("SPEAKER_00", 0.0, 1.0, "hello")], context="Host: Alice.", config=make_config())

    assert result.status == "succeeded_with_issues"
    assert result.mapping == {"SPEAKER_00": "Unknown (SPEAKER_00)"}
    assert [issue["code"] for issue in result.issues] == ["UNSUPPORTED_NAME"]
    assert {issue["severity"] for issue in result.issues} == {"warning"}


def test_legacy_unknown_sentinel_is_rejected(monkeypatch: MonkeyPatch) -> None:
    def fake_call_json_schema_llm(
        prompt: str,
        *,
        config: VoxisConfig,
        model: str,
        schema_name: str,
        schema: dict[str, object],
        max_tokens: int = 2048,
        extra_body: dict[str, object] | None = None,
    ) -> LLMResponse:
        return LLMResponse(
            '{"mapping":{"SPEAKER_00":"Unknown (SPEAKER_00)"}}',
            0.1,
            4,
            2,
        )

    monkeypatch.setattr("voxis.speaker_map.call_json_schema_llm", fake_call_json_schema_llm)

    result = build_speaker_map([Turn("SPEAKER_00", 0.0, 1.0, "hello")], context="Host: Alice.", config=make_config())

    assert result.status == "failed"
    assert result.mapping == {}
    assert [issue["code"] for issue in result.issues] == ["INVALID_UNKNOWN_VALUE"]


def make_config(api_key: str | None = "key", *, llm_disable_thinking: bool = True) -> VoxisConfig:
    return VoxisConfig(
        llm_base_url="https://api.example/v1",
        llm_api_key=api_key,
        polish_model="model",
        speaker_map_model="speaker-map-model",
        llm_disable_thinking=llm_disable_thinking,
    )
