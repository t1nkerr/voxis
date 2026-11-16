from __future__ import annotations

import pytest

from voxis.asr_terms import (
    detect_prompt_echo,
    global_terms,
    normalize_terms,
    prompt_echo_tail_start,
    prompt_terms,
    system_prompt_from_terms,
)
from voxis.config import VoxisConfig


def _config(asr_terms: tuple[str, ...] = ()) -> VoxisConfig:
    return VoxisConfig(
        llm_base_url="https://example.com/v1",
        llm_api_key=None,
        polish_model="model",
        speaker_map_model="model",
        asr_terms=asr_terms,
    )


def test_normalize_terms_keeps_specific_terms_and_drops_contained_terms() -> None:
    assert normalize_terms([" Claude Code ", "Claude", "audio", "https://example.com", "CAR-T"]) == (
        "Claude Code",
        "CAR-T",
    )


def test_system_prompt_caps_bare_word_list() -> None:
    config = _config(asr_terms=("Alpha", "Beta"))
    terms = [f"Term{index}" for index in range(35)]
    prompt = system_prompt_from_terms(terms, config=config)

    assert len(prompt_terms(terms, config=config)) == 30
    assert prompt_terms(terms, config=config)[:2] == ("Alpha", "Beta")
    assert prompt is not None
    assert prompt.splitlines() == list(prompt_terms(terms, config=config))


def test_global_terms_default_to_empty() -> None:
    assert global_terms(_config()) == ()


def test_global_terms_come_from_config() -> None:
    config = _config(asr_terms=("Claude Code", "Claude"))
    assert global_terms(config) == ("Claude Code", "Claude")
    assert prompt_terms(["Claude Code", "CAR-T"], config=config)[:3] == (
        "Claude Code",
        "Claude",
        "CAR-T",
    )


def test_detect_prompt_echo_flags_terms_in_prompt_order() -> None:
    result = detect_prompt_echo(
        [
            {"start": 0.0, "end": 1.0, "text": "normal content"},
            {"start": 1.0, "end": 2.0, "text": "Claude Code. CAR-T. LangChain. Manus."},
        ],
        ["Claude Code", "CAR-T", "LangChain", "Manus"],
    )

    assert result.status == "detected"
    assert result.suspect_segments[0].index == 1
    assert result.suspect_segments[0].ordered_term_count == 4
    assert prompt_echo_tail_start(result) == 1


def test_detect_prompt_echo_ignores_ordinary_term_mentions() -> None:
    result = detect_prompt_echo(
        [{"start": 0.0, "end": 1.0, "text": "We used Claude Code to fix one CAR-T reference."}],
        ["Claude Code", "CAR-T", "LangChain", "Manus"],
    )

    assert result.status == "passed"
    assert prompt_echo_tail_start(result) is None


def test_prompt_echo_tail_start_rejects_middle_echo() -> None:
    result = detect_prompt_echo(
        [
            {"start": 0.0, "end": 1.0, "text": "Claude Code. CAR-T. LangChain. Manus."},
            {"start": 1.0, "end": 2.0, "text": "normal content"},
        ],
        ["Claude Code", "CAR-T", "LangChain", "Manus"],
    )

    assert result.status == "detected"
    assert prompt_echo_tail_start(result) is None


def test_prompt_echo_tail_includes_trailing_term_only_segments() -> None:
    terms = ["The Circuit", "Anthropic", "Cloudflare", "AMD", "ARM", "SiTime"]
    result = detect_prompt_echo(
        [
            {"start": 0.0, "end": 1.0, "text": "Thanks for listening, everybody."},
            {"start": 1.0, "end": 2.0, "text": "Anthropic. Cloudflare. AMD. ARM. SiTime."},
            {"start": 2.0, "end": 3.0, "text": "The Circuit."},
        ],
        terms,
    )

    assert [segment.index for segment in result.suspect_segments] == [1]
    assert result.term_only_segment_indexes == (2,)
    assert prompt_echo_tail_start(result) == 1


def test_term_only_segment_after_ordinary_speech_is_not_trimmed() -> None:
    terms = ["The Circuit", "Anthropic", "Cloudflare", "AMD", "ARM", "SiTime"]
    result = detect_prompt_echo(
        [
            {"start": 0.0, "end": 1.0, "text": "Anthropic. Cloudflare. AMD. ARM. SiTime."},
            {"start": 1.0, "end": 2.0, "text": "Welcome back to the show."},
            {"start": 2.0, "end": 3.0, "text": "The Circuit."},
        ],
        terms,
    )

    assert result.term_only_segment_indexes == (2,)
    assert prompt_echo_tail_start(result) is None


def test_term_only_tail_without_any_echo_passes() -> None:
    result = detect_prompt_echo(
        [
            {"start": 0.0, "end": 1.0, "text": "Welcome back to the show."},
            {"start": 1.0, "end": 2.0, "text": "The Circuit."},
        ],
        ["The Circuit", "Anthropic", "Cloudflare", "AMD", "ARM", "SiTime"],
    )

    assert result.status == "passed"
    assert result.term_only_segment_indexes == (1,)
    assert prompt_echo_tail_start(result) is None


def test_term_only_segment_before_trailing_echo_is_kept() -> None:
    result = detect_prompt_echo(
        [
            {"start": 0.0, "end": 1.0, "text": "Real speech."},
            {"start": 1.0, "end": 2.0, "text": "Anthropic."},
            {"start": 2.0, "end": 3.0, "text": "Anthropic. Cloudflare. AMD. ARM. SiTime."},
        ],
        ["The Circuit", "Anthropic", "Cloudflare", "AMD", "ARM", "SiTime"],
    )

    assert result.term_only_segment_indexes == (1,)
    assert prompt_echo_tail_start(result) == 2


def test_term_only_segment_between_echoes_does_not_join_them() -> None:
    result = detect_prompt_echo(
        [
            {"start": 0.0, "end": 1.0, "text": "Real speech."},
            {"start": 1.0, "end": 2.0, "text": "Anthropic. Cloudflare. AMD. ARM. SiTime."},
            {"start": 2.0, "end": 3.0, "text": "The Circuit."},
            {"start": 3.0, "end": 4.0, "text": "Anthropic. Cloudflare. AMD. ARM. SiTime."},
        ],
        ["The Circuit", "Anthropic", "Cloudflare", "AMD", "ARM", "SiTime"],
    )

    assert [segment.index for segment in result.suspect_segments] == [1, 3]
    assert prompt_echo_tail_start(result) is None


def test_term_only_does_not_split_ordinary_words_into_terms() -> None:
    result = detect_prompt_echo(
        [
            {"start": 0.0, "end": 1.0, "text": "Real speech."},
            {"start": 1.0, "end": 2.0, "text": "Anthropic. Cloudflare. AMD. ARM. SiTime."},
            {"start": 2.0, "end": 3.0, "text": "armrest."},
        ],
        ["The Circuit", "Anthropic", "Cloudflare", "AMD", "ARM", "SiTime", "REST"],
    )

    assert result.term_only_segment_indexes == ()
    assert prompt_echo_tail_start(result) is None


@pytest.mark.parametrize("tail", ["The Circuit\u2026", "The Circuit.\u00a0", "The\nCircuit.", "the  circuit!"])
def test_term_only_accepts_punctuation_and_whitespace_variants(tail: str) -> None:
    result = detect_prompt_echo(
        [
            {"start": 0.0, "end": 1.0, "text": "Real speech."},
            {"start": 1.0, "end": 2.0, "text": "Anthropic. Cloudflare. AMD. ARM. SiTime."},
            {"start": 2.0, "end": 3.0, "text": tail},
        ],
        ["The Circuit", "Anthropic", "Cloudflare", "AMD", "ARM", "SiTime"],
    )

    assert result.term_only_segment_indexes == (2,)
    assert prompt_echo_tail_start(result) == 1


def test_term_only_detection_needs_every_word_to_be_a_term() -> None:
    terms = ["The Circuit", "Anthropic", "Cloudflare", "AMD", "ARM", "SiTime"]
    result = detect_prompt_echo(
        [
            {"start": 0.0, "end": 1.0, "text": "Anthropic. Cloudflare. AMD. ARM. SiTime."},
            {"start": 1.0, "end": 2.0, "text": "This was The Circuit."},
            {"start": 2.0, "end": 3.0, "text": "这是 The Circuit。"},
            {"start": 3.0, "end": 4.0, "text": "AMD, $5."},
        ],
        terms,
    )

    assert result.term_only_segment_indexes == ()
    assert prompt_echo_tail_start(result) is None
