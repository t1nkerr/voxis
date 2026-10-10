from __future__ import annotations

from dataclasses import replace

from voxis.diarization_policy import (
    AssignedWord,
    apply_sortformer_policy,
    assign_words_by_largest_overlap,
    turns_from_assigned_words,
)
from voxis.diarization_types import DiarizeSpan


def test_assign_words_by_largest_overlap_uses_full_word_interval() -> None:
    words = [{"text": "boundary", "start": 0.0, "end": 1.0}]
    spans = [
        DiarizeSpan(start=0.0, end=0.49, speaker="SPEAKER_00"),
        DiarizeSpan(start=0.49, end=0.52, speaker="SPEAKER_01"),
    ]

    assigned = assign_words_by_largest_overlap(words, spans)

    assert assigned[0].speaker == "SPEAKER_00"
    assert assigned[0].coverage_seconds == 0.49
    assert {speaker: round(value, 2) for speaker, value in assigned[0].overlaps.items()} == {
        "SPEAKER_00": 0.49,
        "SPEAKER_01": 0.03,
    }


def test_assign_words_by_largest_overlap_keeps_zero_length_words() -> None:
    # Aligner output from a real run: "40" collapsed to a zero-length timestamp.
    words = [
        {"text": "lost", "start": 4002.0, "end": 4002.48},
        {"text": "about", "start": 4002.48, "end": 4002.72},
        {"text": "40", "start": 4002.72, "end": 4002.72},
        {"text": "of", "start": 4003.12, "end": 4003.28},
    ]
    spans = [
        DiarizeSpan(start=3990.0, end=4002.6, speaker="SPEAKER_00"),
        DiarizeSpan(start=4002.6, end=4010.0, speaker="SPEAKER_01"),
    ]

    assigned = assign_words_by_largest_overlap(words, spans)

    assert [word.text for word in assigned] == ["lost", "about", "40", "of"]
    assert assigned[2].speaker == "SPEAKER_01"
    assert assigned[2].assignment_reason == "nearest_span_zero_length"
    assert assigned[2].coverage_ratio == 0.0


def test_assign_words_by_largest_overlap_skips_inverted_words() -> None:
    words = [{"text": "bad", "start": 2.0, "end": 1.0}]

    assert assign_words_by_largest_overlap(words, [DiarizeSpan(start=0.0, end=3.0, speaker="SPEAKER_00")]) == []


def test_sortformer_policy_repairs_short_unknown_between_same_named_speaker() -> None:
    words = [
        _word("This", 0.0, 0.3, "SPEAKER_00", {"SPEAKER_00": 0.3}),
        _word("lab", 0.3, 0.6, "SPEAKER_02", {"SPEAKER_02": 0.3}),
        _word("works", 0.6, 0.9, "SPEAKER_00", {"SPEAKER_00": 0.3}),
    ]

    result = apply_sortformer_policy(
        words,
        speaker_mapping={"SPEAKER_00": "Jensen", "SPEAKER_02": "Unknown (SPEAKER_02)"},
        language="en",
    )

    assert [word.speaker for word in result.words] == ["SPEAKER_00", "SPEAKER_00", "SPEAKER_00"]
    assert result.audit["unknown_boundary_repair_count"] == 1
    assert result.audit["boundary_word_guard_move_count"] == 0


def test_sortformer_policy_preserves_short_unknown_backchannel() -> None:
    words = [
        _word("This", 0.0, 0.3, "SPEAKER_00", {"SPEAKER_00": 0.3}),
        _word("yeah", 0.3, 0.55, "SPEAKER_02", {"SPEAKER_02": 0.25}),
        _word("works", 0.55, 0.9, "SPEAKER_00", {"SPEAKER_00": 0.35}),
    ]

    result = apply_sortformer_policy(
        words,
        speaker_mapping={"SPEAKER_00": "Jensen", "SPEAKER_02": "Unknown (SPEAKER_02)"},
        language="en",
    )

    assert [word.speaker for word in result.words] == ["SPEAKER_00", "SPEAKER_02", "SPEAKER_00"]
    assert result.audit["unknown_boundary_repair_count"] == 0


def test_sortformer_policy_repairs_singleton_raw_cluster_without_speaker_map() -> None:
    words = [
        _word("This", 0.0, 0.3, "SPEAKER_00", {"SPEAKER_00": 0.3}),
        _word("lab", 0.3, 0.6, "SPEAKER_02", {"SPEAKER_02": 0.3}),
        _word("works", 0.6, 0.9, "SPEAKER_00", {"SPEAKER_00": 0.3}),
    ]

    result = apply_sortformer_policy(words, speaker_mapping={}, language="en")

    assert [word.speaker for word in result.words] == ["SPEAKER_00", "SPEAKER_00", "SPEAKER_00"]
    assert result.audit["unknown_boundary_repair_count"] == 1


def test_sortformer_policy_does_not_absorb_recurring_raw_speaker_without_speaker_map() -> None:
    words = [
        _word("A1", 0.0, 0.3, "SPEAKER_00", {"SPEAKER_00": 0.3}),
        _word("B1", 0.3, 0.6, "SPEAKER_01", {"SPEAKER_01": 0.3}),
        _word("A2", 0.6, 0.9, "SPEAKER_00", {"SPEAKER_00": 0.3}),
        _word("B2", 0.9, 1.2, "SPEAKER_01", {"SPEAKER_01": 0.3}),
        _word("A3", 1.2, 1.5, "SPEAKER_00", {"SPEAKER_00": 0.3}),
    ]

    result = apply_sortformer_policy(words, speaker_mapping={}, language="en")

    assert [word.speaker for word in result.words] == [
        "SPEAKER_00",
        "SPEAKER_01",
        "SPEAKER_00",
        "SPEAKER_01",
        "SPEAKER_00",
    ]
    assert result.audit["unknown_boundary_repair_count"] == 0


def test_english_boundary_guard_can_move_weak_edge_word_to_stable_neighbor() -> None:
    words = [
        _word("We", 0.0, 0.25, "SPEAKER_00", {"SPEAKER_00": 0.25}),
        _word("should", 0.25, 0.55, "SPEAKER_00", {"SPEAKER_00": 0.3}),
        _word("go", 0.55, 0.8, "SPEAKER_01", {"SPEAKER_00": 0.12, "SPEAKER_01": 0.13}),
    ]

    result = apply_sortformer_policy(words, speaker_mapping={"SPEAKER_00": "A", "SPEAKER_01": "B"}, language="en")

    assert [word.speaker for word in result.words] == ["SPEAKER_00", "SPEAKER_00", "SPEAKER_00"]
    assert result.audit["boundary_word_guard_moves"][0]["reason"] == "near_tie_stable_neighbor"


def test_boundary_guard_does_not_move_singleton_raw_cluster_without_speaker_map() -> None:
    words = [
        _word("We", 0.0, 0.25, "SPEAKER_00", {"SPEAKER_00": 0.25}),
        _word("should", 0.25, 0.55, "SPEAKER_00", {"SPEAKER_00": 0.3}),
        _word("go", 0.55, 0.8, "SPEAKER_01", {"SPEAKER_00": 0.12, "SPEAKER_01": 0.13}),
    ]

    result = apply_sortformer_policy(words, speaker_mapping={}, language="en")

    assert [word.speaker for word in result.words] == ["SPEAKER_00", "SPEAKER_00", "SPEAKER_01"]
    assert result.audit["boundary_word_guard_move_count"] == 0


def test_boundary_guard_still_moves_recurring_raw_speakers_without_speaker_map() -> None:
    words = [
        _word("We", 0.0, 0.25, "SPEAKER_00", {"SPEAKER_00": 0.25}),
        _word("should", 0.25, 0.55, "SPEAKER_00", {"SPEAKER_00": 0.3}),
        _word("go", 0.55, 0.8, "SPEAKER_01", {"SPEAKER_00": 0.12, "SPEAKER_01": 0.13}),
        _word("again", 0.8, 1.1, "SPEAKER_00", {"SPEAKER_00": 0.3}),
        _word("later", 1.1, 1.4, "SPEAKER_01", {"SPEAKER_01": 0.3}),
    ]

    result = apply_sortformer_policy(words, speaker_mapping={}, language="en")

    assert [word.speaker for word in result.words] == [
        "SPEAKER_00",
        "SPEAKER_00",
        "SPEAKER_00",
        "SPEAKER_00",
        "SPEAKER_01",
    ]
    assert result.audit["boundary_word_guard_moves"][0]["reason"] == "near_tie_stable_neighbor"


def test_chinese_boundary_policy_rejects_context_only_zero_overlap_move() -> None:
    words = [
        _word("我", 0.0, 0.08, "SPEAKER_00", {"SPEAKER_00": 0.08}),
        _word("们", 0.08, 0.16, "SPEAKER_00", {"SPEAKER_00": 0.08}),
        _word("的", 0.16, 0.24, "SPEAKER_01", {}),
    ]

    result = apply_sortformer_policy(words, speaker_mapping={"SPEAKER_00": "Koji", "SPEAKER_01": "新璐"}, language="zh")

    assert [word.speaker for word in result.words] == ["SPEAKER_00", "SPEAKER_00", "SPEAKER_01"]
    assert result.audit["boundary_word_guard_move_count"] == 0
    assert result.audit["language_policy"] == "zh_boundary_repair_profile_v1"
    assert result.audit["parameters"]["boundary_guard_mode"] == "strict_overlap_margin"


def test_turns_from_assigned_words_uses_asr_spacing() -> None:
    words = [
        replace(_word("value.", 0.0, 0.2, "SPEAKER_00", {"SPEAKER_00": 0.2}), space_before=None),
        replace(_word("And", 0.2, 0.4, "SPEAKER_00", {"SPEAKER_00": 0.2}), space_before=True),
        replace(_word("60-", 0.4, 0.6, "SPEAKER_00", {"SPEAKER_00": 0.2}), space_before=True),
        replace(_word("70%", 0.6, 0.8, "SPEAKER_00", {"SPEAKER_00": 0.2}), space_before=False),
    ]

    assert turns_from_assigned_words(words)[0].text == "value. And 60-70%"


def test_sortformer_policy_preserves_punctuated_chinese_backchannel() -> None:
    words = [
        _word("我", 0.0, 0.3, "SPEAKER_00", {"SPEAKER_00": 0.3}),
        _word("嗯，", 0.3, 0.55, "SPEAKER_02", {"SPEAKER_02": 0.25}),
        _word("们", 0.55, 0.9, "SPEAKER_00", {"SPEAKER_00": 0.35}),
    ]

    result = apply_sortformer_policy(
        words,
        speaker_mapping={"SPEAKER_00": "Koji", "SPEAKER_02": "Unknown (SPEAKER_02)"},
        language="zh",
    )

    assert [word.speaker for word in result.words] == ["SPEAKER_00", "SPEAKER_02", "SPEAKER_00"]
    assert result.audit["unknown_boundary_repair_count"] == 0


def test_turns_from_assigned_words_joins_chinese_without_spaces() -> None:
    words = [
        _word("我", 0.0, 0.1, "SPEAKER_00", {"SPEAKER_00": 0.1}),
        _word("们", 0.1, 0.2, "SPEAKER_00", {"SPEAKER_00": 0.1}),
        _word("go", 0.2, 0.4, "SPEAKER_01", {"SPEAKER_01": 0.2}),
        _word("now", 0.4, 0.7, "SPEAKER_01", {"SPEAKER_01": 0.3}),
    ]

    turns = turns_from_assigned_words(words)

    assert turns[0].text == "我们"
    assert turns[1].text == "go now"


def _word(text: str, start: float, end: float, speaker: str, overlaps: dict[str, float]) -> AssignedWord:
    coverage = overlaps.get(speaker, 0.0)
    return AssignedWord(
        text=text,
        start=start,
        end=end,
        speaker=speaker,
        original_speaker=speaker,
        coverage_seconds=coverage,
        coverage_ratio=coverage / max(0.001, end - start),
        assignment_reason="test",
        overlaps=overlaps,
    )
