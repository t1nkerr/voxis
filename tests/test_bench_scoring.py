from voxis.bench.scoring import (
    ReferenceSpan,
    ReferenceWord,
    assign_words_by_largest_overlap,
    assign_words_to_voxis_prior,
    build_label_mapping,
    score_diarization_frames,
    score_word_speakers,
    turn_shape_stats,
)


def test_label_mapping_handles_permuted_candidate_speakers() -> None:
    words = [
        ReferenceWord("hello", 0.0, 1.0, "Alice"),
        ReferenceWord("there", 1.0, 2.0, "Alice"),
        ReferenceWord("reply", 2.0, 3.0, "Bob"),
    ]
    spans = [
        ReferenceSpan(0.0, 2.0, "Alice"),
        ReferenceSpan(2.0, 3.0, "Bob"),
    ]
    predicted = ["SPEAKER_01", "SPEAKER_01", "SPEAKER_00"]

    score = score_word_speakers(words, predicted, spans)

    assert score.mapping.mapping == {"SPEAKER_00": "Bob", "SPEAKER_01": "Alice"}
    assert score.main_accuracy == 1.0


def test_overlap_words_are_excluded_from_main_and_scored_as_any_active_speaker() -> None:
    words = [
        ReferenceWord("solo", 0.0, 1.0, "Alice"),
        ReferenceWord("overlap", 1.0, 2.0, "Alice"),
        ReferenceWord("back", 2.0, 3.0, "Bob"),
    ]
    spans = [
        ReferenceSpan(0.0, 2.0, "Alice"),
        ReferenceSpan(1.0, 3.0, "Bob"),
    ]
    predicted = ["X", "Y", "Y"]

    score = score_word_speakers(words, predicted, spans)

    assert score.mapping.mapping == {"X": "Alice", "Y": "Bob"}
    assert score.scored_main_words == 2
    assert score.main_correct_words == 2
    assert score.overlap_words == 1
    assert score.overlap_correct_words == 1


def test_overlap_word_prediction_must_match_one_active_speaker_after_mapping() -> None:
    words = [
        ReferenceWord("solo", 0.0, 1.0, "Alice"),
        ReferenceWord("other", 2.0, 3.0, "Carol"),
        ReferenceWord("overlap", 1.0, 2.0, "Alice"),
    ]
    spans = [
        ReferenceSpan(0.0, 2.0, "Alice"),
        ReferenceSpan(1.0, 2.0, "Bob"),
        ReferenceSpan(2.0, 3.0, "Carol"),
    ]
    predicted = ["X", "Z", "Z"]

    score = score_word_speakers(words, predicted, spans)

    assert score.mapping.mapping == {"X": "Alice", "Z": "Carol"}
    assert score.overlap_words == 1
    assert score.overlap_correct_words == 0


def test_build_label_mapping_leaves_zero_evidence_candidate_unmapped() -> None:
    words = [ReferenceWord("a", 0.0, 1.0, "Alice")]
    spans = [ReferenceSpan(0.0, 1.0, "Alice")]

    mapping = build_label_mapping(words, [None], spans)

    assert mapping.mapping == {}
    assert mapping.reference_labels == ["Alice"]
    assert mapping.candidate_labels == []


def test_score_diarization_frames_handles_permuted_labels() -> None:
    reference = [
        ReferenceSpan(0.0, 1.0, "Alice"),
        ReferenceSpan(1.0, 2.0, "Bob"),
    ]
    candidate = [
        ReferenceSpan(0.0, 1.0, "SPEAKER_01"),
        ReferenceSpan(1.0, 2.0, "SPEAKER_00"),
    ]

    score = score_diarization_frames(reference, candidate, frame_step_seconds=0.5, include_overlap=True)

    assert score.mapping.mapping == {"SPEAKER_00": "Bob", "SPEAKER_01": "Alice"}
    assert score.der_like == 0.0


def test_score_diarization_frames_can_exclude_overlap_frames() -> None:
    reference = [
        ReferenceSpan(0.0, 2.0, "Alice"),
        ReferenceSpan(1.0, 2.0, "Bob"),
    ]
    candidate = [ReferenceSpan(0.0, 2.0, "SPEAKER_00")]

    include_overlap = score_diarization_frames(reference, candidate, frame_step_seconds=0.5, include_overlap=True)
    exclude_overlap = score_diarization_frames(reference, candidate, frame_step_seconds=0.5, include_overlap=False)

    assert include_overlap.der_like is not None
    assert include_overlap.der_like > 0.0
    assert exclude_overlap.der_like == 0.0


def test_score_diarization_frames_counts_unmapped_hypothesis_as_error() -> None:
    reference = [ReferenceSpan(0.0, 1.0, "Alice")]
    candidate = [
        ReferenceSpan(0.0, 1.0, "SPEAKER_00"),
        ReferenceSpan(0.0, 0.5, "SPEAKER_EXTRA"),
        ReferenceSpan(1.0, 2.0, "SPEAKER_EXTRA"),
    ]

    score = score_diarization_frames(reference, candidate, frame_step_seconds=0.5, include_overlap=True)

    assert score.mapping.mapping == {"SPEAKER_00": "Alice"}
    assert score.confusion_speaker_seconds == 0.5
    assert score.false_alarm_speaker_seconds == 1.0


def test_assign_words_to_voxis_prior_uses_midpoint_then_nearest_span() -> None:
    words = [
        ReferenceWord("first", 0.0, 1.0, "Alice"),
        ReferenceWord("second", 1.0, 2.0, "Bob"),
        ReferenceWord("outside", 3.0, 4.0, "Bob"),
    ]
    spans = [
        ReferenceSpan(0.0, 2.0, "LATER_LONG"),
        ReferenceSpan(0.5, 1.5, "EARLIER_SHORT"),
    ]

    predictions = assign_words_to_voxis_prior(words, spans)

    assert predictions == ["LATER_LONG", "LATER_LONG", "LATER_LONG"]


def test_turn_shape_stats_groups_consecutive_predicted_speakers() -> None:
    words = [
        ReferenceWord("a", 0.0, 0.4, "Alice"),
        ReferenceWord("b", 0.4, 1.0, "Alice"),
        ReferenceWord("c", 1.2, 1.6, "Bob"),
        ReferenceWord("d", 1.6, 2.2, "Bob"),
        ReferenceWord("e", 3.0, 3.4, "Alice"),
    ]

    stats = turn_shape_stats(words, ["S1", "S1", "S2", "S2", None])

    assert stats == {
        "turns": 3,
        "turns_per_minute": 52.941,
        "speaker_switches_per_minute": 35.294,
        "median_duration_seconds": 1.0,
        "under_0_5_seconds": 1,
        "under_1_0_seconds": 1,
    }


def test_assign_words_by_largest_overlap_records_midpoint_debug_and_coverage() -> None:
    words = [
        ReferenceWord("boundary", 0.0, 1.0, "Alice"),
        ReferenceWord("tie", 2.0, 3.0, "Alice"),
    ]
    spans = [
        ReferenceSpan(0.0, 0.49, "A"),
        ReferenceSpan(0.49, 1.0, "B"),
        ReferenceSpan(2.0, 2.5, "A"),
        ReferenceSpan(2.5, 3.0, "B"),
    ]

    assignments = assign_words_by_largest_overlap(words, spans, coverage_threshold=0.5)

    assert assignments[0].assigned_speaker == "B"
    assert assignments[0].midpoint_speaker == "B"
    assert assignments[0].largest_overlap_speaker == "B"
    assert assignments[0].coverage_ratio == 0.51
    assert assignments[1].assigned_speaker == "A"
    assert assignments[1].midpoint_speaker == "A"
    assert assignments[1].coverage_ratio == 0.5
