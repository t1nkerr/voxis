"""Word-speaker scoring for diarization benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import isfinite
from statistics import median
from collections.abc import Sequence
from typing import Any


@dataclass(frozen=True)
class ReferenceSpan:
    start: float
    end: float
    speaker: str


@dataclass(frozen=True)
class ReferenceWord:
    text: str
    start: float
    end: float
    speaker: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def midpoint(self) -> float:
        return (self.start + self.end) / 2.0


@dataclass(frozen=True)
class PredictedWord:
    text: str
    start: float
    end: float
    speaker: str


@dataclass(frozen=True)
class LabelMapping:
    mapping: dict[str, str]
    score_seconds: float
    reference_labels: list[str]
    candidate_labels: list[str]


@dataclass(frozen=True)
class WordSpeakerScore:
    mapping: LabelMapping
    total_reference_words: int
    scored_main_words: int
    main_correct_words: int
    main_accuracy: float | None
    overlap_words: int
    overlap_correct_words: int
    overlap_accuracy: float | None
    missing_prediction_words: int

    def to_json(self) -> dict[str, Any]:
        return {
            "mapping": self.mapping.mapping,
            "mapping_score_seconds": self.mapping.score_seconds,
            "reference_labels": self.mapping.reference_labels,
            "candidate_labels": self.mapping.candidate_labels,
            "total_reference_words": self.total_reference_words,
            "scored_main_words": self.scored_main_words,
            "main_correct_words": self.main_correct_words,
            "main_accuracy": self.main_accuracy,
            "overlap_words": self.overlap_words,
            "overlap_correct_words": self.overlap_correct_words,
            "overlap_accuracy": self.overlap_accuracy,
            "missing_prediction_words": self.missing_prediction_words,
        }


@dataclass(frozen=True)
class FrameDiarizationScore:
    mapping: LabelMapping
    frame_step_seconds: float
    include_overlap: bool
    scored_reference_seconds: float
    missed_speaker_seconds: float
    false_alarm_speaker_seconds: float
    confusion_speaker_seconds: float
    der_like: float | None

    def to_json(self) -> dict[str, Any]:
        return {
            "mapping": self.mapping.mapping,
            "mapping_score_seconds": self.mapping.score_seconds,
            "frame_step_seconds": self.frame_step_seconds,
            "include_overlap": self.include_overlap,
            "scored_reference_seconds": self.scored_reference_seconds,
            "missed_speaker_seconds": self.missed_speaker_seconds,
            "false_alarm_speaker_seconds": self.false_alarm_speaker_seconds,
            "confusion_speaker_seconds": self.confusion_speaker_seconds,
            "der_like": self.der_like,
        }


@dataclass(frozen=True)
class WordAssignment:
    word: ReferenceWord
    midpoint_speaker: str | None
    largest_overlap_speaker: str | None
    assigned_speaker: str
    coverage_ratio: float


def validate_timed_record(start: float, end: float, speaker: str) -> None:
    if not isfinite(start) or not isfinite(end):
        raise ValueError("start/end must be finite")
    if start < 0 or end <= start:
        raise ValueError("start/end must describe a positive non-negative span")
    if not speaker.strip():
        raise ValueError("speaker must be non-empty")


def span_shape_stats(spans: list[ReferenceSpan]) -> dict[str, float | int]:
    durations = [span.end - span.start for span in spans]
    return {
        "segments": len(spans),
        "median_duration_seconds": round(median(durations), 6) if durations else 0.0,
        "under_0_5_seconds": sum(duration < 0.5 for duration in durations),
        "under_1_0_seconds": sum(duration < 1.0 for duration in durations),
    }


def turn_shape_stats(
    words: Sequence[ReferenceWord | PredictedWord], predicted_speakers: Sequence[str | None]
) -> dict[str, float | int]:
    if len(words) != len(predicted_speakers):
        raise ValueError("words and predicted_speakers must have the same length")
    if not words:
        return {
            "turns": 0,
            "turns_per_minute": 0.0,
            "speaker_switches_per_minute": 0.0,
            "median_duration_seconds": 0.0,
            "under_0_5_seconds": 0,
            "under_1_0_seconds": 0,
        }

    turns: list[tuple[str, float, float]] = []
    current_speaker: str | None = None
    current_start = 0.0
    current_end = 0.0
    for word, speaker in zip(words, predicted_speakers, strict=True):
        label = speaker or "UNKNOWN"
        if current_speaker is None or label != current_speaker:
            if current_speaker is not None:
                turns.append((current_speaker, current_start, current_end))
            current_speaker = label
            current_start = word.start
        current_end = word.end
    if current_speaker is not None:
        turns.append((current_speaker, current_start, current_end))

    durations = [end - start for _, start, end in turns]
    total_duration = max(word.end for word in words) - min(word.start for word in words)
    minutes = total_duration / 60.0 if total_duration > 0 else 0.0
    return {
        "turns": len(turns),
        "turns_per_minute": round(len(turns) / minutes, 3) if minutes else 0.0,
        "speaker_switches_per_minute": round(max(0, len(turns) - 1) / minutes, 3) if minutes else 0.0,
        "median_duration_seconds": round(median(durations), 6) if durations else 0.0,
        "under_0_5_seconds": sum(duration < 0.5 for duration in durations),
        "under_1_0_seconds": sum(duration < 1.0 for duration in durations),
    }


def assign_words_by_largest_overlap(
    words: list[ReferenceWord], spans: list[ReferenceSpan], *, coverage_threshold: float
) -> list[WordAssignment]:
    if not 0.0 <= coverage_threshold <= 1.0:
        raise ValueError("coverage_threshold must be between 0 and 1")
    midpoint_speakers = assign_words_to_voxis_prior(words, spans)
    assignments: list[WordAssignment] = []
    for word, midpoint_speaker in zip(words, midpoint_speakers, strict=True):
        overlaps = [(span.speaker, _overlap_seconds(word.start, word.end, span.start, span.end)) for span in spans]
        overlaps = [(speaker, overlap) for speaker, overlap in overlaps if overlap > 0]
        if overlaps:
            max_overlap = max(overlap for _, overlap in overlaps)
            tied_speakers = {speaker for speaker, overlap in overlaps if overlap == max_overlap}
            if midpoint_speaker in tied_speakers:
                largest_speaker = midpoint_speaker
            else:
                largest_speaker = sorted(tied_speakers)[0]
            overlap = max_overlap
            coverage_ratio = overlap / word.duration if word.duration else 0.0
            assigned_speaker = largest_speaker
        else:
            coverage_ratio = 0.0
            assigned_speaker = midpoint_speaker or "UNKNOWN"
        assignments.append(
            WordAssignment(
                word=word,
                midpoint_speaker=midpoint_speaker,
                largest_overlap_speaker=assigned_speaker if overlaps else None,
                assigned_speaker=assigned_speaker,
                coverage_ratio=coverage_ratio,
            )
        )
    return assignments


def _overlap_seconds(start: float, end: float, other_start: float, other_end: float) -> float:
    return max(0.0, min(end, other_end) - max(start, other_start))


def active_reference_speakers(word: ReferenceWord, reference_spans: list[ReferenceSpan]) -> frozenset[str]:
    active = {
        span.speaker
        for span in reference_spans
        if span.start <= word.midpoint < span.end or (word.midpoint == span.end and word.end == span.end)
    }
    if active:
        return frozenset(active)
    return frozenset({word.speaker})


def assign_words_to_spans(words: list[ReferenceWord], spans: list[ReferenceSpan]) -> list[str | None]:
    predictions: list[str | None] = []
    for word in words:
        active = [span for span in spans if span.start <= word.midpoint < span.end]
        if not active:
            predictions.append(None)
            continue
        active.sort(key=lambda span: (span.end - span.start, -span.start), reverse=True)
        predictions.append(active[0].speaker)
    return predictions


def assign_words_to_voxis_prior(words: list[ReferenceWord], spans: list[ReferenceSpan]) -> list[str | None]:
    if not spans:
        return [None for _ in words]
    sorted_spans = sorted(spans, key=lambda span: (span.start, span.end, span.speaker))
    predictions: list[str | None] = []
    for word in words:
        for span in sorted_spans:
            if span.start <= word.midpoint <= span.end:
                predictions.append(span.speaker)
                break
        else:
            nearest = min(
                sorted_spans,
                key=lambda span: min(abs(span.start - word.midpoint), abs(span.end - word.midpoint)),
            )
            predictions.append(nearest.speaker)
    return predictions


def assign_predicted_words(
    reference_words: list[ReferenceWord], predicted_words: list[PredictedWord]
) -> list[str | None]:
    predicted_spans = [ReferenceSpan(start=word.start, end=word.end, speaker=word.speaker) for word in predicted_words]
    return assign_words_to_spans(reference_words, predicted_spans)


def build_label_mapping(
    reference_words: list[ReferenceWord],
    predicted_speakers: Sequence[str | None],
    reference_spans: list[ReferenceSpan],
) -> LabelMapping:
    if len(reference_words) != len(predicted_speakers):
        raise ValueError("reference_words and predicted_speakers must have the same length")

    reference_labels = sorted(
        {speaker for word in reference_words for speaker in active_reference_speakers(word, reference_spans)}
    )
    candidate_labels = sorted({speaker for speaker in predicted_speakers if speaker is not None})
    weights: dict[tuple[str, str], float] = {}
    for word, predicted in zip(reference_words, predicted_speakers, strict=True):
        if predicted is None:
            continue
        active = active_reference_speakers(word, reference_spans)
        if len(active) != 1:
            continue
        expected = next(iter(active))
        weights[(predicted, expected)] = weights.get((predicted, expected), 0.0) + word.duration

    if len(reference_labels) > 16:
        mapping = _greedy_mapping(candidate_labels, reference_labels, weights)
    else:
        mapping = _exact_mapping(candidate_labels, reference_labels, weights)
    score = sum(weights.get((candidate, reference), 0.0) for candidate, reference in mapping.items())
    return LabelMapping(
        mapping=mapping,
        score_seconds=score,
        reference_labels=reference_labels,
        candidate_labels=candidate_labels,
    )


def score_word_speakers(
    reference_words: list[ReferenceWord],
    predicted_speakers: Sequence[str | None],
    reference_spans: list[ReferenceSpan],
    *,
    label_mapping: LabelMapping | None = None,
) -> WordSpeakerScore:
    if len(reference_words) != len(predicted_speakers):
        raise ValueError("reference_words and predicted_speakers must have the same length")

    mapping = label_mapping or build_label_mapping(reference_words, predicted_speakers, reference_spans)
    scored_main = 0
    correct_main = 0
    overlap_words = 0
    overlap_correct = 0
    missing_predictions = 0

    for word, predicted in zip(reference_words, predicted_speakers, strict=True):
        if predicted is None:
            missing_predictions += 1
            mapped_prediction = None
        else:
            mapped_prediction = mapping.mapping.get(predicted)

        active = active_reference_speakers(word, reference_spans)
        if len(active) > 1:
            overlap_words += 1
            if mapped_prediction in active:
                overlap_correct += 1
            continue

        scored_main += 1
        expected = next(iter(active))
        if mapped_prediction == expected:
            correct_main += 1

    return WordSpeakerScore(
        mapping=mapping,
        total_reference_words=len(reference_words),
        scored_main_words=scored_main,
        main_correct_words=correct_main,
        main_accuracy=_ratio(correct_main, scored_main),
        overlap_words=overlap_words,
        overlap_correct_words=overlap_correct,
        overlap_accuracy=_ratio(overlap_correct, overlap_words),
        missing_prediction_words=missing_predictions,
    )


def score_diarization_frames(
    reference_spans: list[ReferenceSpan],
    candidate_spans: list[ReferenceSpan],
    *,
    frame_step_seconds: float = 0.01,
    include_overlap: bool,
) -> FrameDiarizationScore:
    if frame_step_seconds <= 0:
        raise ValueError("frame_step_seconds must be positive")
    mapping = build_frame_label_mapping(reference_spans, candidate_spans, frame_step_seconds=frame_step_seconds)
    start, end = _evaluation_bounds(reference_spans, candidate_spans)
    if end <= start:
        return FrameDiarizationScore(
            mapping=mapping,
            frame_step_seconds=frame_step_seconds,
            include_overlap=include_overlap,
            scored_reference_seconds=0.0,
            missed_speaker_seconds=0.0,
            false_alarm_speaker_seconds=0.0,
            confusion_speaker_seconds=0.0,
            der_like=None,
        )

    missed = 0.0
    false_alarm = 0.0
    confusion = 0.0
    reference_total = 0.0
    frame_start = start
    while frame_start < end:
        frame_end = min(frame_start + frame_step_seconds, end)
        midpoint = (frame_start + frame_end) / 2.0
        frame_duration = frame_end - frame_start
        reference_active = _active_span_speakers(reference_spans, midpoint)
        if not include_overlap and len(reference_active) > 1:
            frame_start = frame_end
            continue
        candidate_active = _active_span_speakers(candidate_spans, midpoint)
        mapped_candidate = {mapping.mapping[speaker] for speaker in candidate_active if speaker in mapping.mapping}
        unmapped_candidate_count = len(candidate_active) - len(mapped_candidate)

        reference_total += len(reference_active) * frame_duration
        missed += len(reference_active - mapped_candidate) * frame_duration

        mapped_extra = mapped_candidate - reference_active
        if reference_active:
            confusion += (len(mapped_extra) + unmapped_candidate_count) * frame_duration
        else:
            false_alarm += (len(mapped_extra) + unmapped_candidate_count) * frame_duration

        frame_start = frame_end

    der_like = None
    if reference_total > 0:
        der_like = (missed + false_alarm + confusion) / reference_total
    return FrameDiarizationScore(
        mapping=mapping,
        frame_step_seconds=frame_step_seconds,
        include_overlap=include_overlap,
        scored_reference_seconds=reference_total,
        missed_speaker_seconds=missed,
        false_alarm_speaker_seconds=false_alarm,
        confusion_speaker_seconds=confusion,
        der_like=der_like,
    )


def build_frame_label_mapping(
    reference_spans: list[ReferenceSpan],
    candidate_spans: list[ReferenceSpan],
    *,
    frame_step_seconds: float = 0.01,
) -> LabelMapping:
    if frame_step_seconds <= 0:
        raise ValueError("frame_step_seconds must be positive")
    reference_labels = sorted({span.speaker for span in reference_spans})
    candidate_labels = sorted({span.speaker for span in candidate_spans})
    weights: dict[tuple[str, str], float] = {}
    start, end = _evaluation_bounds(reference_spans, candidate_spans)
    frame_start = start
    while frame_start < end:
        frame_end = min(frame_start + frame_step_seconds, end)
        midpoint = (frame_start + frame_end) / 2.0
        frame_duration = frame_end - frame_start
        for candidate in _active_span_speakers(candidate_spans, midpoint):
            for reference in _active_span_speakers(reference_spans, midpoint):
                weights[(candidate, reference)] = weights.get((candidate, reference), 0.0) + frame_duration
        frame_start = frame_end

    if len(reference_labels) > 16:
        mapping = _greedy_mapping(candidate_labels, reference_labels, weights)
    else:
        mapping = _exact_mapping(candidate_labels, reference_labels, weights)
    score = sum(weights.get((candidate, reference), 0.0) for candidate, reference in mapping.items())
    return LabelMapping(
        mapping=mapping,
        score_seconds=score,
        reference_labels=reference_labels,
        candidate_labels=candidate_labels,
    )


def _exact_mapping(
    candidate_labels: list[str],
    reference_labels: list[str],
    weights: dict[tuple[str, str], float],
) -> dict[str, str]:
    @lru_cache(maxsize=None)
    def solve(candidate_index: int, used_mask: int) -> tuple[float, tuple[str | None, ...]]:
        if candidate_index >= len(candidate_labels):
            return 0.0, ()

        best_score, best_assignment = solve(candidate_index + 1, used_mask)
        best_assignment = (None, *best_assignment)

        candidate = candidate_labels[candidate_index]
        for reference_index, reference in enumerate(reference_labels):
            bit = 1 << reference_index
            if used_mask & bit:
                continue
            tail_score, tail_assignment = solve(candidate_index + 1, used_mask | bit)
            score = weights.get((candidate, reference), 0.0) + tail_score
            if score > best_score:
                best_score = score
                best_assignment = (reference, *tail_assignment)
        return best_score, best_assignment

    _, assignment = solve(0, 0)
    return {
        candidate: reference
        for candidate, reference in zip(candidate_labels, assignment, strict=True)
        if reference is not None
    }


def _greedy_mapping(
    candidate_labels: list[str],
    reference_labels: list[str],
    weights: dict[tuple[str, str], float],
) -> dict[str, str]:
    pairs = sorted(
        (
            (weights.get((candidate, reference), 0.0), candidate, reference)
            for candidate in candidate_labels
            for reference in reference_labels
        ),
        reverse=True,
    )
    used_candidates: set[str] = set()
    used_references: set[str] = set()
    mapping: dict[str, str] = {}
    for score, candidate, reference in pairs:
        if score <= 0:
            break
        if candidate in used_candidates or reference in used_references:
            continue
        mapping[candidate] = reference
        used_candidates.add(candidate)
        used_references.add(reference)
    return mapping


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _evaluation_bounds(
    reference_spans: list[ReferenceSpan], candidate_spans: list[ReferenceSpan]
) -> tuple[float, float]:
    spans = reference_spans + candidate_spans
    if not spans:
        return 0.0, 0.0
    return min(span.start for span in spans), max(span.end for span in spans)


def _active_span_speakers(spans: list[ReferenceSpan], midpoint: float) -> set[str]:
    return {span.speaker for span in spans if span.start <= midpoint < span.end}
