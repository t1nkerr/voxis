from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any

from voxis.audio import json_time
from voxis.diarization_types import DiarizeSpan, Turn

WORD_ASSIGNMENT_METHOD = "largest_overlap_v1"
SORTFORMER_POLICY_ID = "sortformer_language_boundary_policy_v1"

EN_BACKCHANNELS = frozenset(
    {"yeah", "yep", "yes", "okay", "ok", "right", "mm", "mhm", "mmhmm", "uhhuh", "uh-huh", "no"}
)
ZH_BACKCHANNELS = frozenset({"嗯", "嗯嗯", "呃", "啊", "哦", "对", "对对", "好", "是", "诶", "哎", "哈哈"})


@dataclass(frozen=True)
class AssignedWord:
    text: str
    start: float
    end: float
    speaker: str
    original_speaker: str
    coverage_seconds: float
    coverage_ratio: float
    assignment_reason: str
    overlaps: dict[str, float]


@dataclass(frozen=True)
class SortformerPolicyResult:
    words: list[AssignedWord]
    audit: dict[str, Any]


def assign_words_by_largest_overlap(words: list[dict[str, Any]], spans: list[DiarizeSpan]) -> list[AssignedWord]:
    if not words or not spans:
        return []
    sorted_spans = sorted(spans, key=lambda span: (span.start, span.end, span.speaker))
    assigned: list[AssignedWord] = []
    for row in words:
        text = str(row.get("text", "")).strip()
        start = float(row["start"])
        end = float(row["end"])
        if not text or end <= start:
            continue
        overlaps = _speaker_overlaps(start, end, sorted_spans)
        if overlaps:
            speaker, coverage_seconds = max(overlaps.items(), key=lambda item: (item[1], item[0]))
            assignment_reason = "largest_overlap"
        else:
            nearest = _nearest_span(start, end, sorted_spans)
            speaker = nearest.speaker
            coverage_seconds = 0.0
            assignment_reason = "nearest_span_no_overlap"
        assigned.append(
            AssignedWord(
                text=text,
                start=start,
                end=end,
                speaker=speaker,
                original_speaker=speaker,
                coverage_seconds=coverage_seconds,
                coverage_ratio=coverage_seconds / max(0.001, end - start),
                assignment_reason=assignment_reason,
                overlaps=overlaps,
            )
        )
    return assigned


def turns_from_assigned_words(words: list[AssignedWord]) -> list[Turn]:
    turns: list[Turn] = []
    current: Turn | None = None
    for word in words:
        if current is None or word.speaker != current.speaker:
            if current is not None:
                turns.append(current)
            current = Turn(speaker=word.speaker, start=word.start, end=word.end, text=word.text)
        else:
            separator = " " if _needs_space(current.text, word.text) else ""
            current = Turn(
                speaker=current.speaker,
                start=current.start,
                end=word.end,
                text=current.text + separator + word.text,
            )
    if current is not None:
        turns.append(current)
    return turns


def apply_sortformer_policy(
    words: list[AssignedWord],
    *,
    speaker_mapping: dict[str, str],
    language: str,
) -> SortformerPolicyResult:
    language_policy = _language_policy(language)
    repaired_words, unknown_repairs = _repair_unknown_boundary_steals(
        words,
        speaker_mapping=speaker_mapping,
        language_policy=language_policy,
    )
    guarded_words, boundary_moves = _apply_boundary_word_guard(
        repaired_words,
        speaker_mapping=speaker_mapping,
        language_policy=language_policy,
    )
    audit = {
        "policy_id": SORTFORMER_POLICY_ID,
        "word_assignment_method": WORD_ASSIGNMENT_METHOD,
        "language_policy": language_policy["policy_id"],
        "language": language,
        "changed_word_count": sum(
            1 for before, after in zip(words, guarded_words, strict=True) if before.speaker != after.speaker
        ),
        "unknown_boundary_repair_count": len(unknown_repairs),
        "boundary_word_guard_move_count": len(boundary_moves),
        "unknown_boundary_repairs": unknown_repairs,
        "boundary_word_guard_moves": boundary_moves,
        "parameters": language_policy,
    }
    return SortformerPolicyResult(words=guarded_words, audit=audit)


def assigned_words_payload(words: list[AssignedWord]) -> list[dict[str, Any]]:
    return [
        {
            "text": word.text,
            "start": json_time(word.start),
            "end": json_time(word.end),
            "speaker": word.speaker,
            "original_speaker": word.original_speaker,
            "coverage_seconds": json_time(word.coverage_seconds),
            "coverage_ratio": round(word.coverage_ratio, 4),
            "assignment_reason": word.assignment_reason,
            "overlaps": {speaker: json_time(value) for speaker, value in sorted(word.overlaps.items())},
        }
        for word in words
    ]


def _repair_unknown_boundary_steals(
    words: list[AssignedWord],
    *,
    speaker_mapping: dict[str, str],
    language_policy: dict[str, Any],
) -> tuple[list[AssignedWord], list[dict[str, Any]]]:
    speakers = [word.speaker for word in words]
    repairs: list[dict[str, Any]] = []
    changed = True
    while changed:
        changed = False
        current_runs = _runs(speakers)
        speaker_run_counts = _speaker_run_counts(current_runs)
        for run_index, (start, end, speaker) in enumerate(current_runs):
            if not _is_unknown_repair_candidate(
                speaker,
                speaker_mapping,
                run_count=speaker_run_counts.get(speaker, 0),
            ):
                continue
            duration = words[end - 1].end - words[start].start
            if duration > language_policy["unknown_repair_max_seconds"]:
                continue
            if end - start > language_policy["unknown_repair_max_words"]:
                continue
            if all(_is_backchannel(words[index].text, language_policy) for index in range(start, end)):
                continue
            if run_index == 0 or run_index + 1 >= len(current_runs):
                continue
            previous_run = current_runs[run_index - 1]
            next_run = current_runs[run_index + 1]
            if previous_run[2] != next_run[2]:
                continue
            destination = previous_run[2]
            if _is_unknown_repair_destination(destination, speaker_mapping):
                continue
            for index in range(start, end):
                speakers[index] = destination
            repairs.append(
                {
                    "start_index": start,
                    "end_index": end,
                    "from_speaker": speaker,
                    "to_speaker": destination,
                    "text": _join_words(words[index].text for index in range(start, end)),
                    "start": json_time(words[start].start),
                    "end": json_time(words[end - 1].end),
                    "reason": "unknown_between_same_named_context",
                }
            )
            changed = True
            break
    return [
        _with_speaker(word, speakers[index], "unknown_boundary_repair") for index, word in enumerate(words)
    ], repairs


def _apply_boundary_word_guard(
    words: list[AssignedWord],
    *,
    speaker_mapping: dict[str, str],
    language_policy: dict[str, Any],
) -> tuple[list[AssignedWord], list[dict[str, Any]]]:
    speakers = [word.speaker for word in words]
    moves: list[dict[str, Any]] = []
    moved_indices: set[int] = set()
    current_runs = _runs(speakers)
    speaker_run_counts = _speaker_run_counts(current_runs)
    for run_index in range(len(current_runs) - 1):
        left_run = current_runs[run_index]
        right_run = current_runs[run_index + 1]
        candidates = (
            ("left_edge", left_run[1] - 1, left_run, right_run),
            ("right_edge", right_run[0], right_run, left_run),
        )
        for side, word_index, current_run, alternate_run in candidates:
            if word_index in moved_indices:
                continue
            word = words[word_index]
            if _is_backchannel(word.text, language_policy):
                continue
            boundary = words[left_run[1] - 1].end if side == "left_edge" else words[right_run[0]].start
            distance = min(abs(word.start - boundary), abs(word.end - boundary))
            if distance > language_policy["edge_word_window_seconds"]:
                continue
            if not _is_stable_run(words, alternate_run, language_policy):
                continue
            current_speaker = current_run[2]
            alternate_speaker = alternate_run[2]
            if _is_boundary_guard_protected_speaker(
                current_speaker,
                speaker_mapping,
                run_count=speaker_run_counts.get(current_speaker, 0),
            ):
                continue
            if _is_boundary_guard_protected_speaker(
                alternate_speaker,
                speaker_mapping,
                run_count=speaker_run_counts.get(alternate_speaker, 0),
            ):
                continue
            current_overlap = word.overlaps.get(current_speaker, 0.0)
            alternate_overlap = word.overlaps.get(alternate_speaker, 0.0)
            margin = current_overlap - alternate_overlap
            alternate_stronger = alternate_overlap > current_overlap + language_policy["alternate_must_win_by_seconds"]
            near_tie = (
                language_policy["allow_near_tie_stable_neighbor"]
                and abs(margin) <= language_policy["near_zero_margin_seconds"]
            )
            if language_policy["require_positive_alternate_overlap"] and alternate_overlap <= 0.0:
                continue
            if not (alternate_stronger or near_tie):
                continue
            speakers[word_index] = alternate_speaker
            moved_indices.add(word_index)
            moves.append(
                {
                    "word_index": word_index,
                    "text": word.text,
                    "start": json_time(word.start),
                    "end": json_time(word.end),
                    "from_speaker": current_speaker,
                    "from_name": _visible_name(current_speaker, speaker_mapping),
                    "to_speaker": alternate_speaker,
                    "to_name": _visible_name(alternate_speaker, speaker_mapping),
                    "current_overlap_seconds": json_time(current_overlap),
                    "alternate_overlap_seconds": json_time(alternate_overlap),
                    "margin_seconds": round(margin, 4),
                    "side": side,
                    "reason": "alternate_stronger" if alternate_stronger else "near_tie_stable_neighbor",
                }
            )
    return [_with_speaker(word, speakers[index], "boundary_word_guard") for index, word in enumerate(words)], moves


def _language_policy(language: str) -> dict[str, Any]:
    if language == "zh":
        return {
            "policy_id": "zh_boundary_repair_profile_v1",
            "boundary_guard_mode": "strict_overlap_margin",
            "edge_word_window_seconds": 0.7,
            "alternate_must_win_by_seconds": 0.04,
            "near_zero_margin_seconds": 0.0,
            "allow_near_tie_stable_neighbor": False,
            "require_positive_alternate_overlap": True,
            "stable_destination_min_words": 2,
            "stable_destination_min_seconds": 0.8,
            "unknown_repair_max_seconds": 0.9,
            "unknown_repair_max_words": 6,
            "backchannels": sorted(EN_BACKCHANNELS | ZH_BACKCHANNELS),
        }
    return {
        "policy_id": "en_boundary_repair_profile_v1",
        "boundary_guard_mode": "permissive_near_tie",
        "edge_word_window_seconds": 0.7,
        "alternate_must_win_by_seconds": 0.02,
        "near_zero_margin_seconds": 0.04,
        "allow_near_tie_stable_neighbor": True,
        "require_positive_alternate_overlap": False,
        "stable_destination_min_words": 2,
        "stable_destination_min_seconds": 0.8,
        "unknown_repair_max_seconds": 0.9,
        "unknown_repair_max_words": 3,
        "backchannels": sorted(EN_BACKCHANNELS),
    }


def _speaker_overlaps(start: float, end: float, spans: list[DiarizeSpan]) -> dict[str, float]:
    overlaps: dict[str, float] = {}
    for span in spans:
        overlap = _overlap(start, end, span.start, span.end)
        if overlap > 0.0:
            overlaps[span.speaker] = overlaps.get(span.speaker, 0.0) + overlap
    return overlaps


def _nearest_span(start: float, end: float, spans: list[DiarizeSpan]) -> DiarizeSpan:
    midpoint = (start + end) / 2.0
    return min(
        spans,
        key=lambda span: (
            0.0 if span.start <= midpoint <= span.end else min(abs(midpoint - span.start), abs(midpoint - span.end)),
            span.start,
        ),
    )


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _runs(speakers: list[str]) -> list[tuple[int, int, str]]:
    if not speakers:
        return []
    runs: list[tuple[int, int, str]] = []
    start = 0
    for index in range(1, len(speakers) + 1):
        if index == len(speakers) or speakers[index] != speakers[start]:
            runs.append((start, index, speakers[start]))
            start = index
    return runs


def _visible_name(speaker: str, speaker_mapping: dict[str, str]) -> str:
    return speaker_mapping.get(speaker, speaker)


def _speaker_run_counts(runs: list[tuple[int, int, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for _start, _end, speaker in runs:
        counts[speaker] = counts.get(speaker, 0) + 1
    return counts


def _is_unknown_repair_candidate(speaker: str, speaker_mapping: dict[str, str], *, run_count: int) -> bool:
    if not speaker_mapping:
        return run_count == 1
    return _visible_name(speaker, speaker_mapping).startswith("Unknown")


def _is_unknown_repair_destination(speaker: str, speaker_mapping: dict[str, str]) -> bool:
    if not speaker_mapping:
        return False
    return _visible_name(speaker, speaker_mapping).startswith("Unknown")


def _is_boundary_guard_protected_speaker(speaker: str, speaker_mapping: dict[str, str], *, run_count: int) -> bool:
    if not speaker_mapping:
        return run_count == 1
    return _visible_name(speaker, speaker_mapping).startswith("Unknown")


def _is_stable_run(words: list[AssignedWord], run: tuple[int, int, str], language_policy: dict[str, Any]) -> bool:
    start, end, _speaker = run
    duration = words[end - 1].end - words[start].start
    min_words = int(language_policy["stable_destination_min_words"])
    min_seconds = float(language_policy["stable_destination_min_seconds"])
    return end - start >= min_words or duration >= min_seconds


def _is_backchannel(text: str, language_policy: dict[str, Any]) -> bool:
    normalized_ascii = re.sub(r"[^a-z]+", "", text.lower())
    compact = re.sub(r"\s+", "", text.strip())
    backchannels = set(language_policy["backchannels"])
    return normalized_ascii in backchannels or compact in backchannels


def _with_speaker(word: AssignedWord, speaker: str, reason: str) -> AssignedWord:
    if speaker == word.speaker:
        return word
    return replace(word, speaker=speaker, assignment_reason=reason)


def _join_words(words: Any) -> str:
    text = ""
    for word in words:
        separator = " " if _needs_space(text, str(word)) else ""
        text += separator + str(word)
    return text


def _needs_space(left: str, right: str) -> bool:
    return bool(
        left and right and left[-1].isascii() and left[-1].isalnum() and right[0].isascii() and right[0].isalnum()
    )
