"""Parsers for benchmark reference and candidate artifacts."""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from voxis.bench.scoring import PredictedWord, ReferenceSpan, ReferenceWord, validate_timed_record

_ANNOTATION_TOKEN_RE = re.compile(r"^(\[[^\]]+\]|<[^>]+>|\{[^}]+\}|\([^)]+\)|[*]+)$")


def load_reference_words(path: Path) -> tuple[list[ReferenceWord], int]:
    raw_words = _load_json_list(path)
    words: list[ReferenceWord] = []
    dropped = 0
    for item in raw_words:
        text = _string_field(item, "text", path)
        if _is_annotation_token(text):
            dropped += 1
            continue
        words.append(_reference_word_from_json(item, path))
    kept, later_dropped = filter_reference_words(words)
    return kept, dropped + later_dropped


def load_predicted_words(path: Path) -> list[PredictedWord]:
    return [_predicted_word_from_json(item, path) for item in _load_json_list(path)]


def load_candidate_spans_json(path: Path) -> list[ReferenceSpan]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("segments"), list):
        payload = payload["segments"]
    if not isinstance(payload, list):
        raise ValueError(f"{path}: expected a segment list or an object with a segments list")

    spans: list[ReferenceSpan] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"{path}: segment row {index} must be an object")
        start = _first_number_field(item, ("startTimeSeconds", "start_time", "startTime", "start"), path)
        end = _first_number_field(item, ("endTimeSeconds", "end_time", "endTime", "end"), path)
        speaker = _first_string_field(
            item,
            ("speakerId", "speaker_id", "speakerLabel", "speaker", "speakerIndex"),
            path,
        )
        validate_timed_record(start, end, speaker)
        spans.append(ReferenceSpan(start=start, end=end, speaker=speaker))
    return spans


def filter_reference_words(words: list[ReferenceWord]) -> tuple[list[ReferenceWord], int]:
    kept: list[ReferenceWord] = []
    dropped = 0
    for word in words:
        if _is_annotation_token(word.text):
            dropped += 1
            continue
        kept.append(word)
    return kept, dropped


def parse_rttm(path: Path) -> list[ReferenceSpan]:
    spans: list[ReferenceSpan] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8 or parts[0] != "SPEAKER":
            raise ValueError(f"{path}:{line_number}: expected RTTM SPEAKER line")
        start = _float_field(parts[3], path, line_number, "start")
        duration = _float_field(parts[4], path, line_number, "duration")
        speaker = parts[7]
        span = ReferenceSpan(start=start, end=start + duration, speaker=speaker)
        validate_timed_record(span.start, span.end, span.speaker)
        spans.append(span)
    return spans


def parse_ami_words_xml(path: Path, *, speaker: str | None = None) -> tuple[list[ReferenceWord], int]:
    tree = ET.parse(path)
    default_speaker = speaker or _speaker_from_ami_path(path)
    words: list[ReferenceWord] = []
    dropped = 0
    for element in tree.getroot().iter():
        if _local_tag(element.tag) != "w":
            continue
        text = "".join(element.itertext()).strip()
        if not text:
            continue
        if element.attrib.get("punc") == "true" or _is_annotation_token(text):
            dropped += 1
            continue
        if not _has_any_attr(element, ("starttime", "start")) or not _has_any_attr(element, ("endtime", "end")):
            dropped += 1
            continue
        start = _xml_float_attr(element, ("starttime", "start"), path)
        end = _xml_float_attr(element, ("endtime", "end"), path)
        word_speaker = element.attrib.get("speaker") or element.attrib.get("who") or default_speaker
        validate_timed_record(start, end, word_speaker)
        words.append(ReferenceWord(text=text, start=start, end=end, speaker=word_speaker))
    kept, later_dropped = filter_reference_words(words)
    return kept, dropped + later_dropped


def parse_ami_segments_xml(path: Path, *, speaker: str | None = None) -> list[ReferenceSpan]:
    tree = ET.parse(path)
    default_speaker = speaker or _speaker_from_ami_path(path)
    spans: list[ReferenceSpan] = []
    for element in tree.getroot().iter():
        if _local_tag(element.tag) != "segment":
            continue
        start = _xml_float_attr(element, ("transcriber_start", "starttime", "start"), path)
        end = _xml_float_attr(element, ("transcriber_end", "endtime", "end"), path)
        word_speaker = element.attrib.get("speaker") or element.attrib.get("who") or default_speaker
        validate_timed_record(start, end, word_speaker)
        spans.append(ReferenceSpan(start=start, end=end, speaker=word_speaker))
    return spans


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("words"), list):
        payload = payload["words"]
    if not isinstance(payload, list):
        raise ValueError(f"{path}: expected a list or an object with a words list")
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"{path}: word row {index} must be an object")
        rows.append(item)
    return rows


def _reference_word_from_json(item: dict[str, Any], path: Path) -> ReferenceWord:
    text = _string_field(item, "text", path)
    start = _number_field(item, "start", path)
    end = _number_field(item, "end", path)
    speaker = _string_field(item, "speaker", path)
    validate_timed_record(start, end, speaker)
    return ReferenceWord(text=text, start=start, end=end, speaker=speaker)


def _predicted_word_from_json(item: dict[str, Any], path: Path) -> PredictedWord:
    text = _string_field(item, "text", path)
    start = _number_field(item, "start", path)
    end = _number_field(item, "end", path)
    speaker = _string_field(item, "speaker", path)
    validate_timed_record(start, end, speaker)
    return PredictedWord(text=text, start=start, end=end, speaker=speaker)


def _string_field(item: dict[str, Any], key: str, path: Path) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}: field {key!r} must be a non-empty string")
    return value.strip()


def _number_field(item: dict[str, Any], key: str, path: Path) -> float:
    value = item.get(key)
    if not isinstance(value, int | float):
        raise ValueError(f"{path}: field {key!r} must be numeric")
    return float(value)


def _first_number_field(item: dict[str, Any], keys: tuple[str, ...], path: Path) -> float:
    for key in keys:
        value = item.get(key)
        if isinstance(value, int | float):
            return float(value)
    joined = "/".join(keys)
    raise ValueError(f"{path}: one of {joined!r} must be numeric")


def _first_string_field(item: dict[str, Any], keys: tuple[str, ...], path: Path) -> str:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, int):
            return str(value)
    joined = "/".join(keys)
    raise ValueError(f"{path}: one of {joined!r} must be a non-empty speaker label")


def _float_field(value: str, path: Path, line_number: int, key: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{path}:{line_number}: field {key!r} must be numeric") from exc


def _is_annotation_token(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if not any(character.isalnum() for character in stripped):
        return True
    return bool(_ANNOTATION_TOKEN_RE.match(stripped))


def _xml_float_attr(element: ET.Element, names: tuple[str, ...], path: Path) -> float:
    for name in names:
        value = element.attrib.get(name)
        if value is None:
            continue
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(f"{path}: XML attribute {name!r} must be numeric") from exc
    joined = "/".join(names)
    raise ValueError(f"{path}: XML word is missing {joined} timing attribute")


def _has_any_attr(element: ET.Element, names: tuple[str, ...]) -> bool:
    return any(name in element.attrib for name in names)


def _local_tag(tag: str) -> str:
    return tag.rsplit("}", maxsplit=1)[-1]


def _speaker_from_ami_path(path: Path) -> str:
    parts = path.name.split(".")
    if len(parts) >= 3 and parts[-2] in {"words", "segments"}:
        return parts[-3]
    return path.stem
