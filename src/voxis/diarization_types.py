from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DiarizeSpan:
    start: float
    end: float
    speaker: str


@dataclass(frozen=True)
class Turn:
    speaker: str
    start: float
    end: float
    text: str


def turns_payload(turns: list[Turn]) -> list[dict[str, Any]]:
    return [
        {
            "speaker": turn.speaker,
            "start": _json_time(turn.start),
            "end": _json_time(turn.end),
            "text": turn.text,
        }
        for turn in turns
    ]


def format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def spans_payload(spans: list[DiarizeSpan]) -> list[dict[str, Any]]:
    return [
        {
            "speaker": span.speaker,
            "start": _json_time(span.start),
            "end": _json_time(span.end),
        }
        for span in spans
    ]


def _json_time(value: float) -> float:
    return round(float(value), 3)
