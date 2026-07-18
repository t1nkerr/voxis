from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from voxis.audio import LoadedAudio
from voxis.diarization_types import DiarizeSpan

DIARIZATION_BACKEND_ID = "sortformer"
DIARIZATION_MODEL_ID = "FluidAudio Sortformer high-context"
SORTFORMER_EXECUTABLE = "fluidaudio"
SORTFORMER_CONFIG = "high"
SORTFORMER_INSTALL_HINT = (
    "Sortformer executable not found. Install the FluidAudio CLI as `fluidaudio` on PATH "
    "(see https://github.com/t1nkerr/voxis#diarization-setup), then verify with `which fluidaudio`."
)
SORTFORMER_TIMEOUT_SECONDS = 3600
SORTFORMER_SPEAKER_RE = re.compile(r"(?:speaker\s*)?(\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class DiarizationResult:
    backend: str
    model: str
    spans: list[DiarizeSpan]
    command: list[str]
    runtime_seconds: float
    runner: dict[str, Any]
    raw_payload: dict[str, Any]


def diarize_audio(audio: LoadedAudio) -> DiarizationResult:
    """Run the production Sortformer diarizer and normalize spans.

    This is intentionally a clean Sortformer path with no alternate
    diarization backend in the active production flow.
    """

    executable = SORTFORMER_EXECUTABLE
    if shutil.which(executable) is None:
        raise RuntimeError(SORTFORMER_INSTALL_HINT)

    with tempfile.TemporaryDirectory(prefix="voxis-sortformer-") as tmp:
        raw_path = Path(tmp) / "sortformer_raw.json"
        command = _sortformer_command(audio.path, raw_path)
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command, check=False, capture_output=True, text=True, timeout=SORTFORMER_TIMEOUT_SECONDS
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Sortformer timed out after {SORTFORMER_TIMEOUT_SECONDS} seconds: {command[0]}"
            ) from exc
        runtime_seconds = time.perf_counter() - started
        if completed.returncode != 0:
            stderr = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"Sortformer failed with exit code {completed.returncode}: {stderr[:1000]}")
        if not raw_path.exists():
            raise RuntimeError(f"Sortformer did not write expected output: {raw_path}")

        raw_payload = json.loads(raw_path.read_text(encoding="utf-8"))

    spans = _parse_sortformer_spans(raw_payload)
    return DiarizationResult(
        backend=DIARIZATION_BACKEND_ID,
        model=DIARIZATION_MODEL_ID,
        spans=spans,
        command=command,
        runtime_seconds=runtime_seconds,
        runner={
            "executable": executable,
            "returncode": completed.returncode,
            "stderr": completed.stderr.strip(),
            "stdout": completed.stdout.strip(),
        },
        raw_payload=_raw_payload_summary(raw_payload),
    )


def _sortformer_command(audio_path: Path, output_path: Path) -> list[str]:
    return [
        SORTFORMER_EXECUTABLE,
        "sortformer",
        str(audio_path),
        "--output",
        str(output_path),
        "--config",
        SORTFORMER_CONFIG,
    ]


def _parse_sortformer_spans(payload: Any) -> list[DiarizeSpan]:
    segments = payload.get("segments") if isinstance(payload, dict) else payload
    if not isinstance(segments, list):
        raise ValueError("Sortformer output must be a list or an object with a segments list")

    speaker_remap: dict[str, str] = {}
    spans: list[DiarizeSpan] = []
    for index, item in enumerate(segments):
        if not isinstance(item, dict):
            raise ValueError(f"Sortformer segment {index} must be an object")
        start = _first_number(item, ("startTimeSeconds", "start_time", "startTime", "start"), index)
        end = _first_number(item, ("endTimeSeconds", "end_time", "endTime", "end"), index)
        if end <= start:
            continue
        raw_speaker = _raw_sortformer_speaker(item, index)
        speaker = _normalize_sortformer_speaker(raw_speaker, speaker_remap)
        spans.append(DiarizeSpan(start=start, end=end, speaker=speaker))
    return sorted(spans, key=lambda span: (span.start, span.end, span.speaker))


def _first_number(item: dict[str, Any], keys: tuple[str, ...], index: int) -> float:
    for key in keys:
        value = item.get(key)
        if isinstance(value, int | float):
            return float(value)
    raise ValueError(f"Sortformer segment {index} is missing numeric start/end fields")


def _raw_sortformer_speaker(item: dict[str, Any], index: int) -> str:
    value = item.get("speakerIndex")
    if isinstance(value, int):
        return str(value)
    for key in ("speakerId", "speaker_id", "speakerLabel", "speaker"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, int):
            return str(value)
    return str(index)


def _normalize_sortformer_speaker(raw_speaker: str, remap: dict[str, str]) -> str:
    match = SORTFORMER_SPEAKER_RE.search(raw_speaker.strip())
    if match:
        return f"SPEAKER_{int(match.group(1)):02d}"
    if raw_speaker not in remap:
        remap[raw_speaker] = f"SPEAKER_{len(remap):02d}"
    return remap[raw_speaker]


def _raw_payload_summary(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"segment_count": len(payload) if isinstance(payload, list) else None}
    return {
        "audio_file": payload.get("audioFile"),
        "duration_seconds": payload.get("durationSeconds"),
        "frame_duration_seconds": payload.get("frameDurationSeconds"),
        "processing_time_seconds": payload.get("processingTimeSeconds"),
        "rtfx": payload.get("rtfx"),
        "segment_count": payload.get("segmentCount") or len(payload.get("segments", [])),
    }
