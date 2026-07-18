from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LoadedAudio:
    path: Path
    sample_rate_hz: int
    waveform: np.ndarray
    duration_s: float


def load_audio_file(path: Path, *, sample_rate_hz: int) -> LoadedAudio:
    from mlx_audio.stt.utils import load_audio  # type: ignore[import-untyped]

    waveform = _mono_float32_waveform(load_audio(str(path), sr=sample_rate_hz))
    return LoadedAudio(
        path=path,
        sample_rate_hz=sample_rate_hz,
        waveform=waveform,
        duration_s=len(waveform) / sample_rate_hz,
    )


def _mono_float32_waveform(waveform: Any) -> np.ndarray:
    arr = np.asarray(waveform, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr[:, 0] if arr.shape[1] < arr.shape[0] else arr[0]
    return arr


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def json_time(value: float) -> float:
    return round(float(value), 3)


def provider_segment_payload(segment: Any) -> dict[str, Any]:
    return {
        "text": str(segment["text"]),
        "start": json_time(float(segment["start"])),
        "end": json_time(float(segment["end"])),
        "language": str(segment["language"]),
    }
