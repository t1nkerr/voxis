"""Bench-only Sortformer candidate adapter."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from voxis.bench.parsers import load_candidate_spans_json
from voxis.bench.scoring import ReferenceSpan, span_shape_stats
from voxis.diarize import SORTFORMER_INSTALL_HINT, SORTFORMER_TIMEOUT_SECONDS


@dataclass(frozen=True)
class SortformerRunConfig:
    audio: Path
    out: Path
    model_id: str
    onset: float | None
    offset: float | None
    pad_onset: float | None
    pad_offset: float | None
    min_duration_on: float | None
    min_duration_off: float | None
    debug: bool


def run_sortformer_candidate(config: SortformerRunConfig) -> dict[str, Any]:
    """Run FluidAudio Sortformer and write a stable candidate artifact directory.

    This is deliberately bench-only. It shells out to FluidAudio's CLI and normalizes
    the raw result into the candidate JSON shape consumed by the existing scorers.
    """

    _validate_config(config)
    config.out.mkdir(parents=True, exist_ok=True)
    raw_json_path = config.out / "sortformer_raw.json"
    candidate_json_path = config.out / "candidate.json"
    metadata_path = config.out / "metadata.json"
    stdout_path = config.out / "sortformer.stdout.log"
    stderr_path = config.out / "sortformer.stderr.log"

    command = _build_command(config, raw_json_path)
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=SORTFORMER_TIMEOUT_SECONDS
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Sortformer command timed out after {SORTFORMER_TIMEOUT_SECONDS} seconds: {command[0]}"
        ) from exc
    runtime_seconds = time.perf_counter() - started
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"Sortformer command failed with exit code {completed.returncode}; see {stderr_path} and {stdout_path}"
        )
    if not raw_json_path.exists():
        raise RuntimeError(f"Sortformer command did not create expected output JSON: {raw_json_path}")

    raw_spans = load_candidate_spans_json(raw_json_path)
    candidate_spans = raw_spans
    candidate_payload = {"segments": [_span_to_json(span) for span in candidate_spans]}
    candidate_json_path.write_text(json.dumps(candidate_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    metadata = {
        "backend": "sortformer",
        "adapter": "voxis-bench run-sortformer",
        "audio": str(config.audio),
        "model_id": config.model_id,
        "runner": {
            "command": command,
            "returncode": completed.returncode,
            "runtime_seconds": round(runtime_seconds, 3),
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
        },
        "post_processing": {
            "sortformer_native": {
                "onset": config.onset,
                "offset": config.offset,
                "pad_onset": config.pad_onset,
                "pad_offset": config.pad_offset,
                "min_duration_on": config.min_duration_on,
                "min_duration_off": config.min_duration_off,
            },
        },
        "artifacts": {
            "raw_json": str(raw_json_path),
            "candidate_json": str(candidate_json_path),
            "metadata_json": str(metadata_path),
        },
        "raw_span_stats": span_shape_stats(raw_spans),
        "candidate_span_stats": span_shape_stats(candidate_spans),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def _validate_config(config: SortformerRunConfig) -> None:
    if not config.audio.exists():
        raise ValueError(f"audio file does not exist: {config.audio}")


def _build_command(config: SortformerRunConfig, output_path: Path) -> list[str]:
    if shutil.which("fluidaudio") is None:
        raise RuntimeError(SORTFORMER_INSTALL_HINT)
    command = ["fluidaudio", "sortformer"]

    command.extend([str(config.audio), "--output", str(output_path)])
    _append_float_arg(command, "--onset", config.onset)
    _append_float_arg(command, "--offset", config.offset)
    _append_float_arg(command, "--pad-onset", config.pad_onset)
    _append_float_arg(command, "--pad-offset", config.pad_offset)
    _append_float_arg(command, "--min-duration-on", config.min_duration_on)
    _append_float_arg(command, "--min-duration-off", config.min_duration_off)
    if config.debug:
        command.append("--debug")
    return command


def _append_float_arg(command: list[str], name: str, value: float | None) -> None:
    if value is not None:
        command.extend([name, str(value)])


def _span_to_json(span: ReferenceSpan) -> dict[str, float | str]:
    return {
        "startTimeSeconds": span.start,
        "endTimeSeconds": span.end,
        "speakerId": span.speaker,
    }
