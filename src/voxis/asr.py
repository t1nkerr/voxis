from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from voxis.asr_terms import (
    PromptEchoResult,
    TermExtractionResult,
    asr_prompt_payload,
    detect_prompt_echo,
    extract_terms,
    global_terms,
    prompt_echo_payload,
    prompt_echo_tail_start,
    prompt_terms,
    system_prompt_from_terms,
    term_extraction_payload,
)
from voxis.audio import LoadedAudio, json_time, provider_segment_payload, sha256_text
from voxis.config import VoxisConfig

ASR_MODEL_ID = "mlx-community/Qwen3-ASR-1.7B-8bit"
ALIGNER_MODEL_ID = "mlx-community/Qwen3-ForcedAligner-0.6B-8bit"
SAMPLE_RATE_HZ = 16_000
CHUNK_DURATION_SECONDS = 60.0
MAX_TOKENS_BUDGET = 524_288
TRUNCATION_MIN_GAP_SECONDS = 120.0
TRUNCATION_GAP_RATIO = 0.2
PROFILE_ID = "qwen3_asr_forced_aligner_v1"
ProgressCallback = Callable[[str, str, dict[str, Any]], None]


@dataclass(frozen=True)
class AsrResult:
    text: str
    language: str
    audio_duration_s: float
    terms: tuple[str, ...]
    profile: dict[str, Any]
    segments: list[dict[str, Any]]
    words: list[dict[str, Any]]
    alignment_errors: list[dict[str, str]]
    term_extraction: TermExtractionResult
    prompt_echo: PromptEchoResult
    prompt_echo_trimmed_segment_count: int


def transcribe_audio(
    audio: LoadedAudio,
    *,
    context: str | None,
    language: str,
    config: VoxisConfig,
    progress: ProgressCallback | None = None,
) -> AsrResult:
    term_extraction = extract_terms(context, config=config)
    terms = prompt_terms(term_extraction.terms, config=config)
    waveform = audio.waveform

    asr_kwargs: dict[str, Any] = {
        "language": None if language == "auto" else language,
        "chunk_duration": CHUNK_DURATION_SECONDS,
        "max_tokens": MAX_TOKENS_BUDGET,
    }
    system_prompt = system_prompt_from_terms(terms, config=config)
    if system_prompt is not None:
        asr_kwargs["system_prompt"] = system_prompt

    _emit_progress(
        progress,
        "asr",
        "started",
        language=language,
        chunk_duration_seconds=CHUNK_DURATION_SECONDS,
        max_tokens_budget=MAX_TOKENS_BUDGET,
        asr_prompt_term_count=len(terms),
    )
    try:
        asr_output = _generate_asr(waveform, asr_kwargs)
    except Exception as exc:
        _emit_progress(progress, "asr", "failed", error_type=type(exc).__name__, error_message=str(exc))
        raise

    segments = [provider_segment_payload(segment) for segment in asr_output.segments]
    prompt_echo = detect_prompt_echo(segments, terms)
    prompt_echo_trimmed_segment_count = 0
    if system_prompt is not None and prompt_echo.status == "detected":
        tail_start = prompt_echo_tail_start(prompt_echo)
        if tail_start is None:
            _emit_progress(
                progress,
                "asr",
                "failed",
                reason="prompt_echo_outside_tail",
                suspect_segment_count=len(prompt_echo.suspect_segments),
            )
            raise RuntimeError("ASR prompt echo detected outside contiguous tail")
        if tail_start == 0:
            _emit_progress(
                progress,
                "asr",
                "failed",
                reason="prompt_echo_all_segments",
                suspect_segment_count=len(prompt_echo.suspect_segments),
            )
            raise RuntimeError("ASR output contains only prompt echo")
        prompt_echo_trimmed_segment_count = len(segments) - tail_start
        segments = segments[:tail_start]
        _emit_progress(
            progress,
            "asr",
            "prompt_echo_trimmed",
            suspect_segment_count=len(prompt_echo.suspect_segments),
            trimmed_segment_count=prompt_echo_trimmed_segment_count,
        )
    truncation_gap = _truncation_gap_seconds(segments, audio_duration_s=audio.duration_s)
    if truncation_gap is not None:
        _emit_progress(
            progress,
            "asr",
            "failed",
            reason="transcript_truncated",
            audio_duration_s=round(audio.duration_s, 3),
            last_segment_end_s=round(float(segments[-1]["end"]), 3),
            uncovered_tail_s=round(truncation_gap, 3),
        )
        raise RuntimeError(
            "ASR transcript appears truncated: segments end "
            f"{float(segments[-1]['end']):.1f}s into {audio.duration_s:.1f}s of audio, leaving "
            f"{truncation_gap:.1f}s uncovered. The max_tokens budget ({MAX_TOKENS_BUDGET}) is the likely cause: "
            "mlx-audio treats it as a whole-file budget and stops early when it is exhausted."
        )
    primary_language = segments[0]["language"] if segments else "unknown"
    _emit_progress(
        progress,
        "asr",
        "done",
        detected_language=str(primary_language),
        segment_count=len(segments),
        transcript_chars=len(str(asr_output.text)),
    )

    _emit_progress(progress, "alignment", "started", segment_count=len(segments))
    try:
        aligner = _load_aligner_model()
        words: list[dict[str, Any]] = []
        alignment_errors: list[dict[str, str]] = []

        for segment in segments:
            start = float(segment["start"])
            end = float(segment["end"])
            start_sample = max(0, int(start * SAMPLE_RATE_HZ))
            end_sample = min(len(waveform), max(start_sample, int(end * SAMPLE_RATE_HZ)))
            try:
                aligned = aligner.generate(
                    audio=waveform[start_sample:end_sample],
                    text=str(segment["text"]),
                    language=str(segment["language"]),
                )
            except Exception as exc:
                alignment_errors.append(
                    {
                        "segment_start": str(segment["start"]),
                        "segment_end": str(segment["end"]),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )
                continue
            for item in _alignment_items(aligned):
                words.append(
                    {
                        "text": str(item.text),
                        "start": json_time(float(item.start_time) + start),
                        "end": json_time(float(item.end_time) + start),
                    }
                )
    except Exception as exc:
        _emit_progress(progress, "alignment", "failed", error_type=type(exc).__name__, error_message=str(exc))
        raise

    _emit_progress(
        progress,
        "alignment",
        "done",
        word_count=len(words),
        error_count=len(alignment_errors),
    )
    return AsrResult(
        text=_segments_text(segments) or str(asr_output.text),
        language=str(primary_language),
        audio_duration_s=audio.duration_s,
        terms=terms,
        profile=profile_payload(
            language=language,
            terms=terms,
            term_extraction=term_extraction,
            prompt_echo=prompt_echo,
            prompt_echo_trimmed_segment_count=prompt_echo_trimmed_segment_count,
            config=config,
        ),
        segments=segments,
        words=words,
        alignment_errors=alignment_errors,
        term_extraction=term_extraction,
        prompt_echo=prompt_echo,
        prompt_echo_trimmed_segment_count=prompt_echo_trimmed_segment_count,
    )


def _alignment_items(aligned: Any) -> list[Any]:
    outputs = aligned if isinstance(aligned, list) else [aligned]
    items: list[Any] = []
    for output in outputs:
        items.extend(list(output.items))
    return items


def profile_payload(
    *,
    language: str,
    terms: tuple[str, ...] = (),
    term_extraction: TermExtractionResult | None = None,
    prompt_echo: PromptEchoResult | None = None,
    prompt_echo_trimmed_segment_count: int = 0,
    config: VoxisConfig | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "profile_id": PROFILE_ID,
        "asr_model_id": ASR_MODEL_ID,
        "aligner_model_id": ALIGNER_MODEL_ID,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "chunk_duration_seconds": CHUNK_DURATION_SECONDS,
        "max_tokens_budget": MAX_TOKENS_BUDGET,
        "language": language,
    }
    if term_extraction is not None:
        payload["term_extraction"] = term_extraction_payload(term_extraction)
    if terms:
        payload["asr_prompt"] = asr_prompt_payload(
            terms,
            extracted_terms=term_extraction.terms if term_extraction is not None else (),
            global_terms=global_terms(config) if config is not None else (),
            trimmed_segment_count=prompt_echo_trimmed_segment_count,
        )
    if prompt_echo is not None:
        payload["prompt_echo"] = prompt_echo_payload(prompt_echo)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    payload["profile_hash"] = sha256_text(encoded)
    return payload


def _emit_progress(
    progress: ProgressCallback | None,
    stage: str,
    event: str,
    **fields: Any,
) -> None:
    if progress is not None:
        progress(stage, event, fields)


def _generate_asr(waveform: Any, kwargs: dict[str, Any]) -> Any:
    return _load_asr_model().generate(waveform, **kwargs)


def _segments_text(segments: list[dict[str, Any]]) -> str:
    return "\n".join(
        str(segment.get("text", "")).strip() for segment in segments if str(segment.get("text", "")).strip()
    )


def _truncation_gap_seconds(segments: list[dict[str, Any]], *, audio_duration_s: float) -> float | None:
    if not segments:
        return None
    gap = audio_duration_s - float(segments[-1]["end"])
    threshold = max(TRUNCATION_MIN_GAP_SECONDS, TRUNCATION_GAP_RATIO * audio_duration_s)
    return gap if gap > threshold else None


@lru_cache(maxsize=1)
def _load_asr_model() -> Any:
    from mlx_audio.stt.utils import load_model  # type: ignore[import-untyped]

    return load_model(ASR_MODEL_ID)


@lru_cache(maxsize=1)
def _load_aligner_model() -> Any:
    from mlx_audio.stt.utils import load_model  # type: ignore[import-untyped]

    return load_model(ALIGNER_MODEL_ID)
