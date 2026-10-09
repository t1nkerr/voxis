from __future__ import annotations

import json
import math
import time
from collections import Counter
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
# Per-chunk generation cap. Observed speech peaks near 6 tokens/s (Chinese ~350 chars/min), so 32 tokens/s leaves
# ~5x headroom while bounding the tokens a looping chunk can generate, instead of sharing one whole-file budget.
MAX_TOKENS_PER_SECOND = 32.0
MIN_CHUNK_MAX_TOKENS = 256
DEGENERATION_RETRY_CHUNK_DURATION_SECONDS = 30.0
DEGENERATION_RETRY_MAX_COVERAGE_GAP_SECONDS = 0.1
DEGENERATION_RETRY_MAX_TOTAL_UNCOVERED_SECONDS = 0.1
DEGENERATION_MIN_DURATION_SECONDS = 10.0
DEGENERATION_MIN_TEXT_CHARS = 1_000
DEGENERATION_MIN_CHARS_PER_SECOND = 50.0
DEGENERATION_MAX_LOOP_UNIT_CHARS = 32
DEGENERATION_MIN_REPEATED_SEQUENCE_COVERAGE = 0.8
DEGENERATION_POLICY_ID = "asr_segment_degeneration_guard_v1"
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
    degeneration_retries: list[dict[str, Any]]


def transcribe_audio(
    audio: LoadedAudio,
    *,
    context: str | None,
    language: str,
    config: VoxisConfig,
    progress: ProgressCallback | None = None,
    require_complete_alignment: bool = False,
) -> AsrResult:
    term_extraction = extract_terms(context, config=config)
    terms = prompt_terms(term_extraction.terms, config=config)
    waveform = audio.waveform

    system_prompt = system_prompt_from_terms(terms, config=config)

    _emit_progress(
        progress,
        "asr",
        "started",
        language=language,
        chunk_duration_seconds=CHUNK_DURATION_SECONDS,
        max_tokens_per_second=MAX_TOKENS_PER_SECOND,
        asr_prompt_term_count=len(terms),
    )
    try:
        asr_output = _generate_chunked_asr(
            waveform,
            language=language,
            chunk_duration=CHUNK_DURATION_SECONDS,
            system_prompt=system_prompt,
            progress=progress,
        )
    except Exception as exc:
        _emit_progress(progress, "asr", "failed", error_type=type(exc).__name__, error_message=str(exc))
        raise

    segments = asr_output.segments
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
    segments, degeneration_retries = _recover_degenerate_segments(
        segments,
        capped_segment_indices=asr_output.capped_segment_indices,
        waveform=waveform,
        language=language,
        progress=progress,
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
            f"{truncation_gap:.1f}s uncovered."
        )
    primary_language = segments[0]["language"] if segments else "unknown"
    _emit_progress(
        progress,
        "asr",
        "done",
        detected_language=str(primary_language),
        segment_count=len(segments),
        transcript_chars=len(_segments_text(segments)),
        degeneration_retry_count=len(degeneration_retries),
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

    if require_complete_alignment and alignment_errors:
        first_error = alignment_errors[0]
        _emit_progress(
            progress,
            "alignment",
            "failed",
            reason="unresolved_segment_errors",
            error_count=len(alignment_errors),
            first_error=first_error,
        )
        raise RuntimeError(
            f"forced alignment failed for {len(alignment_errors)} segment(s); "
            "diarization requires complete word timestamps"
        )
    _emit_progress(progress, "alignment", "done", word_count=len(words), error_count=len(alignment_errors))
    return AsrResult(
        text=_segments_text(segments),
        language=str(primary_language),
        audio_duration_s=audio.duration_s,
        terms=terms,
        profile=profile_payload(
            language=language,
            terms=terms,
            term_extraction=term_extraction,
            prompt_echo=prompt_echo,
            prompt_echo_trimmed_segment_count=prompt_echo_trimmed_segment_count,
            degeneration_retry_count=len(degeneration_retries),
            config=config,
        ),
        segments=segments,
        words=words,
        alignment_errors=alignment_errors,
        term_extraction=term_extraction,
        prompt_echo=prompt_echo,
        prompt_echo_trimmed_segment_count=prompt_echo_trimmed_segment_count,
        degeneration_retries=degeneration_retries,
    )


def _recover_degenerate_segments(
    segments: list[dict[str, Any]],
    *,
    capped_segment_indices: frozenset[int],
    waveform: Any,
    language: str,
    progress: ProgressCallback | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    recovered: list[dict[str, Any]] = []
    retries: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        initial_metrics = _segment_degeneration_metrics(segment, token_cap_hit=index in capped_segment_indices)
        if not initial_metrics["suspect"]:
            recovered.append(segment)
            continue

        _emit_progress(
            progress,
            "asr",
            "degenerate_segment_detected",
            attempt="initial",
            segment_index=index,
            metrics=initial_metrics,
        )
        start = float(segment["start"])
        end = float(segment["end"])
        start_sample = max(0, int(start * SAMPLE_RATE_HZ))
        end_sample = min(len(waveform), max(start_sample, int(end * SAMPLE_RATE_HZ)))
        _emit_progress(
            progress,
            "asr",
            "segment_retry_started",
            segment_index=index,
            segment_start=segment["start"],
            segment_end=segment["end"],
            chunk_duration_seconds=DEGENERATION_RETRY_CHUNK_DURATION_SECONDS,
            max_tokens_per_second=MAX_TOKENS_PER_SECOND,
            context_prompt=False,
            metrics=initial_metrics,
        )
        try:
            retry_output = _generate_chunked_asr(
                waveform[start_sample:end_sample],
                language=language,
                chunk_duration=DEGENERATION_RETRY_CHUNK_DURATION_SECONDS,
                system_prompt=None,
                progress=None,
            )
        except Exception as exc:
            _emit_progress(
                progress,
                "asr",
                "failed",
                reason="degenerate_segment",
                segment_index=index,
                retry_error_type=type(exc).__name__,
                retry_error_message=str(exc),
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            raise RuntimeError(f"ASR degenerate segment retry failed at {start:.3f}-{end:.3f}s") from exc

        replacement, replacement_capped = _offset_retry_segments(
            retry_output.segments,
            capped_indices=retry_output.capped_segment_indices,
            offset_seconds=start,
            duration_seconds=end - start,
        )
        retry_metrics = [
            _segment_degeneration_metrics(item, token_cap_hit=capped)
            for item, capped in zip(replacement, replacement_capped, strict=True)
        ]
        coverage_metrics = _retry_coverage_metrics(
            replacement,
            span_start=start,
            span_end=end,
        )
        retry_record = {
            "segment_index": index,
            "segment_start": segment["start"],
            "segment_end": segment["end"],
            "initial_metrics": initial_metrics,
            "retry_metrics": retry_metrics,
            "coverage_metrics": coverage_metrics,
            "replacement_segment_count": len(replacement),
        }
        retries.append(retry_record)
        _emit_progress(
            progress,
            "asr",
            "segment_retry_done",
            segment_index=index,
            replacement_segment_count=len(replacement),
            metrics=retry_metrics,
            coverage_metrics=coverage_metrics,
        )
        if not replacement or not coverage_metrics["complete"] or any(metrics["suspect"] for metrics in retry_metrics):
            _emit_progress(
                progress,
                "asr",
                "failed",
                reason="degenerate_segment",
                segment_index=index,
                initial_metrics=initial_metrics,
                retry_metrics=retry_metrics,
                coverage_metrics=coverage_metrics,
            )
            raise RuntimeError(f"ASR degenerate segment persisted after retry at {start:.3f}-{end:.3f}s")
        recovered.extend(replacement)
    return recovered, retries


def _offset_retry_segments(
    segments: list[dict[str, Any]],
    *,
    capped_indices: frozenset[int],
    offset_seconds: float,
    duration_seconds: float,
) -> tuple[list[dict[str, Any]], list[bool]]:
    replacement: list[dict[str, Any]] = []
    capped: list[bool] = []
    for index, segment in enumerate(segments):
        relative_start = max(0.0, float(segment["start"]))
        relative_end = min(duration_seconds, float(segment["end"]))
        if relative_end <= relative_start:
            continue
        capped.append(index in capped_indices)
        replacement.append(
            {
                **segment,
                "start": json_time(offset_seconds + relative_start),
                "end": json_time(offset_seconds + relative_end),
            }
        )
    return replacement, capped


def _segment_degeneration_metrics(segment: dict[str, Any], *, token_cap_hit: bool = False) -> dict[str, Any]:
    text = str(segment.get("text", "")).strip()
    duration_seconds = max(0.0, float(segment["end"]) - float(segment["start"]))
    content_chars = [char for char in text if not char.isspace()]
    character_counts = Counter(content_chars)
    dominant_count = max(character_counts.values(), default=0)
    dominant_ratio = dominant_count / len(content_chars) if content_chars else 0.0
    repeated_sequence = _repeated_sequence_metrics("".join(content_chars))
    chars_per_second = len(text) / duration_seconds if duration_seconds > 0 else float("inf")
    suspect = token_cap_hit or (
        duration_seconds >= DEGENERATION_MIN_DURATION_SECONDS
        and len(text) >= DEGENERATION_MIN_TEXT_CHARS
        and chars_per_second >= DEGENERATION_MIN_CHARS_PER_SECOND
        and repeated_sequence["coverage"] >= DEGENERATION_MIN_REPEATED_SEQUENCE_COVERAGE
    )
    return {
        "policy_id": DEGENERATION_POLICY_ID,
        "duration_seconds": round(duration_seconds, 3),
        "text_chars": len(text),
        "chars_per_second": round(chars_per_second, 3),
        "dominant_char": character_counts.most_common(1)[0][0] if content_chars else None,
        "dominant_char_ratio": round(dominant_ratio, 4),
        "repeated_sequence_unit_chars": repeated_sequence["unit_chars"],
        "repeated_sequence": repeated_sequence["sequence"],
        "repeated_sequence_coverage": repeated_sequence["coverage"],
        "token_cap_hit": token_cap_hit,
        "suspect": suspect,
    }


def _repeated_sequence_metrics(text: str) -> dict[str, Any]:
    if not text:
        return {"unit_chars": 0, "sequence": None, "coverage": 0.0}

    best_unit_chars = 0
    best_sequence: str | None = None
    best_coverage = 0.0
    max_unit_chars = min(DEGENERATION_MAX_LOOP_UNIT_CHARS, len(text))
    for unit_chars in range(1, max_unit_chars + 1):
        sequence_counts = Counter(text[index : index + unit_chars] for index in range(len(text) - unit_chars + 1))
        sequence, count = sequence_counts.most_common(1)[0]
        coverage = min(1.0, count * unit_chars / len(text))
        if coverage > best_coverage:
            best_unit_chars = unit_chars
            best_sequence = sequence
            best_coverage = coverage
    return {
        "unit_chars": best_unit_chars,
        "sequence": best_sequence,
        "coverage": round(best_coverage, 4),
    }


def _retry_coverage_metrics(segments: list[dict[str, Any]], *, span_start: float, span_end: float) -> dict[str, Any]:
    if not segments:
        return {
            "start_gap_seconds": None,
            "end_gap_seconds": None,
            "maximum_internal_gap_seconds": None,
            "total_uncovered_seconds": None,
            "ordered_non_overlapping": True,
            "complete": False,
        }

    starts = [float(segment["start"]) for segment in segments]
    ends = [float(segment["end"]) for segment in segments]
    ordered_non_overlapping = all(starts[index] >= ends[index - 1] for index in range(1, len(segments)))
    internal_gaps = [max(0.0, starts[index] - ends[index - 1]) for index in range(1, len(segments))]
    start_gap = max(0.0, starts[0] - span_start)
    end_gap = max(0.0, span_end - ends[-1])
    maximum_internal_gap = max(internal_gaps, default=0.0)
    total_uncovered = start_gap + end_gap + sum(internal_gaps)
    complete = (
        ordered_non_overlapping
        and all(
            gap <= DEGENERATION_RETRY_MAX_COVERAGE_GAP_SECONDS for gap in (start_gap, end_gap, maximum_internal_gap)
        )
        and total_uncovered <= DEGENERATION_RETRY_MAX_TOTAL_UNCOVERED_SECONDS
    )
    return {
        "start_gap_seconds": round(start_gap, 3),
        "end_gap_seconds": round(end_gap, 3),
        "maximum_internal_gap_seconds": round(maximum_internal_gap, 3),
        "total_uncovered_seconds": round(total_uncovered, 3),
        "ordered_non_overlapping": ordered_non_overlapping,
        "complete": complete,
    }


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
    degeneration_retry_count: int = 0,
    config: VoxisConfig | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "profile_id": PROFILE_ID,
        "asr_model_id": ASR_MODEL_ID,
        "aligner_model_id": ALIGNER_MODEL_ID,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "chunk_duration_seconds": CHUNK_DURATION_SECONDS,
        "max_tokens_per_second": MAX_TOKENS_PER_SECOND,
        "min_chunk_max_tokens": MIN_CHUNK_MAX_TOKENS,
        "language": language,
        "degeneration_guard": {
            "policy_id": DEGENERATION_POLICY_ID,
            "minimum_duration_seconds": DEGENERATION_MIN_DURATION_SECONDS,
            "minimum_text_chars": DEGENERATION_MIN_TEXT_CHARS,
            "minimum_chars_per_second": DEGENERATION_MIN_CHARS_PER_SECOND,
            "maximum_loop_unit_chars": DEGENERATION_MAX_LOOP_UNIT_CHARS,
            "minimum_repeated_sequence_coverage": DEGENERATION_MIN_REPEATED_SEQUENCE_COVERAGE,
            "retry_chunk_duration_seconds": DEGENERATION_RETRY_CHUNK_DURATION_SECONDS,
            "retry_max_coverage_gap_seconds": DEGENERATION_RETRY_MAX_COVERAGE_GAP_SECONDS,
            "retry_max_total_uncovered_seconds": DEGENERATION_RETRY_MAX_TOTAL_UNCOVERED_SECONDS,
            "retry_count": degeneration_retry_count,
        },
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


@dataclass(frozen=True)
class _ChunkedAsrOutput:
    segments: list[dict[str, Any]]
    capped_segment_indices: frozenset[int]


def _generate_chunked_asr(
    waveform: Any,
    *,
    language: str,
    chunk_duration: float,
    system_prompt: str | None,
    progress: ProgressCallback | None,
) -> _ChunkedAsrOutput:
    """Run ASR one chunk at a time so each chunk gets its own token cap.

    mlx-audio's chunked generate shares max_tokens across the whole file, so a single looping chunk can generate
    for hours before the degeneration guard sees any output. Driving the chunk loop here bounds each chunk and
    flags cap hits for the existing retry.
    """
    from mlx_audio.stt.models.qwen3_asr.qwen3_asr import split_audio_into_chunks  # type: ignore[import-untyped]

    model = _load_asr_model()
    chunks = split_audio_into_chunks(waveform, sr=SAMPLE_RATE_HZ, chunk_duration=chunk_duration)
    # mlx-audio pins the first chunk's detected language for later chunks; keep that behavior.
    chunk_language: str | None = None if language == "auto" else language
    segments: list[dict[str, Any]] = []
    capped: set[int] = set()
    for chunk_index, (chunk, offset_seconds) in enumerate(chunks):
        chunk_seconds = len(chunk) / SAMPLE_RATE_HZ
        max_tokens = max(MIN_CHUNK_MAX_TOKENS, math.ceil(chunk_seconds * MAX_TOKENS_PER_SECOND))
        kwargs: dict[str, Any] = {
            "language": chunk_language,
            # The chunk is already split; an infinite duration keeps mlx-audio from splitting it again.
            "chunk_duration": math.inf,
            "max_tokens": max_tokens,
        }
        if system_prompt is not None:
            kwargs["system_prompt"] = system_prompt
        started = time.monotonic()
        output = model.generate(chunk, **kwargs)
        generation_tokens = int(getattr(output, "generation_tokens", 0))
        token_cap_hit = generation_tokens >= max_tokens
        # Offset before rounding: rounding chunk-relative times first can make adjacent chunks overlap by 1 ms.
        chunk_segments = [
            provider_segment_payload(
                {
                    **segment,
                    "start": float(segment["start"]) + offset_seconds,
                    "end": float(segment["end"]) + offset_seconds,
                }
            )
            for segment in output.segments
        ]
        for segment in chunk_segments:
            if token_cap_hit:
                capped.add(len(segments))
            segments.append(segment)
        if chunk_language is None and chunk_segments:
            chunk_language = chunk_segments[0]["language"]
        _emit_progress(
            progress,
            "asr",
            "chunk_done",
            chunk_index=chunk_index,
            chunk_count=len(chunks),
            chunk_start=json_time(offset_seconds),
            chunk_seconds=round(chunk_seconds, 3),
            max_tokens=max_tokens,
            generation_tokens=generation_tokens,
            token_cap_hit=token_cap_hit,
            chunk_elapsed_seconds=round(time.monotonic() - started, 3),
        )
    return _ChunkedAsrOutput(segments=segments, capped_segment_indices=frozenset(capped))


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
