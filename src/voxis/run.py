from __future__ import annotations

import json
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from voxis.audio import LoadedAudio, load_audio_file, sha256_file
from voxis.asr import SAMPLE_RATE_HZ, AsrResult, transcribe_audio
from voxis.config import VoxisConfig
from voxis.device import mlx_device_payload
from voxis.diarization_policy import (
    SortformerPolicyResult,
    apply_sortformer_policy,
    assigned_words_payload,
    assign_words_by_largest_overlap,
    turns_from_assigned_words,
)
from voxis.diarize import (
    DIARIZATION_BACKEND_ID,
    DIARIZATION_MODEL_ID,
    DiarizationResult,
    SORTFORMER_CONFIG,
    diarize_audio,
)
from voxis.diarization_types import (
    Turn,
    format_timestamp,
    spans_payload,
    turns_payload,
)
from voxis.polish import PolishResult, polish_turns, render_user_facing_polished_text
from voxis.speaker_map import (
    SpeakerMapResult,
    apply_speaker_map_to_turns,
    build_speaker_map,
    speaker_map_payload,
    speaker_map_provider_payload,
)

Language = Literal["auto", "en", "zh"]


@dataclass(frozen=True)
class TranscribeOptions:
    audio: Path
    out: Path
    context: str | None = None
    language: Language = "auto"
    diarize: bool = False
    polish: bool = False
    debug: bool = False


class _ProgressReporter:
    def __init__(self, out_dir: Path, *, debug: bool) -> None:
        self._started = time.monotonic()
        self._path: Path | None = None
        if debug:
            debug_dir = out_dir / "_debug"
            shutil.rmtree(debug_dir, ignore_errors=True)
            debug_dir.mkdir(exist_ok=True)
            self._path = debug_dir / "progress.jsonl"
            self._path.write_text("", encoding="utf-8")

    def event(self, stage: str, event: str, fields: dict[str, Any] | None = None) -> None:
        elapsed = round(time.monotonic() - self._started, 3)
        payload = {
            "time_utc": _now(),
            "elapsed_seconds": elapsed,
            "stage": stage,
            "event": event,
            **(fields or {}),
        }
        print(f"voxis {stage} {event} elapsed={elapsed:.3f}s", file=sys.stderr, flush=True)
        if self._path is not None:
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def transcribe(options: TranscribeOptions, config: VoxisConfig) -> int:
    started_at = _now()
    audio_path = options.audio.resolve()
    out_dir = options.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    progress = _ProgressReporter(out_dir, debug=options.debug)

    transcript_path = out_dir / "transcript.md"
    run_path = out_dir / "run.json"
    try:
        progress.event("run", "started", {"audio_path": str(audio_path), "debug": options.debug})
        progress.event("load_audio", "started", {"audio_path": str(audio_path)})
        audio = load_audio_file(audio_path, sample_rate_hz=SAMPLE_RATE_HZ)
        progress.event("load_audio", "done", {"duration_seconds": round(audio.duration_s, 3)})
        asr_result, diarization_result = _run_asr_and_diarization(
            audio,
            options=options,
            config=config,
            progress=progress,
        )
        assigned_words = (
            assign_words_by_largest_overlap(asr_result.words, diarization_result.spans)
            if diarization_result is not None
            else []
        )
        pre_policy_turns = turns_from_assigned_words(assigned_words)
        if assigned_words:
            progress.event("policy", "started", {"word_count": len(assigned_words)})
            policy_result = apply_sortformer_policy(
                assigned_words,
                speaker_mapping={},
                language=_sortformer_policy_language(options=options, asr_result=asr_result),
            )
            raw_turns = turns_from_assigned_words(policy_result.words)
            progress.event(
                "policy",
                "done",
                {
                    "changed_word_count": policy_result.audit["changed_word_count"],
                    "turn_count": len(raw_turns),
                },
            )
        else:
            policy_result = None
            raw_turns = []
        if options.diarize and raw_turns:
            progress.event(
                "speaker_map",
                "started",
                {"turn_count": len(raw_turns), "context_provided": options.context is not None},
            )
            speaker_map_result = build_speaker_map(raw_turns, context=options.context, config=config)
            progress.event(
                "speaker_map",
                "failed" if speaker_map_result.status == "failed" else "done",
                {
                    "status": speaker_map_result.status,
                    "failure_kind": speaker_map_result.failure_kind,
                    "issue_count": len(speaker_map_result.issues),
                },
            )
        else:
            speaker_map_result = None
        speaker_map_failed = speaker_map_result is not None and speaker_map_result.status == "failed"
        if speaker_map_result is not None and speaker_map_result.status == "failed":
            progress.event(
                "speaker_map",
                "fallback",
                {
                    "reason": "speaker map failed; keeping unmapped SPEAKER_XX labels",
                    "failure_kind": speaker_map_result.failure_kind,
                },
            )
        turns = (
            apply_speaker_map_to_turns(raw_turns, speaker_map_result.mapping)
            if speaker_map_result is not None and not speaker_map_failed
            else raw_turns
        )
        if options.polish and turns:
            progress.event("polish", "started", {"turn_count": len(turns)})
            polish_result = polish_turns(turns, context=options.context, config=config)
            progress.event(
                "polish",
                "done",
                {"status": polish_result.status, "issue_count": len(polish_result.issues)},
            )
        else:
            polish_result = None
        progress.event("outputs", "started", {"out_dir": str(out_dir)})
        transcript_path.write_text(
            _transcript_markdown(
                asr_result,
                turns=turns,
                polish_result=polish_result,
            ),
            encoding="utf-8",
        )
        if options.debug:
            _write_debug(
                out_dir / "_debug",
                asr_result,
                diarize_enabled=options.diarize,
                diarization_result=diarization_result,
                policy_result=policy_result,
                pre_policy_turns=pre_policy_turns,
                raw_turns=raw_turns,
                speaker_map_result=speaker_map_result,
                polish_result=polish_result,
            )
        run_path.write_text(
            json.dumps(
                _run_receipt(
                    options=options,
                    config=config,
                    audio_path=audio_path,
                    transcript_path=transcript_path,
                    asr_result=asr_result,
                    diarization_result=diarization_result,
                    policy_result=policy_result,
                    speaker_map_result=speaker_map_result,
                    turns=turns,
                    polish_result=polish_result,
                    started_at=started_at,
                    finished_at=_now(),
                ),
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        progress.event("outputs", "done", {"transcript_md": str(transcript_path), "run_json": str(run_path)})
        progress.event("run", "done")
        return 0
    except Exception as exc:
        progress.event("run", "failed", {"error_type": type(exc).__name__, "error_message": str(exc)})
        _remove_stale_failure_outputs(
            transcript_path,
            run_path,
            out_dir / "_debug",
            preserve_debug=options.debug,
        )
        raise


def _remove_stale_failure_outputs(
    transcript_path: Path,
    run_path: Path,
    debug_dir: Path,
    *,
    preserve_debug: bool = False,
) -> None:
    transcript_path.unlink(missing_ok=True)
    run_path.unlink(missing_ok=True)
    if not preserve_debug:
        shutil.rmtree(debug_dir, ignore_errors=True)


def _run_asr_and_diarization(
    audio: LoadedAudio, *, options: TranscribeOptions, config: VoxisConfig, progress: _ProgressReporter
) -> tuple[AsrResult, DiarizationResult | None]:
    if not options.diarize:
        return (
            transcribe_audio(
                audio,
                context=options.context,
                language=options.language,
                config=config,
                progress=progress.event,
            ),
            None,
        )
    asr_result = transcribe_audio(
        audio,
        context=options.context,
        language=options.language,
        config=config,
        progress=progress.event,
    )
    progress.event("diarization", "started")
    try:
        diarization_result = diarize_audio(audio)
    except Exception as exc:
        progress.event("diarization", "failed", {"error_type": type(exc).__name__, "error_message": str(exc)})
        raise
    progress.event(
        "diarization",
        "done",
        {
            "span_count": len(diarization_result.spans),
            "runtime_seconds": round(diarization_result.runtime_seconds, 3),
        },
    )
    return asr_result, diarization_result


def _transcript_markdown(
    asr_result: AsrResult,
    *,
    turns: list[Turn],
    polish_result: PolishResult | None,
) -> str:
    if polish_result is not None and polish_result.text:
        return "# Transcript\n\n" + render_user_facing_polished_text(polish_result.text).strip() + "\n"
    if turns:
        lines = [_format_turn_line(turn) for turn in turns]
        return "# Transcript\n\n" + "\n\n".join(lines).strip() + "\n"
    text = asr_result.text.strip()
    return f"# Transcript\n\n{text}\n" if text else "# Transcript\n\n"


def _write_debug(
    debug_dir: Path,
    asr_result: AsrResult,
    *,
    diarize_enabled: bool,
    diarization_result: DiarizationResult | None,
    policy_result: SortformerPolicyResult | None,
    pre_policy_turns: list[Turn],
    raw_turns: list[Turn],
    speaker_map_result: SpeakerMapResult | None,
    polish_result: PolishResult | None,
) -> None:
    debug_dir.mkdir(exist_ok=True)
    (debug_dir / "asr_segments.json").write_text(
        json.dumps(
            {
                "segments": asr_result.segments,
                "words": asr_result.words,
                "alignment_errors": asr_result.alignment_errors,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (debug_dir / "terms.txt").write_text(
        "\n".join(asr_result.terms) + ("\n" if asr_result.terms else ""), encoding="utf-8"
    )
    if diarize_enabled:
        (debug_dir / "diarization.json").write_text(
            json.dumps(
                {
                    "backend": DIARIZATION_BACKEND_ID,
                    "model": DIARIZATION_MODEL_ID,
                    "spans": spans_payload(diarization_result.spans if diarization_result is not None else []),
                    "pre_policy_turns": turns_payload(pre_policy_turns),
                    "turns": turns_payload(raw_turns),
                    "sortformer": _diarization_result_payload(diarization_result),
                    "policy": policy_result.audit if policy_result is not None else None,
                    "words": assigned_words_payload(policy_result.words) if policy_result is not None else [],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if speaker_map_result is not None:
            (debug_dir / "speaker_map.json").write_text(
                json.dumps(speaker_map_payload(speaker_map_result), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    if polish_result is not None:
        (debug_dir / "polish.json").write_text(
            json.dumps(
                {
                    "status": polish_result.status,
                    "prompt_template_id": polish_result.prompt_template_id,
                    "prompt_tokens": polish_result.prompt_tokens,
                    "completion_tokens": polish_result.completion_tokens,
                    "elapsed_seconds": round(polish_result.elapsed_seconds, 3),
                    "issues": list(polish_result.issues),
                    "chunk_count": polish_result.chunk_count,
                    "max_workers": polish_result.max_workers,
                    "effective_workers": polish_result.effective_workers,
                    "chunks": list(polish_result.chunks),
                    "text": polish_result.text,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def _format_turn_line(turn: Turn) -> str:
    return f"[{format_timestamp(turn.start)}] {turn.speaker}: {turn.text}"


def _sortformer_policy_language(*, options: TranscribeOptions, asr_result: AsrResult) -> str:
    if options.language in {"en", "zh"}:
        return options.language
    detected = asr_result.language.lower()
    if "chinese" in detected or detected.startswith("zh"):
        return "zh"
    return "en"


def _diarization_result_payload(result: DiarizationResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "backend": result.backend,
        "model": result.model,
        "runtime_seconds": round(result.runtime_seconds, 3),
        "command": result.command,
        "runner": result.runner,
        "raw_payload": result.raw_payload,
    }


def _run_receipt(
    *,
    options: TranscribeOptions,
    config: VoxisConfig,
    audio_path: Path,
    transcript_path: Path,
    asr_result: AsrResult,
    diarization_result: DiarizationResult | None,
    policy_result: SortformerPolicyResult | None,
    speaker_map_result: SpeakerMapResult | None,
    turns: list[Turn],
    polish_result: PolishResult | None,
    started_at: str,
    finished_at: str,
) -> dict[str, Any]:
    stat = audio_path.stat()
    mlx_device = mlx_device_payload()
    return {
        "schema_version": "voxis_run_v1",
        "status": "succeeded",
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "audio": {
            "path": str(audio_path),
            "size_bytes": stat.st_size,
            "sha256": sha256_file(audio_path),
            "duration_seconds": round(asr_result.audio_duration_s, 3),
        },
        "options": {
            "language": options.language,
            "diarize": options.diarize,
            "polish": options.polish,
            "debug": options.debug,
            "context_provided": options.context is not None,
        },
        "providers": {
            "asr": {
                **asr_result.profile,
                "device": mlx_device,
            },
            "alignment": {
                "model": asr_result.profile["aligner_model_id"],
                "enabled": True,
                "device": mlx_device,
                "status": "failed" if asr_result.alignment_errors else "succeeded",
                "error_count": len(asr_result.alignment_errors),
            },
            "diarization": {
                "backend": DIARIZATION_BACKEND_ID,
                "model": DIARIZATION_MODEL_ID,
                "enabled": options.diarize,
                "status": _diarization_status(options=options, diarization_result=diarization_result),
                "runtime_seconds": (
                    round(diarization_result.runtime_seconds, 3) if diarization_result is not None else 0
                ),
                "span_count": len(diarization_result.spans) if diarization_result is not None else 0,
                "turn_count": len(turns),
                "word_assignment_method": (
                    policy_result.audit["word_assignment_method"] if policy_result is not None else None
                ),
                "policy_id": policy_result.audit["policy_id"] if policy_result is not None else None,
                "language_policy": policy_result.audit["language_policy"] if policy_result is not None else None,
                "policy_changed_word_count": (
                    policy_result.audit["changed_word_count"] if policy_result is not None else 0
                ),
                "sortformer_config": SORTFORMER_CONFIG,
            },
            "speaker_map": speaker_map_provider_payload(speaker_map_result, enabled=options.diarize),
            "polish": {
                "base_url": config.llm_base_url,
                "model": config.polish_model,
                "enabled": options.polish,
                "status": _polish_status(options=options, turns=turns, polish_result=polish_result),
                "prompt_template_id": polish_result.prompt_template_id if polish_result is not None else None,
                "prompt_tokens": polish_result.prompt_tokens if polish_result is not None else 0,
                "completion_tokens": polish_result.completion_tokens if polish_result is not None else 0,
                "issue_count": len(polish_result.issues) if polish_result is not None else 0,
                "chunk_count": polish_result.chunk_count if polish_result is not None else 0,
                "max_workers": polish_result.max_workers if polish_result is not None else config.polish_max_workers,
                "effective_workers": polish_result.effective_workers if polish_result is not None else 0,
            },
        },
        "transcript": {
            "language": asr_result.language,
            "segment_count": len(asr_result.segments),
            "word_count": len(asr_result.words),
            "turn_count": len(turns),
            "asr_terms": list(asr_result.terms),
        },
        "outputs": {
            "transcript_md": str(transcript_path),
        },
    }


def _polish_status(*, options: TranscribeOptions, turns: list[Turn], polish_result: PolishResult | None) -> str:
    if not options.polish:
        return "disabled"
    if not turns:
        return "skipped_no_turns"
    if polish_result is None:
        return "failed"
    return polish_result.status


def _diarization_status(*, options: TranscribeOptions, diarization_result: DiarizationResult | None) -> str:
    if not options.diarize:
        return "disabled"
    if diarization_result is None:
        return "failed"
    return "succeeded"


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
