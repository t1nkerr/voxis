from __future__ import annotations

import json
import math
import sys
import types
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from pytest import MonkeyPatch

from voxis import asr
from voxis.audio import LoadedAudio
from voxis.cli import build_parser, main
from voxis.diarize import DiarizationResult
from voxis.diarization_types import DiarizeSpan
from voxis.llm import LLMResponse
from voxis.polish import PolishResult


def test_cli_parser_has_program_name() -> None:
    parser = build_parser()

    assert parser.prog == "voxis"


def test_doctor_cli_forwards_strict_diarization_mode(monkeypatch: MonkeyPatch) -> None:
    calls: list[bool] = []
    monkeypatch.setattr("voxis.cli.run_doctor", lambda *, require_diarize: calls.append(require_diarize) or 0)

    assert main(["doctor", "--require-diarize"]) == 0
    assert calls == [True]


def test_transcribe_writes_phase_one_outputs(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    install_fake_mlx(monkeypatch)
    install_fake_devices(monkeypatch)
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fake audio")
    out = tmp_path / "out"
    monkeypatch.setenv("BAILIAN_API_KEY", "")

    assert main(["transcribe", "--audio", str(audio), "--out", str(out), "--debug"]) == 0

    transcript = (out / "transcript.md").read_text(encoding="utf-8")
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert transcript.startswith("# Transcript\n")
    assert "Hello Claude Code" in transcript
    assert run["schema_version"] == "voxis_run_v1"
    assert run["status"] == "succeeded"
    assert run["audio"]["size_bytes"] == len(b"fake audio")
    assert run["audio"]["duration_seconds"] == 1.0
    assert run["options"]["language"] == "auto"
    assert run["options"]["debug"] is True
    assert run["providers"]["asr"]["asr_model_id"] == "mlx-community/Qwen3-ASR-1.7B-8bit"
    assert run["providers"]["asr"]["device"]["selected"] == "Device(gpu, 0)"
    assert run["providers"]["alignment"]["device"]["selected"] == "Device(gpu, 0)"
    assert run["providers"]["diarization"]["enabled"] is False
    assert run["providers"]["diarization"]["backend"] == "sortformer"
    assert run["providers"]["diarization"]["status"] == "disabled"
    assert (out / "_debug").is_dir()
    assert (out / "_debug" / "asr_segments.json").exists()
    progress = [
        json.loads(line) for line in (out / "_debug" / "progress.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [(event["stage"], event["event"]) for event in progress] == [
        ("run", "started"),
        ("load_audio", "started"),
        ("load_audio", "done"),
        ("asr", "started"),
        ("asr", "chunk_done"),
        ("asr", "done"),
        ("alignment", "started"),
        ("alignment", "done"),
        ("outputs", "started"),
        ("outputs", "done"),
        ("run", "done"),
    ]
    assert progress[2]["duration_seconds"] == 1.0
    assert progress[4]["chunk_count"] == 1
    assert progress[4]["token_cap_hit"] is False
    assert progress[5]["segment_count"] == 1
    assert progress[7]["word_count"] == 1
    stderr = capsys.readouterr().err
    assert "voxis asr started" in stderr
    assert "voxis run done" in stderr


def test_transcribe_rejects_remote_audio(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["transcribe", "--audio", "https://example.com/audio.mp3", "--out", str(tmp_path / "out")])

    assert exc_info.value.code == 2
    assert "--audio must be a local file path" in capsys.readouterr().err


def test_transcribe_requires_diarize_for_polish(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fake audio")
    monkeypatch.setenv("BAILIAN_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    with pytest.raises(SystemExit) as exc_info:
        main(["transcribe", "--audio", str(audio), "--out", str(tmp_path / "out"), "--polish"])

    assert exc_info.value.code == 2
    assert "--polish is allowed only with --diarize" in capsys.readouterr().err


def test_transcribe_passes_context_terms_and_language_to_asr(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    calls = install_fake_mlx(monkeypatch, aligner_returns_list=True)
    monkeypatch.setenv("BAILIAN_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("VOXIS_ASR_TERMS", "Claude Code,Claude,Codex,OpenClaw,Kimi,Seedance")
    monkeypatch.setattr(
        "voxis.asr_terms.call_json_schema_llm",
        lambda prompt, *, config, model, schema_name, schema, max_tokens=2048, extra_body=None: LLMResponse(
            '{"terms":["Claude Code","CAR-T"]}',
            elapsed_seconds=0.2,
            prompt_tokens=8,
            completion_tokens=4,
        ),
    )
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fake audio")
    out = tmp_path / "out"

    assert (
        main(
            [
                "transcribe",
                "--audio",
                str(audio),
                "--out",
                str(out),
                "--language",
                "en",
                "--context",
                "Claude Code and CAR-T",
            ]
        )
        == 0
    )

    assert calls["asr"]["language"] == "en"
    assert calls["split_calls"][0]["chunk_duration"] == 60.0
    assert calls["asr"]["max_tokens"] == asr.MIN_CHUNK_MAX_TOKENS
    assert calls["asr"]["system_prompt"] == "Claude Code\nClaude\nCodex\nOpenClaw\nKimi\nSeedance\nCAR-T"
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["transcript"]["asr_terms"] == ["Claude Code", "Claude", "Codex", "OpenClaw", "Kimi", "Seedance", "CAR-T"]
    assert run["providers"]["asr"]["term_extraction"]["status"] == "succeeded"
    assert run["providers"]["asr"]["asr_prompt"]["mode"] == "bare_word_list"
    assert run["providers"]["asr"]["asr_prompt"]["global_terms"] == [
        "Claude Code",
        "Claude",
        "Codex",
        "OpenClaw",
        "Kimi",
        "Seedance",
    ]
    assert run["providers"]["asr"]["asr_prompt"]["extracted_terms"] == ["Claude Code", "CAR-T"]
    assert run["providers"]["asr"]["prompt_echo"]["status"] == "passed"


def test_transcribe_trims_asr_prompt_echo_tail(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    calls = install_fake_mlx(
        monkeypatch,
        aligner_returns_list=True,
        asr_outputs=[
            [
                {"start": 0.0, "end": 0.5, "text": "Real speech", "language": "English"},
                {"start": 0.5, "end": 1.0, "text": "Claude Code CAR-T LangChain Manus", "language": "English"},
            ],
        ],
    )
    monkeypatch.setenv("BAILIAN_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("VOXIS_ASR_TERMS", "Claude Code,Claude,Codex,OpenClaw,Kimi,Seedance")
    monkeypatch.setattr(
        "voxis.asr_terms.call_json_schema_llm",
        lambda prompt, *, config, model, schema_name, schema, max_tokens=2048, extra_body=None: LLMResponse(
            '{"terms":["Claude Code","CAR-T","LangChain","Manus"]}',
            elapsed_seconds=0.2,
            prompt_tokens=8,
            completion_tokens=4,
        ),
    )
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fake audio")
    out = tmp_path / "out"

    assert main(["transcribe", "--audio", str(audio), "--out", str(out), "--language", "en", "--context", "terms"]) == 0

    assert len(calls["asr_calls"]) == 1
    assert (
        calls["asr_calls"][0]["system_prompt"]
        == "Claude Code\nClaude\nCodex\nOpenClaw\nKimi\nSeedance\nCAR-T\nLangChain\nManus"
    )
    transcript = (out / "transcript.md").read_text(encoding="utf-8")
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert "Real speech" in transcript
    assert "Claude Code CAR-T LangChain Manus" not in transcript
    assert run["providers"]["asr"]["asr_prompt"]["trimmed_segment_count"] == 1
    assert run["providers"]["asr"]["prompt_echo"]["status"] == "detected"
    assert run["providers"]["asr"]["prompt_echo"]["suspect_segments"][0]["index"] == 1


def test_transcribe_fails_when_asr_output_is_only_prompt_echo(
    tmp_path: Path, monkeypatch: MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    install_fake_mlx(
        monkeypatch,
        asr_outputs=[
            [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "Claude Code Claude Codex OpenClaw Kimi Seedance",
                    "language": "English",
                }
            ],
        ],
    )
    monkeypatch.setenv("BAILIAN_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(
        "voxis.asr_terms.call_json_schema_llm",
        lambda prompt, *, config, model, schema_name, schema, max_tokens=2048, extra_body=None: LLMResponse(
            '{"terms":[]}',
            elapsed_seconds=0.2,
            prompt_tokens=8,
            completion_tokens=4,
        ),
    )
    monkeypatch.setenv("VOXIS_ASR_TERMS", "Claude Code,Claude,Codex,OpenClaw,Kimi,Seedance")
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fake audio")
    out = tmp_path / "out"

    assert main(["transcribe", "--audio", str(audio), "--out", str(out), "--language", "en", "--context", "terms"]) == 1

    stderr = capsys.readouterr().err
    assert "voxis: error: ASR output contains only prompt echo" in stderr
    assert "Traceback" not in stderr
    assert not (out / "transcript.md").exists()
    assert not (out / "run.json").exists()


def test_bare_voxis_prints_help_and_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 2

    assert "usage: voxis" in capsys.readouterr().out


def test_transcribe_runtime_error_exits_cleanly(
    tmp_path: Path, monkeypatch: MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake_transcribe(options, config):  # type: ignore[no-untyped-def]
        raise RuntimeError("boom")

    monkeypatch.setattr("voxis.cli.transcribe", fake_transcribe)
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fake audio")

    assert main(["transcribe", "--audio", str(audio), "--out", str(tmp_path / "out")]) == 1

    stderr = capsys.readouterr().err
    assert "voxis: error: boom" in stderr
    assert "Traceback" not in stderr


def test_transcribe_runtime_error_prints_traceback_with_debug(
    tmp_path: Path, monkeypatch: MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake_transcribe(options, config):  # type: ignore[no-untyped-def]
        raise RuntimeError("boom")

    monkeypatch.setattr("voxis.cli.transcribe", fake_transcribe)
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fake audio")

    assert main(["transcribe", "--audio", str(audio), "--out", str(tmp_path / "out"), "--debug"]) == 1

    stderr = capsys.readouterr().err
    assert "voxis: error: boom" in stderr
    assert "Traceback (most recent call last)" in stderr
    assert "RuntimeError: boom" in stderr


def test_failed_run_removes_stale_outputs(
    tmp_path: Path, monkeypatch: MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    install_fake_mlx(
        monkeypatch,
        asr_outputs=[
            [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "Claude Code Claude Codex OpenClaw Kimi Seedance",
                    "language": "English",
                }
            ],
        ],
    )
    monkeypatch.setenv("BAILIAN_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(
        "voxis.asr_terms.call_json_schema_llm",
        lambda prompt, *, config, model, schema_name, schema, max_tokens=2048, extra_body=None: LLMResponse(
            '{"terms":[]}',
            elapsed_seconds=0.2,
            prompt_tokens=8,
            completion_tokens=4,
        ),
    )
    monkeypatch.setenv("VOXIS_ASR_TERMS", "Claude Code,Claude,Codex,OpenClaw,Kimi,Seedance")
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fake audio")
    out = tmp_path / "out"
    out.mkdir()
    (out / "transcript.md").write_text("stale transcript\n", encoding="utf-8")
    (out / "run.json").write_text('{"status":"stale"}\n', encoding="utf-8")
    stale_debug = out / "_debug"
    stale_debug.mkdir()
    (stale_debug / "asr_segments.json").write_text("{}\n", encoding="utf-8")

    assert main(["transcribe", "--audio", str(audio), "--out", str(out), "--language", "en", "--context", "terms"]) == 1

    assert "voxis: error:" in capsys.readouterr().err
    assert not (out / "transcript.md").exists()
    assert not (out / "run.json").exists()
    assert not stale_debug.exists()


def test_asr_truncation_gap_fails_loudly(
    tmp_path: Path, monkeypatch: MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    install_fake_mlx(monkeypatch)

    def fake_load_audio_file(path: Path, *, sample_rate_hz: int) -> LoadedAudio:
        return LoadedAudio(
            path=path,
            sample_rate_hz=sample_rate_hz,
            waveform=np.zeros(300 * sample_rate_hz, dtype=np.float32),
            duration_s=300.0,
        )

    monkeypatch.setattr("voxis.run.load_audio_file", fake_load_audio_file)
    audio = tmp_path / "long.wav"
    audio.write_bytes(b"fake audio")
    out = tmp_path / "out"

    assert main(["transcribe", "--audio", str(audio), "--out", str(out)]) == 1

    stderr = capsys.readouterr().err
    assert "voxis: error: ASR transcript appears truncated" in stderr
    assert not (out / "transcript.md").exists()


def test_truncation_gap_thresholds() -> None:
    assert asr._truncation_gap_seconds([{"end": 0.5}], audio_duration_s=300.0) == pytest.approx(299.5)
    assert asr._truncation_gap_seconds([{"end": 0.5}], audio_duration_s=100.0) is None
    assert asr._truncation_gap_seconds([{"end": 700.0}], audio_duration_s=1000.0) == pytest.approx(300.0)
    assert asr._truncation_gap_seconds([{"end": 850.0}], audio_duration_s=1000.0) is None
    assert asr._truncation_gap_seconds([], audio_duration_s=1000.0) is None


def test_transcribe_writes_plain_transcript_when_alignment_fails(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    install_fake_mlx(monkeypatch, fail_alignment=True)
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fake audio")
    out = tmp_path / "out"

    assert main(["transcribe", "--audio", str(audio), "--out", str(out), "--debug"]) == 0

    transcript = (out / "transcript.md").read_text(encoding="utf-8")
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    debug = json.loads((out / "_debug" / "asr_segments.json").read_text(encoding="utf-8"))
    assert "Hello Claude Code" in transcript
    assert run["providers"]["alignment"]["status"] == "failed"
    assert run["providers"]["alignment"]["error_count"] == 1
    assert debug["alignment_errors"][0]["error_type"] == "RuntimeError"


def test_diarized_transcribe_fails_before_diarization_when_alignment_fails(
    tmp_path: Path, monkeypatch: MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = install_fake_mlx(monkeypatch, fail_alignment=True)
    diarization_called = False

    def unexpected_diarization(_audio: LoadedAudio) -> DiarizationResult:
        nonlocal diarization_called
        diarization_called = True
        raise AssertionError("diarization must not run after an unresolved alignment error")

    monkeypatch.setattr("voxis.run.diarize_audio", unexpected_diarization)
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fake audio")
    out = tmp_path / "out"

    assert main(["transcribe", "--audio", str(audio), "--out", str(out), "--diarize", "--debug"]) == 1

    assert calls["aligner_calls"]
    assert diarization_called is False
    assert "diarization requires complete word timestamps" in capsys.readouterr().err
    assert not (out / "transcript.md").exists()
    assert not (out / "run.json").exists()
    progress = [
        json.loads(line) for line in (out / "_debug" / "progress.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(
        event["stage"] == "alignment" and event["event"] == "failed" and event["reason"] == "unresolved_segment_errors"
        for event in progress
    )


def test_degenerate_asr_segment_retries_once_then_fails_before_alignment(
    tmp_path: Path, monkeypatch: MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    degenerate_text = "正常开头" + ("对" * 40_370)
    calls = install_fake_mlx(
        monkeypatch,
        asr_outputs=[
            [{"start": 0.0, "end": 64.718, "text": degenerate_text, "language": "Chinese"}],
            [{"start": 0.0, "end": 64.718, "text": degenerate_text, "language": "Chinese"}],
        ],
    )
    monkeypatch.setattr("voxis.run.load_audio_file", _fake_loaded_audio(tmp_path, duration_seconds=64.718))
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fake audio")
    out = tmp_path / "out"

    assert main(["transcribe", "--audio", str(audio), "--out", str(out), "--language", "zh", "--debug"]) == 1

    assert len(calls["asr_calls"]) == 2
    assert calls["split_calls"][1]["chunk_duration"] == 30.0
    assert calls["asr_calls"][1]["max_tokens"] == math.ceil(64.718 * asr.MAX_TOKENS_PER_SECOND)
    assert "system_prompt" not in calls["asr_calls"][1]
    assert "aligner_calls" not in calls
    assert "ASR degenerate segment persisted after retry" in capsys.readouterr().err
    assert not (out / "transcript.md").exists()
    assert not (out / "run.json").exists()
    progress = [
        json.loads(line) for line in (out / "_debug" / "progress.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    detected = next(event for event in progress if event["event"] == "degenerate_segment_detected")
    retried = next(event for event in progress if event["event"] == "segment_retry_done")
    failed = next(event for event in progress if event.get("reason") == "degenerate_segment")
    assert detected["metrics"]["text_chars"] == len(degenerate_text)
    assert detected["metrics"]["dominant_char"] == "对"
    assert retried["metrics"][0]["suspect"] is True
    assert failed["initial_metrics"]["suspect"] is True


def test_clean_segment_retry_replaces_only_suspect_span_with_absolute_timestamps(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    calls = install_fake_mlx(
        monkeypatch,
        asr_outputs=[
            [
                {"start": 0.0, "end": 10.0, "text": "before", "language": "English"},
                {"start": 10.0, "end": 20.0, "text": "x" * 2_000, "language": "English"},
                {"start": 20.0, "end": 30.0, "text": "after", "language": "English"},
            ],
            [
                {"start": 0.0, "end": 4.0, "text": "recovered first", "language": "English"},
                {"start": 4.0, "end": 10.0, "text": "recovered second", "language": "English"},
            ],
        ],
    )
    monkeypatch.setattr("voxis.run.load_audio_file", _fake_loaded_audio(tmp_path, duration_seconds=30.0))
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fake audio")
    out = tmp_path / "out"

    assert main(["transcribe", "--audio", str(audio), "--out", str(out), "--debug"]) == 0

    debug = json.loads((out / "_debug" / "asr_segments.json").read_text(encoding="utf-8"))
    assert [(segment["start"], segment["end"], segment["text"]) for segment in debug["segments"]] == [
        (0.0, 10.0, "before"),
        (10.0, 14.0, "recovered first"),
        (14.0, 20.0, "recovered second"),
        (20.0, 30.0, "after"),
    ]
    assert len(debug["degeneration_retries"]) == 1
    assert len(calls["asr_calls"]) == 2
    assert calls["asr_waveform_lengths"] == [30 * asr.SAMPLE_RATE_HZ, 10 * asr.SAMPLE_RATE_HZ]
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["providers"]["asr"]["degeneration_guard"]["retry_count"] == 1


def test_first_pass_caps_tokens_per_chunk_offsets_segments_and_pins_detected_language(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    calls = install_fake_mlx(
        monkeypatch,
        asr_outputs=[
            [{"start": 0.0, "end": 60.0, "text": "first chunk", "language": "English"}],
            [{"start": 0.0, "end": 30.0, "text": "second chunk", "language": "English"}],
        ],
        split_chunk_seconds=60.0,
    )
    monkeypatch.setattr("voxis.run.load_audio_file", _fake_loaded_audio(tmp_path, duration_seconds=90.0))
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fake audio")
    out = tmp_path / "out"

    assert main(["transcribe", "--audio", str(audio), "--out", str(out), "--debug"]) == 0

    assert [call["max_tokens"] for call in calls["asr_calls"]] == [
        math.ceil(60.0 * asr.MAX_TOKENS_PER_SECOND),
        math.ceil(30.0 * asr.MAX_TOKENS_PER_SECOND),
    ]
    assert all(call["chunk_duration"] == math.inf for call in calls["asr_calls"])
    assert [call["language"] for call in calls["asr_calls"]] == [None, "English"]
    debug = json.loads((out / "_debug" / "asr_segments.json").read_text(encoding="utf-8"))
    assert [(segment["start"], segment["end"]) for segment in debug["segments"]] == [(0.0, 60.0), (60.0, 90.0)]
    progress = [
        json.loads(line) for line in (out / "_debug" / "progress.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    chunk_events = [event for event in progress if event["event"] == "chunk_done"]
    assert [(event["chunk_index"], event["chunk_count"], event["chunk_start"]) for event in chunk_events] == [
        (0, 2, 0.0),
        (1, 2, 60.0),
    ]


def test_token_cap_hit_routes_low_density_chunk_to_retry(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    capped_tokens = math.ceil(60.0 * asr.MAX_TOKENS_PER_SECOND)
    calls = install_fake_mlx(
        monkeypatch,
        asr_outputs=[
            [{"start": 0.0, "end": 60.0, "text": "real speech then 用用用", "language": "en"}],
            [{"start": 0.0, "end": 60.0, "text": "recovered speech", "language": "en"}],
        ],
        asr_generation_tokens=[capped_tokens, 4],
    )
    monkeypatch.setattr("voxis.run.load_audio_file", _fake_loaded_audio(tmp_path, duration_seconds=60.0))
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fake audio")
    out = tmp_path / "out"

    assert main(["transcribe", "--audio", str(audio), "--out", str(out), "--language", "en", "--debug"]) == 0

    assert len(calls["asr_calls"]) == 2
    debug = json.loads((out / "_debug" / "asr_segments.json").read_text(encoding="utf-8"))
    assert [segment["text"] for segment in debug["segments"]] == ["recovered speech"]
    retry = debug["degeneration_retries"][0]
    assert retry["initial_metrics"]["token_cap_hit"] is True
    assert retry["initial_metrics"]["suspect"] is True
    assert retry["retry_metrics"][0]["token_cap_hit"] is False


def test_retry_with_fractional_chunk_boundaries_stays_contiguous(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    retry_chunk_seconds = [25.0506875, 25.0506875, 9.898625]
    install_fake_mlx(
        monkeypatch,
        asr_outputs=[
            [{"start": 0.0, "end": 60.0, "text": "looping", "language": "en"}],
            *[[{"start": 0.0, "end": seconds, "text": "clean", "language": "en"}] for seconds in retry_chunk_seconds],
        ],
        asr_generation_tokens=[math.ceil(60.0 * asr.MAX_TOKENS_PER_SECOND), 4, 4, 4],
        split_plans=[[60.0], retry_chunk_seconds],
    )
    monkeypatch.setattr("voxis.run.load_audio_file", _fake_loaded_audio(tmp_path, duration_seconds=60.0))
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fake audio")
    out = tmp_path / "out"

    assert main(["transcribe", "--audio", str(audio), "--out", str(out), "--language", "en", "--debug"]) == 0

    debug = json.loads((out / "_debug" / "asr_segments.json").read_text(encoding="utf-8"))
    assert [(segment["start"], segment["end"]) for segment in debug["segments"]] == [
        (0.0, 25.051),
        (25.051, 50.101),
        (50.101, 60.0),
    ]
    assert debug["degeneration_retries"][0]["coverage_metrics"]["complete"] is True


def test_token_cap_hit_on_retry_fails_closed(
    tmp_path: Path, monkeypatch: MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    install_fake_mlx(
        monkeypatch,
        asr_outputs=[[{"start": 0.0, "end": 60.0, "text": "still looping", "language": "en"}]],
        asr_generation_tokens=[10_000, 10_000],
    )
    monkeypatch.setattr("voxis.run.load_audio_file", _fake_loaded_audio(tmp_path, duration_seconds=60.0))
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fake audio")
    out = tmp_path / "out"

    assert main(["transcribe", "--audio", str(audio), "--out", str(out), "--language", "en", "--debug"]) == 1

    assert "ASR degenerate segment persisted after retry" in capsys.readouterr().err
    assert not (out / "transcript.md").exists()


def test_segment_retry_rejects_truncated_first_chunk(
    tmp_path: Path, monkeypatch: MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = install_fake_mlx(
        monkeypatch,
        asr_outputs=[
            [{"start": 0.0, "end": 64.718, "text": "对的" * 20_000, "language": "Chinese"}],
            [{"start": 0.0, "end": 30.0, "text": "clean but incomplete retry", "language": "English"}],
        ],
    )
    monkeypatch.setattr("voxis.run.load_audio_file", _fake_loaded_audio(tmp_path, duration_seconds=64.718))
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fake audio")
    out = tmp_path / "out"

    assert main(["transcribe", "--audio", str(audio), "--out", str(out), "--debug"]) == 1

    assert len(calls["asr_calls"]) == 2
    assert "ASR degenerate segment persisted after retry" in capsys.readouterr().err
    progress = [
        json.loads(line) for line in (out / "_debug" / "progress.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    retried = next(event for event in progress if event["event"] == "segment_retry_done")
    failed = next(event for event in progress if event.get("reason") == "degenerate_segment")
    assert retried["coverage_metrics"] == {
        "start_gap_seconds": 0.0,
        "end_gap_seconds": 34.718,
        "maximum_internal_gap_seconds": 0.0,
        "total_uncovered_seconds": 34.718,
        "ordered_non_overlapping": True,
        "complete": False,
    }
    assert failed["coverage_metrics"]["complete"] is False
    assert not (out / "transcript.md").exists()
    assert not (out / "run.json").exists()


@pytest.mark.parametrize(
    ("segments", "expected_total_uncovered", "expected_ordered"),
    [
        ([{"start": 0.11, "end": 9.89}], 0.22, True),
        ([{"start": 0.0, "end": 4.9}, {"start": 5.11, "end": 10.0}], 0.21, True),
        (
            [
                {"start": 0.0, "end": 2.0},
                {"start": 2.05, "end": 4.0},
                {"start": 4.05, "end": 6.0},
                {"start": 6.05, "end": 10.0},
            ],
            0.15,
            True,
        ),
        ([{"start": 0.0, "end": 6.0}, {"start": 5.9, "end": 10.0}], 0.0, False),
    ],
)
def test_retry_coverage_rejects_edge_internal_accumulated_gaps_and_overlap(
    segments: list[dict[str, float]], expected_total_uncovered: float, expected_ordered: bool
) -> None:
    metrics = asr._retry_coverage_metrics(segments, span_start=0.0, span_end=10.0)

    assert metrics["total_uncovered_seconds"] == pytest.approx(expected_total_uncovered)
    assert metrics["ordered_non_overlapping"] is expected_ordered
    assert metrics["complete"] is False


def test_segment_retry_rejects_empty_output(
    tmp_path: Path, monkeypatch: MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    install_fake_mlx(
        monkeypatch,
        asr_outputs=[
            [{"start": 0.0, "end": 20.0, "text": "对的" * 5_000, "language": "Chinese"}],
            [],
        ],
    )
    monkeypatch.setattr("voxis.run.load_audio_file", _fake_loaded_audio(tmp_path, duration_seconds=20.0))
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fake audio")
    out = tmp_path / "out"

    assert main(["transcribe", "--audio", str(audio), "--out", str(out), "--debug"]) == 1

    assert "ASR degenerate segment persisted after retry" in capsys.readouterr().err
    progress = [
        json.loads(line) for line in (out / "_debug" / "progress.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    failed = next(event for event in progress if event.get("reason") == "degenerate_segment")
    assert failed["retry_metrics"] == []
    assert failed["coverage_metrics"]["complete"] is False


def test_segment_retry_provider_exception_reports_structured_reason(
    tmp_path: Path, monkeypatch: MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    install_fake_mlx(
        monkeypatch,
        asr_outputs=[[{"start": 0.0, "end": 20.0, "text": "这个很重要" * 2_000, "language": "Chinese"}]],
        fail_asr_on_call=2,
    )
    monkeypatch.setattr("voxis.run.load_audio_file", _fake_loaded_audio(tmp_path, duration_seconds=20.0))
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fake audio")
    out = tmp_path / "out"

    assert main(["transcribe", "--audio", str(audio), "--out", str(out), "--debug"]) == 1

    assert "ASR degenerate segment retry failed" in capsys.readouterr().err
    progress = [
        json.loads(line) for line in (out / "_debug" / "progress.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    failed = next(event for event in progress if event.get("reason") == "degenerate_segment")
    assert failed["retry_error_type"] == "RuntimeError"
    assert failed["retry_error_message"] == "ASR provider retry failed"


@pytest.mark.parametrize("loop", ["对的", "这个很重要"])
def test_segment_degeneration_gate_detects_repeated_sequences(loop: str) -> None:
    text = loop * ((40_000 // len(loop)) + 1)

    metrics = asr._segment_degeneration_metrics({"start": 0.0, "end": 64.718, "text": text[:40_000]})

    assert metrics["chars_per_second"] > 600.0
    assert metrics["repeated_sequence_unit_chars"] >= len(loop)
    assert metrics["repeated_sequence_coverage"] >= 0.99
    assert metrics["suspect"] is True


@pytest.mark.parametrize(
    "text",
    [
        (
            "欢迎来到高能量，我是李翔。然后这一期我们是接着和李峰峰叔做宏观漫谈。可能上一期的节目大家也已经听到了，峰叔刚跟我说他有进一步搞明白了其中一些上一次还没有特别搞明白的地方。"
            "过程中间因为我们聊的时候，已经对富途和老虎这样的机构发了一个监管的新政策。国务院又发了一个包括个人在内的对外投资规定。我当时也比较困惑，到底这些政策的连续性和起因是什么。"
            "通常国家出台特定政策，今天还不能叫法律，因为它其实是国务院关于个人和机构对海外进行投资的规定。这里边有几个事情非常重要，也解释了我之前的一些困惑。第一个是从现象来看比较少见，因为这是国务院发的。"
            "以前这种执行通知几乎都是所属部门，比如发改委、经管局、银监会、证监会等机构来发对应通知。这次用了国务院统筹，几乎所有相关单位都由国务院进行发布。通常在法律或媒体解释上，它被称为上位法。"
            "简单来讲，最上层机构做了约束的方向和框架，然后具体的司、部、委再对各自管辖范围如何执行上位政策进行细则颁布。现在由最高政府执行单位对整件事情的方向和趋势做出规范和约束，并确定边界。"
            "范围既包括个人，也包括企业和相关实体，约束的是海外投资。这解释了很多小的迹象，有些跟行业相关，有些跟大家看到的新闻现象有关。过去两年很多件事都跟这个有关系。"
            "当全社会讨论恒大地产问题时，有一个炒得沸沸扬扬的新闻，关于海外家族信托是否参与赔偿和是否会被击穿。中国境内个人的海外信托涉及很多问题，比如钱是怎么出境的，因为什么来的。"
            "这些钱是通过正规途径出境，还是通过真实或虚拟的海外投资转成。另一个新闻事件是娃哈哈家族信托的问题。当时讨论的背景，是信托会不会参与欠款、分家产的赔偿和支付。"
            "从政策层面往回追溯，关心的是这些钱如何形成、如何出境，以及以什么目的和方式出境。第二波事情跟股权投资行业有关，因为行业里有三十七号文，也就是外管局登记。"
            "人民币股权投资之所以跟境外投资有关，是因为人民币基金投资海外企业并持有股权时，需要人民币换汇出境。这个过程涉及备案审批，也包括 ODI，也就是海外直接投资。"
            "今年初有一个窗口指导，通知从二三年开始还没有办完的 ODI 程序暂停，同时说明不是以后都暂停，而是因为可能有政策要颁布，具体规定还不确定。"
            "当时收到的说明是，政策如何规定还没有完全确定，所以先暂停执行尚未办完的程序。这个窗口指导和后来发布的整体规定彼此对应，也说明相关准备并不是一天两天形成的。"
            "从行业角度看，真正需要判断的是后续细则如何落地、不同审批单位怎样执行，以及个人和企业在新的边界下分别需要准备哪些材料。"
        ),
        (
            "And what's going on? He said the following: If you are on stage or in an interview and talking "
            "about something, you should know that topic inside out already. The only thing you should ever be "
            "talking about is something you know intimately. If you know it intimately, you should be able to "
            "talk about it in a viscerally interesting way that relays your thoughts, and you ought to come "
            "across as an interesting person because it is like sitting across the table talking to a friend. "
            "His training is one hundred percent about getting you to stop doing the other rehearsed material "
            "and become able to do that. The other part is that you assume the media is adversarial. He called "
            "the next technique the pivot, which means you do not answer a bad question directly; you answer "
            "the useful question you came to discuss. That can make an interview feel unnatural, but natural "
            "conversation also means refusing a false premise and substituting a better framing. We have watched "
            "the evolution of the new media landscape for thirty years. If you watch Palmer Luckey, Alex Karp, "
            "Elon Musk, or any of the strongest communicators, that is what they are doing. They know the subject, "
            "choose the useful question, explain the evidence, and keep the conversation grounded in what they "
            "can actually defend. This is not memorizing a slogan. It is understanding the material well enough "
            "to move between examples, objections, history, incentives, and consequences without losing the main "
            "argument. A prepared speaker can still sound spontaneous because preparation supplies knowledge, "
            "not a rigid script. The interviewer may change direction, introduce a misleading assumption, or ask "
            "for a prediction that cannot be supported. The speaker then has to identify the real issue, explain "
            "why the premise is incomplete, and offer a precise answer to the question that matters. That is the "
            "difference between evasion and a useful pivot: one hides from evidence, while the other redirects the "
            "conversation toward evidence the audience can evaluate."
        ),
    ],
)
def test_segment_degeneration_gate_keeps_corpus_derived_high_density_text(text: str) -> None:
    duration = len(text.strip()) / 30.0

    metrics = asr._segment_degeneration_metrics({"start": 0.0, "end": duration, "text": text})

    assert metrics["text_chars"] >= asr.DEGENERATION_MIN_TEXT_CHARS
    assert metrics["chars_per_second"] == pytest.approx(30.0)
    assert metrics["repeated_sequence_coverage"] < asr.DEGENERATION_MIN_REPEATED_SEQUENCE_COVERAGE
    assert metrics["suspect"] is False


@pytest.mark.parametrize(
    ("text", "tokens", "expected"),
    [
        (
            'He said, "lost about 40% of the value." Then 60-70% — gone.',
            ["He", "said", "lost", "about", "40", "of", "the", "value", "Then", "6070", "gone"],
            [
                ("He", None),
                ("said,", True),
                ('"lost', True),
                ("about", True),
                ("40%", True),
                ("of", True),
                ("the", True),
                ('value."', True),
                ("Then", True),
                ("60-70% —", True),
                ("gone.", True),
            ],
        ),
        (
            "我们用Claude，大概40%。",
            ["我", "们", "用", "Claude", "大", "概", "40"],
            [
                ("我", None),
                ("们", False),
                ("用", False),
                ("Claude，", False),
                ("大", False),
                ("概", False),
                ("40%。", False),
            ],
        ),
    ],
)
def test_surface_words_restore_asr_punctuation(
    text: str, tokens: list[str], expected: list[tuple[str, bool | None]]
) -> None:
    assert asr._surface_words(text, tokens) == expected


@pytest.mark.parametrize(
    ("text", "tokens"),
    [
        ("Hello Claude Code", ["Hello"]),
        ("Hello world", ["Hello", "word"]),
    ],
)
def test_surface_words_reject_tokens_that_do_not_cover_the_text(text: str, tokens: list[str]) -> None:
    assert asr._surface_words(text, tokens) is None


def test_diarized_transcript_keeps_asr_punctuation(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    install_fake_mlx(
        monkeypatch,
        words=("It", "lost", "about", "40", "of", "it", "Then", "6070"),
        asr_outputs=[[{"start": 0.0, "end": 4.0, "text": "It lost about 40% of it. Then 60-70%.", "language": "en"}]],
    )
    install_fake_diarize(monkeypatch, [DiarizeSpan(0.0, 4.0, "SPEAKER_00")])
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fake audio")
    out = tmp_path / "out"

    assert main(["transcribe", "--audio", str(audio), "--out", str(out), "--diarize"]) == 0

    transcript = (out / "transcript.md").read_text(encoding="utf-8")
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert transcript == "# Transcript\n\n[00:00:00] SPEAKER_00: It lost about 40% of it. Then 60-70%.\n"
    assert run["providers"]["asr"]["profile_id"] == "qwen3_asr_forced_aligner_v2"


def test_transcribe_offsets_aligned_words_by_segment_start(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    install_fake_mlx(
        monkeypatch,
        words=("Hello",),
        word_starts=(0.25,),
        asr_outputs=[[{"start": 10.0, "end": 11.0, "text": "Hello", "language": "English"}]],
    )
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fake audio")
    out = tmp_path / "out"

    assert main(["transcribe", "--audio", str(audio), "--out", str(out), "--debug"]) == 0

    debug = json.loads((out / "_debug" / "asr_segments.json").read_text(encoding="utf-8"))
    assert debug["words"] == [{"text": "Hello", "start": 10.25, "end": 10.5, "space_before": False}]


def test_diarize_writes_speaker_turns(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    install_fake_mlx(monkeypatch, words=("Hello", "world"))
    install_fake_devices(monkeypatch)
    install_fake_diarize(
        monkeypatch,
        [
            DiarizeSpan(start=0.0, end=0.3, speaker="SPEAKER_00"),
            DiarizeSpan(start=0.3, end=1.0, speaker="SPEAKER_01"),
        ],
    )
    monkeypatch.setenv("BAILIAN_API_KEY", "")
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fake audio")
    out = tmp_path / "out"

    assert main(["transcribe", "--audio", str(audio), "--out", str(out), "--diarize", "--debug"]) == 0

    transcript = (out / "transcript.md").read_text(encoding="utf-8")
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    diarization = json.loads((out / "_debug" / "diarization.json").read_text(encoding="utf-8"))
    speaker_map = json.loads((out / "_debug" / "speaker_map.json").read_text(encoding="utf-8"))
    assert "[00:00:00] SPEAKER_00: Hello" in transcript
    assert "[00:00:00] SPEAKER_01: world" in transcript
    assert run["providers"]["diarization"]["enabled"] is True
    assert run["providers"]["diarization"]["backend"] == "sortformer"
    assert run["providers"]["diarization"]["status"] == "succeeded"
    assert run["providers"]["diarization"]["span_count"] == 2
    assert run["providers"]["diarization"]["turn_count"] == 2
    assert run["providers"]["diarization"]["policy_id"] == "sortformer_language_boundary_policy_v1"
    assert run["providers"]["speaker_map"]["status"] == "skipped_no_llm_key"
    assert run["providers"]["speaker_map"]["evidence_mode"] == "probe_only"
    assert run["transcript"]["turn_count"] == 2
    assert diarization["backend"] == "sortformer"
    assert len(diarization["spans"]) == 2
    assert len(diarization["turns"]) == 2
    assert diarization["policy"]["policy_id"] == "sortformer_language_boundary_policy_v1"
    assert speaker_map["status"] == "skipped_no_llm_key"
    assert speaker_map["evidence_mode"] == "probe_only"
    assert len(diarization["words"]) == 2


def test_diarize_runs_probe_only_speaker_map_without_context(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    install_fake_mlx(monkeypatch, words=("Alice", "speaks"))
    install_fake_diarize(monkeypatch, [DiarizeSpan(0.0, 1.0, "SPEAKER_00")])
    monkeypatch.setenv("BAILIAN_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(
        "voxis.speaker_map.call_json_schema_llm",
        lambda prompt, *, config, model, schema_name, schema, max_tokens=2048, extra_body=None: LLMResponse(
            '{"mapping":{"SPEAKER_00": "Alice"}}',
            elapsed_seconds=0.2,
            prompt_tokens=8,
            completion_tokens=4,
        ),
    )
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fake audio")
    out = tmp_path / "out"

    assert main(["transcribe", "--audio", str(audio), "--out", str(out), "--diarize", "--debug"]) == 0

    transcript = (out / "transcript.md").read_text(encoding="utf-8")
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    speaker_map = json.loads((out / "_debug" / "speaker_map.json").read_text(encoding="utf-8"))
    assert transcript == "# Transcript\n\n[00:00:00] Alice: Alice speaks\n"
    assert run["providers"]["speaker_map"]["status"] == "succeeded"
    assert run["providers"]["speaker_map"]["evidence_mode"] == "probe_only"
    assert speaker_map["evidence_mode"] == "probe_only"


def test_diarized_run_loads_audio_once_and_shares_loaded_audio_with_sortformer(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    calls = install_fake_mlx(monkeypatch)
    seen: dict[str, LoadedAudio] = {}

    def fake_diarize(audio: LoadedAudio) -> DiarizationResult:
        seen["diarize"] = audio
        return fake_diarization_result([DiarizeSpan(start=0.0, end=1.0, speaker="SPEAKER_00")])

    monkeypatch.setattr("voxis.run.diarize_audio", fake_diarize)
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fake audio")
    out = tmp_path / "out"

    assert main(["transcribe", "--audio", str(audio), "--out", str(out), "--diarize"]) == 0

    assert len(calls["load_audio"]) == 1
    assert seen["diarize"].path == audio.resolve()


def test_speaker_map_maps_names_before_transcript_output(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    install_fake_mlx(monkeypatch)
    install_fake_diarize(monkeypatch, [DiarizeSpan(0.0, 1.0, "SPEAKER_00")])
    monkeypatch.setenv("BAILIAN_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(
        "voxis.speaker_map.call_json_schema_llm",
        lambda prompt, *, config, model, schema_name, schema, max_tokens=2048, extra_body=None: LLMResponse(
            '{"mapping":{"SPEAKER_00": "Alice"}}',
            elapsed_seconds=0.2,
            prompt_tokens=8,
            completion_tokens=4,
        ),
    )
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fake audio")
    out = tmp_path / "out"

    assert (
        main(
            [
                "transcribe",
                "--audio",
                str(audio),
                "--out",
                str(out),
                "--diarize",
                "--debug",
                "--context",
                "Host: Alice",
            ]
        )
        == 0
    )

    transcript = (out / "transcript.md").read_text(encoding="utf-8")
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    speaker_map = json.loads((out / "_debug" / "speaker_map.json").read_text(encoding="utf-8"))
    assert transcript == "# Transcript\n\n[00:00:00] Alice: Hello\n"
    assert run["providers"]["speaker_map"]["status"] == "succeeded"
    assert run["providers"]["speaker_map"]["thinking_enabled"] is True
    assert speaker_map["mapping"] == {"SPEAKER_00": "Alice"}
    assert speaker_map["thinking_enabled"] is True


def test_duplicate_speaker_map_names_succeed_with_warning(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    install_fake_mlx(monkeypatch, words=("hello", "first", "second"), word_starts=(0.0, 1.0, 2.0))
    install_fake_diarize(
        monkeypatch,
        [
            DiarizeSpan(start=0.0, end=0.8, speaker="SPEAKER_00"),
            DiarizeSpan(start=1.0, end=1.8, speaker="SPEAKER_01"),
            DiarizeSpan(start=2.0, end=2.8, speaker="SPEAKER_02"),
        ],
    )
    monkeypatch.setenv("BAILIAN_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(
        "voxis.speaker_map.call_json_schema_llm",
        lambda prompt, *, config, model, schema_name, schema, max_tokens=2048, extra_body=None: LLMResponse(
            '{"mapping":{"SPEAKER_00":"Alice","SPEAKER_01":"Jared Friedman","SPEAKER_02":"Jared Friedman"}}',
            elapsed_seconds=0.2,
            prompt_tokens=8,
            completion_tokens=4,
        ),
    )
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fake audio")
    out = tmp_path / "out"
    out.mkdir()
    debug_dir = out / "_debug"
    debug_dir.mkdir()
    (out / "transcript.md").write_text("stale transcript\n", encoding="utf-8")
    (out / "run.json").write_text('{"status":"stale"}\n', encoding="utf-8")
    (debug_dir / "asr_segments.json").write_text('{"status":"stale"}\n', encoding="utf-8")

    main(
        [
            "transcribe",
            "--audio",
            str(audio),
            "--out",
            str(out),
            "--diarize",
            "--debug",
            "--context",
            "Host: Alice. Guest: Jared Friedman.",
        ]
    )

    speaker_map = json.loads((out / "_debug" / "speaker_map.json").read_text(encoding="utf-8"))
    progress = [
        json.loads(line) for line in (out / "_debug" / "progress.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    transcript = (out / "transcript.md").read_text(encoding="utf-8")
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert speaker_map["status"] == "succeeded_with_issues"
    assert speaker_map["mapping"] == {
        "SPEAKER_00": "Alice",
        "SPEAKER_01": "Jared Friedman",
        "SPEAKER_02": "Jared Friedman",
    }
    assert [issue["code"] for issue in speaker_map["issues"]] == [
        "DUPLICATE_REAL_NAME",
        "DUPLICATE_REAL_NAME",
    ]
    assert {issue["severity"] for issue in speaker_map["issues"]} == {"warning"}
    assert "[00:00:01] Jared Friedman: first\n" in transcript
    assert "[00:00:02] Jared Friedman: second\n" in transcript
    assert run["providers"]["speaker_map"]["status"] == "succeeded_with_issues"
    assert (out / "_debug" / "asr_segments.json").exists()
    assert ("speaker_map", "done") in [(event["stage"], event["event"]) for event in progress]
    assert progress[-1]["stage"] == "run"
    assert progress[-1]["event"] == "done"


def test_nonliteral_speaker_name_is_preserved_with_warning(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    install_fake_mlx(monkeypatch, words=("hello", "world"), word_starts=(0.0, 1.0))
    install_fake_diarize(
        monkeypatch,
        [
            DiarizeSpan(start=0.0, end=0.8, speaker="SPEAKER_00"),
            DiarizeSpan(start=1.0, end=1.8, speaker="SPEAKER_01"),
        ],
    )
    monkeypatch.setenv("BAILIAN_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(
        "voxis.speaker_map.call_json_schema_llm",
        lambda prompt, *, config, model, schema_name, schema, max_tokens=2048, extra_body=None: LLMResponse(
            '{"mapping":{"SPEAKER_00":"Alice","SPEAKER_01":"Peter Yang"}}',
            elapsed_seconds=0.2,
            prompt_tokens=8,
            completion_tokens=4,
        ),
    )
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fake audio")
    out = tmp_path / "out"

    assert (
        main(
            [
                "transcribe",
                "--audio",
                str(audio),
                "--out",
                str(out),
                "--diarize",
                "--debug",
                "--context",
                "Host: Alice.",
            ]
        )
        == 0
    )

    speaker_map = json.loads((out / "_debug" / "speaker_map.json").read_text(encoding="utf-8"))
    progress = [
        json.loads(line) for line in (out / "_debug" / "progress.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    transcript = (out / "transcript.md").read_text(encoding="utf-8")
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert speaker_map["status"] == "succeeded_with_issues"
    assert speaker_map["mapping"] == {"SPEAKER_00": "Alice", "SPEAKER_01": "Peter Yang"}
    assert speaker_map["issues"] == [
        {
            "code": "NAME_NOT_LITERAL",
            "severity": "warning",
            "detail": "name does not appear verbatim in context or transcript probe; preserving model output",
            "speaker": "SPEAKER_01",
            "value": "Peter Yang",
        }
    ]
    assert "[00:00:01] Peter Yang: world\n" in transcript
    assert run["providers"]["speaker_map"]["status"] == "succeeded_with_issues"
    assert ("speaker_map", "done") in [(event["stage"], event["event"]) for event in progress]
    assert progress[-1]["stage"] == "run"
    assert progress[-1]["event"] == "done"


def test_speaker_map_failure_falls_back_to_unmapped_turns(
    tmp_path: Path, monkeypatch: MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    install_fake_mlx(monkeypatch)
    install_fake_diarize(monkeypatch, [DiarizeSpan(0.0, 1.0, "SPEAKER_00")])
    monkeypatch.setenv("BAILIAN_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    extra_keys = ",".join(f'"bad_{index}":"noise"' for index in range(6))
    monkeypatch.setattr(
        "voxis.speaker_map.call_json_schema_llm",
        lambda prompt, *, config, model, schema_name, schema, max_tokens=2048, extra_body=None: LLMResponse(
            f'{{"mapping":{{{extra_keys},"SPEAKER_00":""}}}}',
            elapsed_seconds=0.2,
            prompt_tokens=8,
            completion_tokens=4,
        ),
    )
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fake audio")
    out = tmp_path / "out"

    assert (
        main(
            [
                "transcribe",
                "--audio",
                str(audio),
                "--out",
                str(out),
                "--diarize",
                "--debug",
                "--context",
                "Host: Alice.",
            ]
        )
        == 0
    )

    transcript = (out / "transcript.md").read_text(encoding="utf-8")
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    speaker_map = json.loads((out / "_debug" / "speaker_map.json").read_text(encoding="utf-8"))
    progress = [
        json.loads(line) for line in (out / "_debug" / "progress.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert transcript == "# Transcript\n\n[00:00:00] SPEAKER_00: Hello\n"
    assert run["providers"]["speaker_map"]["status"] == "failed"
    assert run["providers"]["speaker_map"]["failure_kind"] == "invalid_output"
    assert speaker_map["status"] == "failed"
    assert speaker_map["failure_kind"] == "invalid_output"
    assert {issue["code"] for issue in speaker_map["issues"]} == {"EXTRA_KEY", "BAD_VALUE"}
    events = [(event["stage"], event["event"]) for event in progress]
    assert ("speaker_map", "failed") in events
    assert ("speaker_map", "fallback") in events
    assert (
        next(event for event in progress if event["stage"] == "speaker_map" and event["event"] == "fallback")[
            "failure_kind"
        ]
        == "invalid_output"
    )
    assert progress[-1]["stage"] == "run"
    assert progress[-1]["event"] == "done"
    stderr = capsys.readouterr().err
    assert "voxis speaker_map fallback" in stderr


def test_diarize_prefixes_turn_timestamps_by_default(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    install_fake_mlx(monkeypatch, words=("Hello", "world"), word_starts=(12.0, 34.0))
    install_fake_diarize(
        monkeypatch,
        [
            DiarizeSpan(start=12.0, end=12.3, speaker="SPEAKER_00"),
            DiarizeSpan(start=34.0, end=35.0, speaker="SPEAKER_01"),
        ],
    )
    monkeypatch.setenv("BAILIAN_API_KEY", "")
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fake audio")
    out = tmp_path / "out"

    assert main(["transcribe", "--audio", str(audio), "--out", str(out), "--diarize"]) == 0

    transcript = (out / "transcript.md").read_text(encoding="utf-8")
    assert "[00:00:12] SPEAKER_00: Hello" in transcript
    assert "[00:00:34] SPEAKER_01: world" in transcript


def test_diarize_debug_writes_empty_diarization_file(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    install_fake_mlx(monkeypatch)
    install_fake_diarize(monkeypatch, [])
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fake audio")
    out = tmp_path / "out"

    assert main(["transcribe", "--audio", str(audio), "--out", str(out), "--diarize", "--debug"]) == 0

    diarization = json.loads((out / "_debug" / "diarization.json").read_text(encoding="utf-8"))
    assert diarization["spans"] == []
    assert diarization["turns"] == []


def test_auto_language_routes_sortformer_policy_from_asr_metadata(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    install_fake_mlx(monkeypatch, language="Chinese", words=("我", "们"))
    install_fake_diarize(
        monkeypatch,
        [
            DiarizeSpan(start=0.0, end=0.3, speaker="SPEAKER_00"),
            DiarizeSpan(start=0.3, end=1.0, speaker="SPEAKER_01"),
        ],
    )
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fake audio")
    out = tmp_path / "out"

    assert main(["transcribe", "--audio", str(audio), "--out", str(out), "--diarize", "--debug"]) == 0

    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    diarization = json.loads((out / "_debug" / "diarization.json").read_text(encoding="utf-8"))
    assert run["transcript"]["language"] == "Chinese"
    assert run["providers"]["diarization"]["language_policy"] == "zh_boundary_repair_profile_v1"
    assert diarization["policy"]["language"] == "zh"
    assert diarization["policy"]["language_policy"] == "zh_boundary_repair_profile_v1"
    assert diarization["policy"]["parameters"]["boundary_guard_mode"] == "strict_overlap_margin"


def test_non_probe_unknown_sortformer_cluster_is_repaired(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    words = tuple(f"w{index}" for index in range(45))
    starts = tuple(index * 0.4 for index in range(len(words)))
    speaker_sequence = ["SPEAKER_00" if index % 2 == 0 else "SPEAKER_01" for index in range(len(words))]
    speaker_sequence[20] = "SPEAKER_00"
    speaker_sequence[21] = "SPEAKER_02"
    speaker_sequence[22] = "SPEAKER_00"
    spans = [
        DiarizeSpan(start=starts[index], end=starts[index] + 0.25, speaker=speaker_sequence[index])
        for index in range(len(words))
    ]
    install_fake_mlx(monkeypatch, words=words, word_starts=starts)
    install_fake_diarize(monkeypatch, spans)
    monkeypatch.setenv("BAILIAN_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    def fake_speaker_map_llm(
        prompt: str,
        *,
        config: Any,
        model: str,
        schema_name: str,
        schema: dict[str, object],
        max_tokens: int = 2048,
        extra_body: dict[str, object] | None = None,
    ) -> LLMResponse:
        mapping_schema = schema["properties"]["mapping"]  # type: ignore[index]
        assert mapping_schema["required"] == ["SPEAKER_00", "SPEAKER_01"]  # type: ignore[index]
        assert "SPEAKER_02" not in mapping_schema["properties"]  # type: ignore[index]
        return LLMResponse(
            '{"mapping":{"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}}',
            elapsed_seconds=0.2,
            prompt_tokens=8,
            completion_tokens=4,
        )

    monkeypatch.setattr("voxis.speaker_map.call_json_schema_llm", fake_speaker_map_llm)
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fake audio")
    out = tmp_path / "out"

    assert (
        main(
            [
                "transcribe",
                "--audio",
                str(audio),
                "--out",
                str(out),
                "--diarize",
                "--debug",
                "--context",
                "Host: Alice. Guest: Bob.",
            ]
        )
        == 0
    )

    transcript = (out / "transcript.md").read_text(encoding="utf-8")
    diarization = json.loads((out / "_debug" / "diarization.json").read_text(encoding="utf-8"))
    speaker_map = json.loads((out / "_debug" / "speaker_map.json").read_text(encoding="utf-8"))
    assert "SPEAKER_02" not in speaker_map["all_turn_speakers"]
    assert "SPEAKER_02" not in speaker_map["mapping"]
    assert diarization["policy"]["unknown_boundary_repair_count"] == 1
    assert "Unknown (SPEAKER_02)" not in transcript
    assert "SPEAKER_02" not in transcript


def test_polish_replaces_diarized_transcript_body(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    install_fake_mlx(monkeypatch)
    install_fake_diarize(monkeypatch, [DiarizeSpan(0.0, 1.0, "SPEAKER_00")])
    monkeypatch.setenv("BAILIAN_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(
        "voxis.speaker_map.call_json_schema_llm",
        lambda prompt, *, config, model, schema_name, schema, max_tokens=2048, extra_body=None: LLMResponse(
            '{"mapping":{"SPEAKER_00": "Alice"}}',
            elapsed_seconds=0.2,
            prompt_tokens=8,
            completion_tokens=4,
        ),
    )

    def fake_polish(turns, *, context, config):  # type: ignore[no-untyped-def]
        assert [turn.speaker for turn in turns] == ["Alice"]
        assert context == "Claude Code. Host: Alice"
        assert config.llm_api_key == "test-key"
        return PolishResult(
            text="[00:00:00.00 - 00:00:01.00] Alice: Hello, Claude Code.",
            prompt_template_id="timestamped_boundary_repair_polish_chunk_v2",
            prompt_tokens=10,
            completion_tokens=5,
            elapsed_seconds=1.2,
        )

    monkeypatch.setattr("voxis.run.polish_turns", fake_polish)
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fake audio")
    out = tmp_path / "out"

    assert (
        main(
            [
                "transcribe",
                "--audio",
                str(audio),
                "--out",
                str(out),
                "--diarize",
                "--polish",
                "--debug",
                "--context",
                "Claude Code. Host: Alice",
            ]
        )
        == 0
    )

    transcript = (out / "transcript.md").read_text(encoding="utf-8")
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    polish = json.loads((out / "_debug" / "polish.json").read_text(encoding="utf-8"))
    assert transcript == "# Transcript\n\n[00:00:00] Alice: Hello, Claude Code.\n"
    assert run["providers"]["speaker_map"]["status"] == "succeeded"
    assert run["providers"]["polish"]["status"] == "succeeded"
    assert run["providers"]["polish"]["prompt_tokens"] == 10
    assert run["providers"]["polish"]["max_workers"] == 20
    assert run["providers"]["polish"]["effective_workers"] == 1
    assert polish["status"] == "succeeded"
    assert polish["max_workers"] == 20
    assert polish["effective_workers"] == 1
    assert polish["text"] == "[00:00:00.00 - 00:00:01.00] Alice: Hello, Claude Code."


def test_polish_invalid_output_falls_back_to_unpolished_turns(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    install_fake_mlx(monkeypatch, word_starts=(12.0,))
    install_fake_diarize(monkeypatch, [DiarizeSpan(12.0, 13.0, "SPEAKER_00")])
    monkeypatch.setenv("BAILIAN_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(
        "voxis.speaker_map.call_json_schema_llm",
        lambda prompt, *, config, model, schema_name, schema, max_tokens=2048, extra_body=None: LLMResponse(
            '{"mapping":{"SPEAKER_00": null}}',
            elapsed_seconds=0.2,
            prompt_tokens=8,
            completion_tokens=4,
        ),
    )
    monkeypatch.setattr(
        "voxis.run.polish_turns",
        lambda turns, *, context, config: PolishResult(
            text="[00:00:12.00 - 00:00:13.00] SPEAKER_00: Hello",
            prompt_template_id="timestamped_boundary_repair_polish_chunk_v2",
            prompt_tokens=10,
            completion_tokens=5,
            elapsed_seconds=1.2,
            status="succeeded_with_fallback",
            issues=("error:MALFORMED_TURN_LINE",),
            chunk_count=1,
        ),
    )
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fake audio")
    out = tmp_path / "out"

    assert main(["transcribe", "--audio", str(audio), "--out", str(out), "--diarize", "--polish"]) == 0

    transcript = (out / "transcript.md").read_text(encoding="utf-8")
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert transcript == "# Transcript\n\n[00:00:12] SPEAKER_00: Hello\n"
    assert run["providers"]["polish"]["status"] == "succeeded_with_fallback"
    assert run["providers"]["polish"]["issue_count"] == 1
    assert run["providers"]["polish"]["chunk_count"] == 1
    assert run["providers"]["polish"]["max_workers"] == 20
    assert run["providers"]["polish"]["effective_workers"] == 1


def fake_diarization_result(spans: list[DiarizeSpan]) -> DiarizationResult:
    return DiarizationResult(
        backend="sortformer",
        model="FluidAudio Sortformer high-context",
        spans=spans,
        command=["fluidaudio", "sortformer"],
        runtime_seconds=0.1,
        runner={"returncode": 0},
        raw_payload={"segment_count": len(spans)},
    )


def install_fake_diarize(monkeypatch: MonkeyPatch, spans: list[DiarizeSpan]) -> None:
    monkeypatch.setattr(
        "voxis.run.diarize_audio",
        lambda audio: fake_diarization_result(spans),
    )


def install_fake_devices(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr("voxis.run.mlx_device_payload", lambda: {"backend": "mlx", "selected": "Device(gpu, 0)"})


def install_fake_mlx(
    monkeypatch: MonkeyPatch,
    *,
    fail_alignment: bool = False,
    aligner_returns_list: bool = False,
    language: str = "English",
    words: tuple[str, ...] = ("Hello",),
    word_starts: tuple[float, ...] | None = None,
    asr_outputs: list[list[dict[str, Any]]] | None = None,
    fail_asr_on_call: int | None = None,
    asr_generation_tokens: list[int] | None = None,
    split_chunk_seconds: float | None = None,
    split_plans: list[list[float]] | None = None,
) -> dict[str, Any]:
    asr._load_asr_model.cache_clear()
    asr._load_aligner_model.cache_clear()
    calls: dict[str, Any] = {}
    utils = cast(Any, types.ModuleType("mlx_audio.stt.utils"))

    def fake_load_audio(audio: str, sr: int) -> list[float]:
        calls.setdefault("load_audio", []).append({"audio": audio, "sr": sr})
        return [0.0] * sr

    default_asr_outputs = [[{"start": 0.0, "end": 0.5, "text": "Hello Claude Code", "language": language}]]
    asr_output_sets = asr_outputs or default_asr_outputs

    class FakeAsrOutput:
        def __init__(self, segments: list[dict[str, Any]], generation_tokens: int) -> None:
            self.segments = segments
            self.text = " ".join(str(segment["text"]) for segment in segments)
            self.generation_tokens = generation_tokens

    class FakeAsrModel:
        def generate(self, waveform: object, **kwargs: object) -> FakeAsrOutput:
            calls["asr"] = kwargs
            calls.setdefault("asr_calls", []).append(kwargs)
            calls.setdefault("asr_waveform_lengths", []).append(len(cast(Any, waveform)))
            if fail_asr_on_call == len(calls["asr_calls"]):
                raise RuntimeError("ASR provider retry failed")
            call_index = len(calls["asr_calls"]) - 1
            output_index = min(call_index, len(asr_output_sets) - 1)
            generation_tokens = asr_generation_tokens[call_index] if asr_generation_tokens else 1
            return FakeAsrOutput(asr_output_sets[output_index], generation_tokens)

    def fake_split_audio_into_chunks(wav: Any, *, sr: int, chunk_duration: float) -> list[tuple[Any, float]]:
        calls.setdefault("split_calls", []).append({"sr": sr, "chunk_duration": chunk_duration})
        if split_plans is not None:
            durations = split_plans[min(len(calls["split_calls"]) - 1, len(split_plans) - 1)]
            plan_chunks: list[tuple[Any, float]] = []
            start_sample = 0
            for seconds in durations:
                end_sample = start_sample + round(seconds * sr)
                plan_chunks.append((wav[start_sample:end_sample], start_sample / sr))
                start_sample = end_sample
            return plan_chunks
        step = len(wav) if split_chunk_seconds is None else round(split_chunk_seconds * sr)
        return [(wav[start : start + step], start / sr) for start in range(0, max(len(wav), 1), max(step, 1))]

    class FakeAlignItem:
        def __init__(self, text: str, start_time: float, end_time: float) -> None:
            self.text = text
            self.start_time = start_time
            self.end_time = end_time

    starts = word_starts or tuple(index * 0.4 for index in range(len(words)))

    class FakeAlignOutput:
        items = [FakeAlignItem(text, starts[index], starts[index] + 0.25) for index, text in enumerate(words)]

    class FakeAlignerModel:
        def generate(self, **kwargs: object) -> FakeAlignOutput | list[FakeAlignOutput]:
            calls["aligner"] = kwargs
            calls.setdefault("aligner_calls", []).append(kwargs)
            if fail_alignment:
                raise RuntimeError("aligner failed")
            if aligner_returns_list:
                return [FakeAlignOutput()]
            return FakeAlignOutput()

    def fake_load_model(model_id: str) -> FakeAsrModel | FakeAlignerModel:
        if "ForcedAligner" in model_id:
            return FakeAlignerModel()
        return FakeAsrModel()

    utils.load_audio = fake_load_audio
    utils.load_model = fake_load_model
    qwen3_asr = cast(Any, types.ModuleType("mlx_audio.stt.models.qwen3_asr.qwen3_asr"))
    qwen3_asr.split_audio_into_chunks = fake_split_audio_into_chunks
    monkeypatch.setitem(sys.modules, "mlx_audio", types.ModuleType("mlx_audio"))
    monkeypatch.setitem(sys.modules, "mlx_audio.stt", types.ModuleType("mlx_audio.stt"))
    monkeypatch.setitem(sys.modules, "mlx_audio.stt.utils", utils)
    for name in ("mlx_audio.stt.models", "mlx_audio.stt.models.qwen3_asr"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "mlx_audio.stt.models.qwen3_asr.qwen3_asr", qwen3_asr)
    return calls


def _fake_loaded_audio(tmp_path: Path, *, duration_seconds: float) -> Any:
    def load(path: Path, *, sample_rate_hz: int) -> LoadedAudio:
        del path
        return LoadedAudio(
            path=tmp_path / "sample.wav",
            sample_rate_hz=sample_rate_hz,
            waveform=np.zeros(round(duration_seconds * sample_rate_hz), dtype=np.float32),
            duration_s=duration_seconds,
        )

    return load
