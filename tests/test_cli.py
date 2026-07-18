from __future__ import annotations

import json
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
        ("asr", "done"),
        ("alignment", "started"),
        ("alignment", "done"),
        ("outputs", "started"),
        ("outputs", "done"),
        ("run", "done"),
    ]
    assert progress[2]["duration_seconds"] == 1.0
    assert progress[4]["segment_count"] == 1
    assert progress[6]["word_count"] == 1
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

    with pytest.raises(SystemExit) as exc_info:
        main(["transcribe", "--audio", str(audio), "--out", str(tmp_path / "out"), "--polish"])

    assert exc_info.value.code == 2
    assert "--polish is allowed only with --diarize" in capsys.readouterr().err


def test_transcribe_passes_context_terms_and_language_to_asr(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    calls = install_fake_mlx(monkeypatch, aligner_returns_list=True)
    monkeypatch.setenv("BAILIAN_API_KEY", "test-key")
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
    assert calls["asr"]["chunk_duration"] == 60.0
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
    assert "max_tokens budget" in stderr
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
    assert debug["words"] == [{"text": "Hello", "start": 10.25, "end": 10.5}]


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
    assert run["providers"]["speaker_map"]["thinking_enabled"] is False
    assert speaker_map["mapping"] == {"SPEAKER_00": "Alice"}
    assert speaker_map["thinking_enabled"] is False


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


def test_unsupported_speaker_map_name_falls_back_to_unknown(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    install_fake_mlx(monkeypatch, words=("hello", "world"), word_starts=(0.0, 1.0))
    install_fake_diarize(
        monkeypatch,
        [
            DiarizeSpan(start=0.0, end=0.8, speaker="SPEAKER_00"),
            DiarizeSpan(start=1.0, end=1.8, speaker="SPEAKER_01"),
        ],
    )
    monkeypatch.setenv("BAILIAN_API_KEY", "test-key")
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
    assert speaker_map["mapping"] == {"SPEAKER_00": "Alice", "SPEAKER_01": "Unknown (SPEAKER_01)"}
    assert speaker_map["issues"] == [
        {
            "code": "UNSUPPORTED_NAME",
            "severity": "warning",
            "detail": "speaker map value must appear in context or transcript probe evidence; falling back to unknown",
            "speaker": "SPEAKER_01",
            "value": "Peter Yang",
        }
    ]
    assert "[00:00:01] Unknown (SPEAKER_01): world\n" in transcript
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
        def __init__(self, segments: list[dict[str, Any]]) -> None:
            self.segments = segments
            self.text = " ".join(str(segment["text"]) for segment in segments)

    class FakeAsrModel:
        def generate(self, waveform: object, **kwargs: object) -> FakeAsrOutput:
            calls["asr"] = kwargs
            calls.setdefault("asr_calls", []).append(kwargs)
            output_index = min(len(calls["asr_calls"]) - 1, len(asr_output_sets) - 1)
            return FakeAsrOutput(asr_output_sets[output_index])

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
    monkeypatch.setitem(sys.modules, "mlx_audio", types.ModuleType("mlx_audio"))
    monkeypatch.setitem(sys.modules, "mlx_audio.stt", types.ModuleType("mlx_audio.stt"))
    monkeypatch.setitem(sys.modules, "mlx_audio.stt.utils", utils)
    return calls
