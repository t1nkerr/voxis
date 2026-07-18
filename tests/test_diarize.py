from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from voxis.audio import LoadedAudio
from voxis.diarize import _parse_sortformer_spans, _sortformer_command, diarize_audio
from voxis.diarization_types import DiarizeSpan, Turn, format_timestamp, spans_payload, turns_payload


def test_format_timestamp_uses_hh_mm_ss() -> None:
    assert format_timestamp(12.9) == "00:00:12"
    assert format_timestamp(3671.2) == "01:01:11"


def test_diarization_payload_helpers_preserve_debug_json_contract() -> None:
    assert spans_payload([DiarizeSpan(start=1.2345, end=2.3456, speaker="SPEAKER_00")]) == [
        {"speaker": "SPEAKER_00", "start": 1.234, "end": 2.346}
    ]
    assert turns_payload([Turn(speaker="Alice", start=1.2345, end=2.3456, text="hello")]) == [
        {"speaker": "Alice", "start": 1.234, "end": 2.346, "text": "hello"}
    ]


def test_parse_sortformer_spans_normalizes_speaker_indices() -> None:
    spans = _parse_sortformer_spans(
        {
            "segments": [
                {
                    "startTimeSeconds": 1.0,
                    "endTimeSeconds": 2.0,
                    "speaker": "Speaker 3",
                    "speakerIndex": 3,
                },
                {
                    "startTimeSeconds": 0.0,
                    "endTimeSeconds": 0.5,
                    "speaker": "Speaker 1",
                    "speakerIndex": 1,
                },
            ]
        }
    )

    assert [(span.start, span.end, span.speaker) for span in spans] == [
        (0.0, 0.5, "SPEAKER_01"),
        (1.0, 2.0, "SPEAKER_03"),
    ]


def test_sortformer_command_uses_hardcoded_production_config() -> None:
    command = _sortformer_command(Path("audio.mp3"), Path("out.json"))

    assert command == [
        "fluidaudio",
        "sortformer",
        "audio.mp3",
        "--output",
        "out.json",
        "--config",
        "high",
    ]


def test_diarize_audio_runs_sortformer_executable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    executable = tmp_path / "fluidaudio"
    executable.write_text(
        """#!/bin/sh
printf '%s\n' '{"segments":[{"startTimeSeconds":0.0,"endTimeSeconds":1.25,"speakerIndex":1}]}' > "$4"
echo runner-ok
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    audio_path = tmp_path / "audio.mp3"
    audio_path.write_bytes(b"fake audio")
    audio = LoadedAudio(
        path=audio_path,
        sample_rate_hz=16_000,
        waveform=np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32),
        duration_s=1.25,
    )

    result = diarize_audio(audio)

    assert [(span.start, span.end, span.speaker) for span in result.spans] == [(0.0, 1.25, "SPEAKER_01")]
    assert result.command[:2] == ["fluidaudio", "sortformer"]
    assert result.command[2].endswith("/sortformer_input.wav")
    assert result.command[2] != str(audio_path)
    assert result.runner["returncode"] == 0
    assert result.runner["stdout"] == "runner-ok"
    assert result.raw_payload["segment_count"] == 1


def test_diarize_audio_writes_mono_16khz_pcm_for_sortformer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    capture = tmp_path / "captured.wav"
    executable = tmp_path / "fluidaudio"
    executable.write_text(
        f"""#!/bin/sh
cp "$2" "{capture}"
printf '%s\n' '{{"segments":[]}}' > "$4"
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:/bin")
    source = tmp_path / "meeting.m4a"
    source.write_bytes(b"compressed input")
    audio = LoadedAudio(
        path=source,
        sample_rate_hz=16_000,
        waveform=np.array([-1.0, 0.0, 1.0], dtype=np.float32),
        duration_s=3 / 16_000,
    )

    diarize_audio(audio)

    with wave.open(str(capture), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getframerate() == 16_000
        assert wav.getsampwidth() == 2
        assert wav.getnframes() == 3


def test_diarize_audio_reports_nonzero_sortformer_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    executable = tmp_path / "fluidaudio"
    executable.write_text(
        """#!/bin/sh
echo sortformer-boom >&2
exit 7
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    audio = LoadedAudio(
        path=tmp_path / "audio.wav",
        sample_rate_hz=16_000,
        waveform=np.zeros(0, dtype=np.float32),
        duration_s=0.0,
    )

    with pytest.raises(RuntimeError) as exc_info:
        diarize_audio(audio)

    assert "Sortformer failed with exit code 7: sortformer-boom" in str(exc_info.value)


def test_diarize_audio_requires_sortformer_output_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    executable = tmp_path / "fluidaudio"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    audio = LoadedAudio(
        path=tmp_path / "audio.wav",
        sample_rate_hz=16_000,
        waveform=np.zeros(0, dtype=np.float32),
        duration_s=0.0,
    )

    with pytest.raises(RuntimeError) as exc_info:
        diarize_audio(audio)

    assert "Sortformer did not write expected output" in str(exc_info.value)


def test_diarize_audio_missing_cli_error_points_to_readme(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    audio = LoadedAudio(
        path=tmp_path / "audio.wav",
        sample_rate_hz=16_000,
        waveform=np.zeros(0, dtype=np.float32),
        duration_s=0.0,
    )

    with pytest.raises(RuntimeError) as exc_info:
        diarize_audio(audio)

    message = str(exc_info.value)
    assert "https://github.com/t1nkerr/voxis#diarization-setup" in message
    assert "`which fluidaudio`" in message
    assert "scripts/install_fluidaudio_cli.sh" not in message
    assert "uv run" not in message


def test_diarize_audio_times_out_hung_sortformer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    executable = tmp_path / "fluidaudio"
    executable.write_text("#!/bin/sh\n/bin/sleep 30\n", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setattr("voxis.diarize.SORTFORMER_TIMEOUT_SECONDS", 0.2)
    audio = LoadedAudio(
        path=tmp_path / "audio.wav",
        sample_rate_hz=16_000,
        waveform=np.zeros(0, dtype=np.float32),
        duration_s=0.0,
    )

    with pytest.raises(RuntimeError) as exc_info:
        diarize_audio(audio)

    assert "Sortformer timed out after 0.2 seconds" in str(exc_info.value)
