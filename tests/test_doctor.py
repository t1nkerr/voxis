from __future__ import annotations

from io import StringIO

from pytest import MonkeyPatch

from voxis.config import VoxisConfig
from voxis.doctor import run_doctor


def test_doctor_base_is_ready_when_optional_features_are_missing(monkeypatch: MonkeyPatch) -> None:
    install_doctor_fakes(monkeypatch, executables={"ffmpeg": "/opt/homebrew/bin/ffmpeg"}, api_key=None)
    output = StringIO()

    exit_code = run_doctor(stream=output)

    assert exit_code == 0
    assert "[ok] ffmpeg:" in output.getvalue()
    assert "[optional] fluidaudio: not found on PATH" in output.getvalue()
    assert "[optional] LLM API key:" in output.getvalue()
    assert "Base transcription: ready" in output.getvalue()
    assert "Diarization: not ready" in output.getvalue()


def test_doctor_require_diarize_fails_without_fluidaudio(monkeypatch: MonkeyPatch) -> None:
    install_doctor_fakes(monkeypatch, executables={"ffmpeg": "/opt/homebrew/bin/ffmpeg"})
    output = StringIO()

    exit_code = run_doctor(require_diarize=True, stream=output)

    assert exit_code == 1
    assert "[missing] fluidaudio: not found on PATH" in output.getvalue()


def test_doctor_reports_full_readiness(monkeypatch: MonkeyPatch) -> None:
    install_doctor_fakes(
        monkeypatch,
        executables={
            "ffmpeg": "/opt/homebrew/bin/ffmpeg",
            "fluidaudio": "/Users/test/.local/bin/fluidaudio",
        },
        api_key="configured",
    )
    output = StringIO()

    exit_code = run_doctor(require_diarize=True, stream=output)

    assert exit_code == 0
    assert "Diarization: ready" in output.getvalue()
    assert "LLM features: ready" in output.getvalue()


def install_doctor_fakes(
    monkeypatch: MonkeyPatch,
    *,
    executables: dict[str, str],
    api_key: str | None = None,
) -> None:
    monkeypatch.setattr("voxis.doctor.platform.system", lambda: "Darwin")
    monkeypatch.setattr("voxis.doctor.platform.machine", lambda: "arm64")
    monkeypatch.setattr("voxis.doctor.platform.python_version", lambda: "3.12.12")
    monkeypatch.setattr("voxis.doctor.sys.version_info", (3, 12, 0))
    monkeypatch.setattr("voxis.doctor.shutil.which", lambda name: executables.get(name))
    monkeypatch.setattr(
        "voxis.doctor.load_config",
        lambda: VoxisConfig(
            llm_base_url="https://example.com/v1",
            llm_api_key=api_key,
            polish_model="model",
            speaker_map_model="model",
        ),
    )
