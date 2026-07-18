from __future__ import annotations

import platform
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from voxis.config import CONFIG_PATH, load_config


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    ready: bool
    detail: str
    required: bool


def run_doctor(*, require_diarize: bool = False, stream: TextIO = sys.stdout) -> int:
    config = load_config()
    checks = [
        DoctorCheck(
            name="platform",
            ready=platform.system() == "Darwin" and platform.machine() == "arm64",
            detail=f"{platform.system()} {platform.machine()}",
            required=True,
        ),
        DoctorCheck(
            name="python",
            ready=sys.version_info[:2] == (3, 12),
            detail=platform.python_version(),
            required=True,
        ),
        _executable_check("ffmpeg", required=True),
        _executable_check("fluidaudio", required=require_diarize),
        DoctorCheck(
            name="LLM API key",
            ready=config.llm_api_key is not None,
            detail="configured" if config.llm_api_key is not None else f"not configured in {CONFIG_PATH}",
            required=False,
        ),
    ]
    for check in checks:
        label = "ok" if check.ready else ("missing" if check.required else "optional")
        print(f"[{label}] {check.name}: {check.detail}", file=stream)

    base_ready = all(check.ready for check in checks if check.required and check.name != "fluidaudio")
    diarize_ready = next(check.ready for check in checks if check.name == "fluidaudio")
    llm_ready = next(check.ready for check in checks if check.name == "LLM API key")
    print(f"Base transcription: {'ready' if base_ready else 'not ready'}", file=stream)
    print(f"Diarization: {'ready' if diarize_ready else 'not ready'}", file=stream)
    print(f"LLM features: {'ready' if llm_ready else 'not ready'}", file=stream)
    return 0 if all(check.ready for check in checks if check.required) else 1


def _executable_check(name: str, *, required: bool) -> DoctorCheck:
    path = shutil.which(name)
    return DoctorCheck(
        name=name,
        ready=path is not None,
        detail=str(Path(path).resolve()) if path is not None else "not found on PATH",
        required=required,
    )
