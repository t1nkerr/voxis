from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from dotenv import dotenv_values

CONFIG_PATH = Path.home() / ".config" / "voxis" / ".env"
DEFAULT_LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_POLISH_MODEL = "qwen3.8-flash"
DEFAULT_ASR_TERMS_MODEL = "qwen3.8-flash"
DEFAULT_SPEAKER_MAP_MODEL = "deepseek-flash"
DEFAULT_SPEAKER_MAP_BASE_URL = "https://api.deepseek.com"
DEFAULT_POLISH_MAX_WORKERS = 20


@dataclass(frozen=True)
class VoxisConfig:
    llm_base_url: str
    llm_api_key: str | None
    polish_model: str
    speaker_map_model: str
    asr_terms_model: str = DEFAULT_ASR_TERMS_MODEL
    polish_max_workers: int = DEFAULT_POLISH_MAX_WORKERS
    asr_terms: tuple[str, ...] = ()
    llm_disable_thinking: bool = True
    speaker_map_base_url: str | None = None
    speaker_map_api_key: str | None = None
    speaker_map_reasoning_effort: str | None = None


def _nonempty(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _parse_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def _parse_term_list(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(part.strip() for part in re.split(r"[,，]", value) if part.strip())


def load_config(
    *,
    env_file: Path = CONFIG_PATH,
    environ: Mapping[str, str] | None = None,
) -> VoxisConfig:
    env = dict(environ if environ is not None else os.environ)
    file_values: dict[str, str | None] = {}
    if env_file.exists():
        file_values = dict(dotenv_values(env_file))

    def get(name: str, default: str | None = None) -> str | None:
        if name in env:
            return _nonempty(env.get(name)) or default
        return _nonempty(file_values.get(name)) or default

    def get_alias(primary: str, legacy: str) -> str | None:
        for values in (env, file_values):
            if primary in values or legacy in values:
                return _nonempty(values.get(primary)) or _nonempty(values.get(legacy))
        return None

    llm_api_key = get_alias("VOXIS_LLM_API_KEY", "BAILIAN_API_KEY")
    llm_base_url = get("VOXIS_LLM_BASE_URL") or DEFAULT_LLM_BASE_URL
    disable_thinking_override = _parse_bool(get("VOXIS_LLM_DISABLE_THINKING"))

    return VoxisConfig(
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        polish_model=get("VOXIS_POLISH_MODEL") or DEFAULT_POLISH_MODEL,
        speaker_map_model=get("VOXIS_SPEAKER_MAP_MODEL") or DEFAULT_SPEAKER_MAP_MODEL,
        speaker_map_base_url=get("VOXIS_SPEAKER_MAP_BASE_URL") or DEFAULT_SPEAKER_MAP_BASE_URL,
        speaker_map_api_key=get_alias("VOXIS_SPEAKER_MAP_API_KEY", "DEEPSEEK_API_KEY"),
        speaker_map_reasoning_effort="low",
        asr_terms_model=get("VOXIS_ASR_TERMS_MODEL") or DEFAULT_ASR_TERMS_MODEL,
        polish_max_workers=DEFAULT_POLISH_MAX_WORKERS,
        asr_terms=_parse_term_list(get("VOXIS_ASR_TERMS")),
        llm_disable_thinking=(
            disable_thinking_override if disable_thinking_override is not None else "dashscope" in llm_base_url
        ),
    )
