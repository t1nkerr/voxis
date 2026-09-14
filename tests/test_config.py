from __future__ import annotations

from pathlib import Path

from voxis.config import load_config


def test_load_config_uses_bailian_defaults_without_file() -> None:
    config = load_config(environ={}, env_file=Path("/missing/.env"))

    assert config.llm_api_key is None
    assert config.llm_base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert config.polish_model == "qwen3.8-flash"
    assert config.speaker_map_model == "deepseek-flash"
    assert config.polish_max_workers == 20


def test_load_config_reads_only_secret_from_environment_over_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("BAILIAN_API_KEY=file-key\n", encoding="utf-8")

    config = load_config(
        env_file=env_file,
        environ={"BAILIAN_API_KEY": "env-key"},
    )

    assert config.llm_api_key == "env-key"
    assert config.llm_base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert config.polish_model == "qwen3.8-flash"
    assert config.speaker_map_model == "deepseek-flash"
    assert config.polish_max_workers == 20


def test_load_config_uses_file_secret_when_environment_is_empty(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("BAILIAN_API_KEY=file-key\n", encoding="utf-8")

    config = load_config(env_file=env_file, environ={})

    assert config.llm_api_key == "file-key"


def test_load_config_voxis_key_takes_precedence_over_bailian(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("BAILIAN_API_KEY=file-bailian\n", encoding="utf-8")

    config = load_config(
        env_file=env_file,
        environ={"VOXIS_LLM_API_KEY": "env-voxis"},
    )

    assert config.llm_api_key == "env-voxis"


def test_load_config_environment_legacy_key_precedes_file_canonical_key(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("VOXIS_LLM_API_KEY=file-voxis\n", encoding="utf-8")

    config = load_config(env_file=env_file, environ={"BAILIAN_API_KEY": "env-bailian"})

    assert config.llm_api_key == "env-bailian"


def test_load_config_reads_provider_overrides_from_environment() -> None:
    config = load_config(
        environ={
            "VOXIS_LLM_BASE_URL": "https://api.openai.com/v1",
            "VOXIS_POLISH_MODEL": "gpt-5",
            "VOXIS_SPEAKER_MAP_MODEL": "gpt-5-mini",
            "VOXIS_ASR_TERMS_MODEL": "gpt-5-nano",
        },
        env_file=Path("/missing/.env"),
    )

    assert config.llm_base_url == "https://api.openai.com/v1"
    assert config.polish_model == "gpt-5"
    assert config.speaker_map_model == "gpt-5-mini"
    assert config.asr_terms_model == "gpt-5-nano"


def test_load_config_reads_provider_overrides_from_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("VOXIS_LLM_BASE_URL=https://example.com/v1\n", encoding="utf-8")

    config = load_config(env_file=env_file, environ={})

    assert config.llm_base_url == "https://example.com/v1"


def test_load_config_defaults_asr_terms_to_empty() -> None:
    config = load_config(environ={}, env_file=Path("/missing/.env"))

    assert config.asr_terms == ()


def test_load_config_parses_asr_terms_list() -> None:
    config = load_config(
        environ={"VOXIS_ASR_TERMS": "Claude Code, CAR-T，泓君,,"},
        env_file=Path("/missing/.env"),
    )

    assert config.asr_terms == ("Claude Code", "CAR-T", "泓君")


def test_load_config_disables_thinking_by_default_for_dashscope() -> None:
    config = load_config(environ={}, env_file=Path("/missing/.env"))

    assert config.llm_disable_thinking is True


def test_load_config_enables_thinking_by_default_for_other_providers() -> None:
    config = load_config(
        environ={"VOXIS_LLM_BASE_URL": "https://api.openai.com/v1"},
        env_file=Path("/missing/.env"),
    )

    assert config.llm_disable_thinking is False


def test_load_config_disable_thinking_explicit_override() -> None:
    enabled = load_config(
        environ={"VOXIS_LLM_DISABLE_THINKING": "true"},
        env_file=Path("/missing/.env"),
    )
    disabled = load_config(
        environ={"VOXIS_LLM_DISABLE_THINKING": "0"},
        env_file=Path("/missing/.env"),
    )

    assert enabled.llm_disable_thinking is True
    assert disabled.llm_disable_thinking is False


def test_speaker_provider_is_separate_from_polish(tmp_path: Path) -> None:
    config = load_config(
        env_file=tmp_path / "missing", environ={"BAILIAN_API_KEY": "shared", "DEEPSEEK_API_KEY": "speaker"}
    )
    assert config.llm_api_key == "shared"
    assert "dashscope" in config.llm_base_url
    assert config.polish_model == "qwen3.8-flash"
    assert config.speaker_map_base_url == "https://api.deepseek.com"
    assert config.speaker_map_api_key == "speaker"
    assert config.speaker_map_reasoning_effort == "low"


def test_missing_speaker_key_never_reuses_shared_key(tmp_path: Path) -> None:
    config = load_config(env_file=tmp_path / "missing", environ={"BAILIAN_API_KEY": "shared"})
    assert config.speaker_map_api_key is None
