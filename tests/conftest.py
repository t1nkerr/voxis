"""Keep unit tests isolated from credentials in the developer environment."""

from __future__ import annotations
import pytest
from pytest import MonkeyPatch


@pytest.fixture(autouse=True)
def no_live_provider_keys(monkeypatch: MonkeyPatch) -> None:
    for name in ("BAILIAN_API_KEY", "VOXIS_LLM_API_KEY", "DEEPSEEK_API_KEY", "VOXIS_SPEAKER_MAP_API_KEY"):
        monkeypatch.setenv(name, "")
