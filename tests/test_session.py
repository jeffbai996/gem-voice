"""Tests for session manager — the integrator of voice + audio + Gemini."""
import asyncio
from unittest.mock import AsyncMock

import pytest

from gem_voice.session import Session, SessionAlreadyActiveError
from gem_voice.types import (
    Config,
    ModelConfig,
    Persona,
    VoiceCredentials,
)


@pytest.fixture
def base_config():
    return Config(
        gemini_api_key="key",
        discord_owner_user_id="owner",
    )


@pytest.fixture
def creds():
    return VoiceCredentials(
        guild_id="g", channel_id="c", user_id="12345",
        session_id="s", endpoint="e:443", token="t",
    )


@pytest.fixture
def persona():
    return Persona(name="P", system_prompt="be P")


@pytest.fixture
def model_config():
    return ModelConfig()


@pytest.fixture
def patched_factories(monkeypatch):
    """Stub voice + gemini factories so session tests don't need real I/O."""
    fake_vc = AsyncMock()
    fake_gem = AsyncMock()
    monkeypatch.setattr("gem_voice.session._make_voice_client", lambda: fake_vc)
    monkeypatch.setattr("gem_voice.session._make_gemini_session", lambda api_key: fake_gem)
    return fake_vc, fake_gem


@pytest.mark.asyncio
async def test_start_then_status_active(base_config, creds, persona, model_config, patched_factories):
    s = Session(base_config)
    sess_id = await s.start(creds, persona, model_config, owner_user_id="owner")
    assert sess_id.startswith("sess-")
    status = s.status()
    assert status.active_session == sess_id


@pytest.mark.asyncio
async def test_start_twice_raises(base_config, creds, persona, model_config, patched_factories):
    s = Session(base_config)
    await s.start(creds, persona, model_config, owner_user_id="owner")
    with pytest.raises(SessionAlreadyActiveError):
        await s.start(creds, persona, model_config, owner_user_id="owner")


@pytest.mark.asyncio
async def test_stop_when_active_returns_true(base_config, creds, persona, model_config, patched_factories):
    s = Session(base_config)
    await s.start(creds, persona, model_config, owner_user_id="owner")
    was = await s.stop()
    assert was is True
    assert s.status().active_session is None


@pytest.mark.asyncio
async def test_stop_when_idle_returns_false(base_config):
    s = Session(base_config)
    was = await s.stop()
    assert was is False


@pytest.mark.asyncio
async def test_stop_then_start_clean(base_config, creds, persona, model_config, patched_factories):
    s = Session(base_config)
    sess1 = await s.start(creds, persona, model_config, owner_user_id="owner")
    await s.stop()
    sess2 = await s.start(creds, persona, model_config, owner_user_id="owner")
    assert sess1 != sess2
