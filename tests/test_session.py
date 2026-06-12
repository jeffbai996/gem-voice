"""Tests for session manager — the integrator of audio pipeline + Gemini Live."""
import asyncio
from unittest.mock import AsyncMock

import pytest

from gem_voice.session import Session, SessionAlreadyActiveError
from gem_voice.types import (
    Config,
    ModelConfig,
    Persona,
)


@pytest.fixture
def base_config():
    return Config(
        gemini_api_key="key",
        discord_owner_user_id="owner",
    )


@pytest.fixture
def persona():
    return Persona(name="P", system_prompt="be P")


@pytest.fixture
def model_config():
    return ModelConfig()


@pytest.fixture
def patched_gemini(monkeypatch):
    """Stub the Gemini session factory so unit tests don't need real I/O."""
    fake_gem = AsyncMock()
    monkeypatch.setattr("gem_voice.session._make_gemini_session", lambda api_key: fake_gem)
    return fake_gem


@pytest.mark.asyncio
async def test_start_then_status_active(base_config, persona, model_config, patched_gemini):
    s = Session(base_config)
    sess_id = await s.start(persona, model_config, owner_user_id="owner")
    assert sess_id.startswith("sess-")
    status = s.status()
    assert status.active_session == sess_id


@pytest.mark.asyncio
async def test_start_twice_raises(base_config, persona, model_config, patched_gemini):
    s = Session(base_config)
    await s.start(persona, model_config, owner_user_id="owner")
    with pytest.raises(SessionAlreadyActiveError):
        await s.start(persona, model_config, owner_user_id="owner")


@pytest.mark.asyncio
async def test_stop_when_active_returns_true(base_config, persona, model_config, patched_gemini):
    s = Session(base_config)
    await s.start(persona, model_config, owner_user_id="owner")
    was = await s.stop()
    assert was is True
    assert s.status().active_session is None


@pytest.mark.asyncio
async def test_stop_when_idle_returns_false(base_config):
    s = Session(base_config)
    was = await s.stop()
    assert was is False


@pytest.mark.asyncio
async def test_stop_then_start_clean(base_config, persona, model_config, patched_gemini):
    s = Session(base_config)
    sess1 = await s.start(persona, model_config, owner_user_id="owner")
    await s.stop()
    sess2 = await s.start(persona, model_config, owner_user_id="owner")
    assert sess1 != sess2


@pytest.mark.asyncio
async def test_push_opus_ignored_when_idle(base_config):
    """push_opus before start() is a no-op, not a crash."""
    s = Session(base_config)
    s.push_opus(b"\x00\x01\x02")
    # No exception, no side effects.
    assert s.status().active_session is None


@pytest.mark.asyncio
async def test_compose_persona_appends_voice_override(base_config, persona):
    """Voice calls suspend inherited text-channel silence etiquette —
    the very first live smoke test went mute because the parent's
    persona told the model to opt out of acknowledgments."""
    from gem_voice.session import _VOICE_OVERRIDE
    s = Session(base_config)
    composed = await s._compose_persona(persona)
    assert composed.system_prompt.startswith("be P")
    assert _VOICE_OVERRIDE in composed.system_prompt


@pytest.mark.asyncio
async def test_compose_persona_override_applies_without_memory_query(
        base_config):
    from gem_voice.session import _VOICE_OVERRIDE
    s = Session(base_config)
    p = Persona(name="P", system_prompt="be P", memory_query=None)
    composed = await s._compose_persona(p)
    assert _VOICE_OVERRIDE in composed.system_prompt
