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


# --- regression: a malformed opus/PCM frame must not kill the loop -------------
@pytest.mark.asyncio
async def test_decode_loop_survives_bad_packet(base_config, monkeypatch):
    """A frame that makes the decoder raise is dropped; the next good frame still flows."""
    class FlakyDecoder:
        def decode(self, opus):
            if opus == b"BAD":
                raise RuntimeError("simulated OpusError: corrupt packet")
            return b"\x00\x00" * 480

    monkeypatch.setattr("gem_voice.session.OpusDecoder", FlakyDecoder)
    monkeypatch.setattr("gem_voice.session.resample_pcm16",
                        lambda pcm, src_rate, dst_rate: pcm)

    s = Session(base_config)
    opus_in: asyncio.Queue = asyncio.Queue()
    pcm_in: asyncio.Queue = asyncio.Queue()
    await opus_in.put(b"BAD")
    await opus_in.put(b"GOOD")

    task = asyncio.create_task(s._decode_loop(opus_in, pcm_in))
    out = await asyncio.wait_for(pcm_in.get(), timeout=2.0)
    task.cancel()

    assert out == b"\x00\x00" * 480
    assert s._opus_decode_dropped == 1


@pytest.mark.asyncio
async def test_encode_loop_survives_bad_frame(base_config, monkeypatch):
    """A frame that makes the encoder raise is dropped; the loop keeps running."""
    class FlakyEncoder:
        def __init__(self):
            self.calls = 0
        def encode(self, frame):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("simulated encode failure")
            return b"opusbytes"

    monkeypatch.setattr("gem_voice.session.OpusEncoder", FlakyEncoder)
    monkeypatch.setattr("gem_voice.session.resample_pcm16",
                        lambda pcm, src_rate, dst_rate: b"\x00" * 3840)

    s = Session(base_config)
    pcm_out: asyncio.Queue = asyncio.Queue()
    await pcm_out.put(b"\x00" * 3840)

    task = asyncio.create_task(s._encode_loop(pcm_out))
    ev = await asyncio.wait_for(s._events.get(), timeout=2.0)
    task.cancel()

    assert ev.data["b64"]
    assert s._opus_encode_dropped == 1

