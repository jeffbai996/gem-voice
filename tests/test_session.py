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
async def test_push_video_frame_ignored_when_idle(base_config):
    """push_video_frame before start() is a no-op (no active session), not a crash."""
    s = Session(base_config)
    s.push_video_frame(b"\xff\xd8\xff")
    assert s._video_in.empty()
    assert s.status().active_session is None


@pytest.mark.asyncio
async def test_push_video_frame_drops_oldest_on_overflow(base_config, monkeypatch):
    """A full video queue drops the OLDEST frame for the newest (freshest-wins,
    since video is 1fps + lossy-tolerant) and bumps the drop counter."""
    monkeypatch.setenv("GEM_VOICE_VIDEO_QUEUE", "3")
    s = Session(base_config)
    s._active_session_id = "sess-test"   # bypass start() to drive push_video_frame
    for i in range(5):
        s.push_video_frame(bytes([i]))
    assert s._video_in.qsize() == 3
    assert s._video_dropped == 2
    # the three retained are the freshest frames (2, 3, 4), oldest two dropped
    retained = [s._video_in.get_nowait() for _ in range(3)]
    assert retained == [bytes([2]), bytes([3]), bytes([4])]


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


# --------------------------------------------------------------------------
# Barge-in: a new /voice speak utterance CANCELS the in-flight one instead of
# queueing behind it. Each test stubs _do_say with a controllable coroutine so
# we exercise the cancel/flush wiring without real TTS I/O.
# --------------------------------------------------------------------------


def _instrumented_do_say(events: list[str]):
    """Build a fake _do_say that records start/cancel and blocks until cancelled,
    so a second say() must interrupt it. `events` accumulates a trace."""
    async def fake(self, text, voice=None):  # noqa: ANN001 — matches _do_say sig
        events.append(f"start:{text}")
        try:
            await asyncio.sleep(3600)  # "playing" — only ends via cancel
        except asyncio.CancelledError:
            events.append(f"cancel:{text}")
            raise
        finally:
            events.append(f"done:{text}")
    return fake


async def _drain_flush_events(s) -> int:
    """Count AUDIO_FLUSH events currently queued without blocking."""
    from gem_voice.types import SessionEventType
    n = 0
    while not s._events.empty():
        ev = s._events.get_nowait()
        if ev.type == SessionEventType.AUDIO_FLUSH:
            n += 1
    return n


@pytest.mark.asyncio
async def test_second_say_cancels_the_first(base_config, monkeypatch):
    """A say() arriving while one is in flight cancels the first and starts the
    second (barge-in), instead of serializing behind it."""
    trace: list[str] = []
    monkeypatch.setattr(Session, "_do_say", _instrumented_do_say(trace))

    s = Session(base_config)
    await s.say("first")
    await asyncio.sleep(0.02)  # let the first say task start
    assert "start:first" in trace

    await s.say("second")
    await asyncio.sleep(0.02)  # let the cancel land + second start
    assert "cancel:first" in trace
    assert "start:second" in trace

    # Clean up the still-running second say.
    await s.cancel_say()


@pytest.mark.asyncio
async def test_barge_in_emits_audio_flush(base_config, monkeypatch):
    """When a new say preempts a playing one, an AUDIO_FLUSH event is emitted so
    the parent drops its banked/playing frames (otherwise the old audio keeps
    playing on the wire for ~a second)."""
    trace: list[str] = []
    monkeypatch.setattr(Session, "_do_say", _instrumented_do_say(trace))

    s = Session(base_config)
    await s.say("first")
    await asyncio.sleep(0.02)
    # First say should not itself have flushed.
    assert await _drain_flush_events(s) == 0

    await s.say("second")
    await asyncio.sleep(0.02)
    assert await _drain_flush_events(s) == 1

    await s.cancel_say()


@pytest.mark.asyncio
async def test_cancel_say_stops_inflight_and_flushes(base_config, monkeypatch):
    """cancel_say() with a say in flight cancels it and emits a flush."""
    trace: list[str] = []
    monkeypatch.setattr(Session, "_do_say", _instrumented_do_say(trace))

    s = Session(base_config)
    await s.say("hello")
    await asyncio.sleep(0.02)
    assert "start:hello" in trace

    cancelled = await s.cancel_say()
    await asyncio.sleep(0.02)
    assert cancelled is True
    assert "cancel:hello" in trace
    assert await _drain_flush_events(s) == 1


@pytest.mark.asyncio
async def test_cancel_say_noop_when_nothing_playing(base_config, monkeypatch):
    """cancel_say() with no active say returns False and emits no flush."""
    trace: list[str] = []
    monkeypatch.setattr(Session, "_do_say", _instrumented_do_say(trace))

    s = Session(base_config)
    cancelled = await s.cancel_say()
    assert cancelled is False
    assert await _drain_flush_events(s) == 0

