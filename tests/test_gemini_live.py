"""Tests for the Gemini Live wrapper. Uses a fake genai client to avoid real network."""
import asyncio
from unittest.mock import MagicMock

import pytest

from gem_voice.gemini_live import GeminiLiveSession
from gem_voice.types import ModelConfig, Persona, SessionEvent


class _FakeLiveSession:
    """Stand-in for genai.aio.live session. Records sent audio, replays canned events.

    The wrapper calls receive() in a `while True` loop (one iterator per turn),
    so the first call yields the canned replay, and every subsequent call
    waits forever — mimicking "session open, no more model turns yet". The
    recv task gets cancelled by the wrapper when the send-side hits its stop
    sentinel.
    """

    def __init__(self, replay=None):
        self.sent = []
        self._replay = replay or []
        self.closed = False
        self._turns_consumed = 0

    async def send_realtime_input(self, audio=None, media=None, **kwargs):
        # Accept either kwarg shape; the working example uses audio=.
        blob = media if media is not None else audio
        if blob is not None:
            self.sent.append(blob)

    async def receive(self):
        if self._turns_consumed == 0:
            self._turns_consumed += 1
            for item in self._replay:
                yield item
        else:
            # No more turns; block until cancelled.
            await asyncio.Event().wait()
            yield  # unreachable, satisfies async-generator shape

    async def close(self):
        self.closed = True


class _FakeContextManager:
    def __init__(self, sess):
        self.sess = sess

    async def __aenter__(self):
        return self.sess

    async def __aexit__(self, *args):
        return False


def _fake_client_factory(session_to_return):
    """Build a fake Client that returns the given session from aio.live.connect()."""
    class _FakeClient:
        class aio:
            class live:
                @staticmethod
                def connect(model, config):
                    _fake_client_factory._last = {"model": model, "config": config}
                    return _FakeContextManager(session_to_return)
    return _FakeClient()


@pytest.mark.asyncio
async def test_connect_passes_persona_and_model(monkeypatch):
    fake_session = _FakeLiveSession()
    monkeypatch.setattr(
        "gem_voice.gemini_live._make_client",
        lambda api_key: _fake_client_factory(fake_session),
    )

    sess = GeminiLiveSession(api_key="k")
    persona = Persona(name="X", system_prompt="You are X.")
    model_cfg = ModelConfig(model="model-y", voice="VoiceY", language="en-US")
    await sess.connect(persona, model_cfg)

    last = _fake_client_factory._last
    assert last["model"] == "model-y"
    serialized = str(last["config"])
    assert "You are X." in serialized
    assert "VoiceY" in serialized


@pytest.mark.asyncio
async def test_stream_forwards_pcm_in_and_collects_pcm_out(monkeypatch):
    # NB: bare MagicMocks poison the wrapper's feature-detection getattrs —
    # session_resumption_update/tool_call/go_away auto-mock as TRUTHY, the
    # sru branch swallows the response (`continue`) and pcm_out never fills
    # (the 2026-07-05 suite hang). Pin every probed attr explicitly.
    fake_session = _FakeLiveSession(replay=[
        MagicMock(
            session_resumption_update=None, tool_call=None,
            go_away=None, setup_complete=None,
            server_content=MagicMock(
                model_turn=MagicMock(parts=[
                    MagicMock(inline_data=MagicMock(data=b"\xaa" * 100, mime_type="audio/pcm"))
                ]),
                input_transcription=None,
                output_transcription=None,
                turn_complete=False,
            )),
        MagicMock(
            session_resumption_update=None, tool_call=None,
            go_away=None, setup_complete=None,
            server_content=MagicMock(
                turn_complete=True,
                model_turn=None,
                input_transcription=None,
                output_transcription=None,
            )),
    ])

    monkeypatch.setattr(
        "gem_voice.gemini_live._make_client",
        lambda api_key: _fake_client_factory(fake_session),
    )

    sess = GeminiLiveSession(api_key="k")
    await sess.connect(Persona(name="X", system_prompt="X"), ModelConfig())

    pcm_in: asyncio.Queue = asyncio.Queue()
    pcm_out: asyncio.Queue = asyncio.Queue()
    events: asyncio.Queue = asyncio.Queue()

    await pcm_in.put(b"\x01" * 640)
    await pcm_in.put(None)  # stop sentinel

    await sess.stream(pcm_in, pcm_out, events)

    assert len(fake_session.sent) == 1
    received = await pcm_out.get()
    assert received == b"\xaa" * 100


@pytest.mark.asyncio
async def test_close_idempotent(monkeypatch):
    fake_session = _FakeLiveSession()
    monkeypatch.setattr(
        "gem_voice.gemini_live._make_client",
        lambda api_key: _fake_client_factory(fake_session),
    )

    sess = GeminiLiveSession(api_key="k")
    await sess.connect(Persona(name="X", system_prompt="X"), ModelConfig())
    await sess.close()
    await sess.close()  # second close should not raise
    assert fake_session.closed is True
