"""End-to-end integration tests.

Runs the daemon's IpcServer + Session against a fake Gemini Live backend,
exercising the real unix socket and the real NDJSON IPC protocol.

v0.2 — gem-voice no longer talks to Discord. Parent streams Opus in via
the IPC audio_in action; gem-voice emits AUDIO_OUT events with Opus back.

Run: pytest tests/test_integration.py -v -m integration
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import uuid
from unittest.mock import MagicMock

import pytest

from gem_voice.ipc_server import IpcServer
from gem_voice.session import Session
from gem_voice.types import (
    Config,
    SessionEvent,
    SessionEventType,
)


pytestmark = pytest.mark.integration


@pytest.fixture
def short_sock_path():
    """AF_UNIX paths are capped at ~104 bytes on macOS."""
    p = f"/tmp/gv-int-{uuid.uuid4().hex[:8]}.sock"
    yield p
    try:
        os.unlink(p)
    except FileNotFoundError:
        pass


@pytest.fixture
def base_config():
    return Config(
        gemini_api_key="fake-key",
        discord_owner_user_id="42",
    )


class _FakeGeminiLive:
    """In-process stand-in for the Gemini Live session.

    stream() reads PCM from pcm_in until None, echoes a canned model
    audio chunk per non-None input frame, and emits MODEL_SPEECH_END
    on stop.
    """

    def __init__(self):
        self.connected = False
        self.closed = False
        self.received_pcm: list[bytes] = []
        self._persona = None
        self._model_config = None

    async def connect(self, persona, model_config):
        self.connected = True
        self._persona = persona
        self._model_config = model_config

    async def stream(self, pcm_in, pcm_out, events):
        while True:
            try:
                frame = await pcm_in.get()
            except asyncio.CancelledError:
                return
            if frame is None:
                await events.put(SessionEvent(
                    type=SessionEventType.MODEL_SPEECH_END,
                    data={"transcript": "[fake model done]"},
                ))
                return
            self.received_pcm.append(frame)
            # Emit one 20ms PCM chunk per input frame at 24kHz.
            # 24kHz mono int16 20ms = 480 samples * 2 bytes = 960 bytes.
            await pcm_out.put(b"\xbb" * 960)

    async def close(self):
        self.closed = True


async def _send_command(reader, writer, payload: dict) -> dict:
    writer.write((json.dumps(payload) + "\n").encode())
    await writer.drain()
    line = await reader.readline()
    return json.loads(line.decode())


def _join_payload() -> dict:
    return {
        "id": "join-1",
        "action": "join",
        "owner_user_id": "42",
        "persona": {
            "name": "TestPersona",
            "system_prompt": "You are TestPersona.",
        },
        "model_config": {
            "model": "fake-model",
            "voice": "FakeVoice",
            "language": "en-US",
        },
    }


# ---------- Tests ------------------------------------------------------

@pytest.mark.asyncio
async def test_full_join_audio_roundtrip(short_sock_path, base_config, monkeypatch):
    """End-to-end: IPC join → push Opus via audio_in → fake Gemini receives
    PCM → fake Gemini emits PCM → gem-voice encodes → AUDIO_OUT event lands
    on the wire as base64 opus."""
    fake_gemini = _FakeGeminiLive()
    monkeypatch.setattr("gem_voice.session._make_gemini_session", lambda api_key: fake_gemini)

    session_mgr = Session(base_config)
    server = IpcServer(socket_path=short_sock_path, session_manager=session_mgr)
    await server.start()

    async def _body():
        reader, writer = await asyncio.open_unix_connection(short_sock_path)

        # 1. Join
        resp = await _send_command(reader, writer, _join_payload())
        assert resp["ok"] is True
        assert fake_gemini.connected is True
        assert fake_gemini._persona.name == "TestPersona"

        await asyncio.sleep(0.05)

        # 2. Send Opus frames via audio_in. Use real Opus packets that decode
        # to speech-loud PCM so VAD passes them through.
        from pathlib import Path
        from gem_voice.audio import OpusEncoder, resample_pcm16
        sine_16k = (Path(__file__).parent / "fixtures" / "sine_440hz_16k_1s.pcm").read_bytes()
        sine_48k = resample_pcm16(sine_16k, src_rate=16000, dst_rate=48000)
        encoder = OpusEncoder()
        opus_packet = encoder.encode(sine_48k[:1920])

        for _ in range(10):
            b64 = base64.b64encode(opus_packet).decode("ascii")
            writer.write((json.dumps({"action": "audio_in", "b64": b64}) + "\n").encode())
        await writer.drain()

        # 3. Drain responses + events from the socket. Look for AUDIO_OUT.
        audio_out_received = False
        deadline = asyncio.get_event_loop().time() + 2.0
        while asyncio.get_event_loop().time() < deadline:
            try:
                line = await asyncio.wait_for(reader.readline(), timeout=0.2)
            except asyncio.TimeoutError:
                continue
            if not line:
                break
            msg = json.loads(line.decode())
            if msg.get("event") == "audio_out":
                audio_out_received = True
                # Decode and verify it's reasonable
                opus_back = base64.b64decode(msg["b64"])
                assert len(opus_back) > 0
                break

        assert audio_out_received, "expected AUDIO_OUT event with model opus"
        assert len(fake_gemini.received_pcm) > 0, "fake Gemini should have received PCM"

        writer.close()
        await writer.wait_closed()

    try:
        await asyncio.wait_for(_body(), timeout=10.0)
        for _ in range(20):
            await asyncio.sleep(0.1)
            if session_mgr.status().active_session is None:
                break
        assert session_mgr.status().active_session is None
        assert fake_gemini.closed is True
    finally:
        await session_mgr.stop()
        await server.stop()


@pytest.mark.asyncio
async def test_explicit_leave_returns_ack(short_sock_path, base_config, monkeypatch):
    fake_gemini = _FakeGeminiLive()
    monkeypatch.setattr("gem_voice.session._make_gemini_session", lambda api_key: fake_gemini)

    session_mgr = Session(base_config)
    server = IpcServer(socket_path=short_sock_path, session_manager=session_mgr)
    await server.start()

    async def _body():
        reader, writer = await asyncio.open_unix_connection(short_sock_path)
        resp = await _send_command(reader, writer, _join_payload())
        assert resp["ok"] is True

        resp = await _send_command(reader, writer, {"id": "leave-1", "action": "leave"})
        assert resp["ok"] is True
        assert resp["was_active"] is True

        writer.close()
        await writer.wait_closed()

    try:
        await asyncio.wait_for(_body(), timeout=5.0)
        assert fake_gemini.closed is True
    finally:
        await session_mgr.stop()
        await server.stop()


@pytest.mark.asyncio
async def test_join_twice_returns_error(short_sock_path, base_config, monkeypatch):
    fake_gemini = _FakeGeminiLive()
    monkeypatch.setattr("gem_voice.session._make_gemini_session", lambda api_key: fake_gemini)

    session_mgr = Session(base_config)
    server = IpcServer(socket_path=short_sock_path, session_manager=session_mgr)
    await server.start()

    try:
        reader, writer = await asyncio.open_unix_connection(short_sock_path)
        resp1 = await _send_command(reader, writer, _join_payload())
        assert resp1["ok"] is True

        payload2 = _join_payload()
        payload2["id"] = "join-2"
        resp2 = await _send_command(reader, writer, payload2)
        assert resp2["ok"] is False
        assert "already" in resp2["error"].lower()

        writer.close()
        await writer.wait_closed()
    finally:
        await session_mgr.stop()
        await server.stop()


@pytest.mark.asyncio
async def test_ipc_disconnect_tears_down_session(short_sock_path, base_config, monkeypatch):
    fake_gemini = _FakeGeminiLive()
    monkeypatch.setattr("gem_voice.session._make_gemini_session", lambda api_key: fake_gemini)

    session_mgr = Session(base_config)
    server = IpcServer(socket_path=short_sock_path, session_manager=session_mgr)
    await server.start()

    try:
        reader, writer = await asyncio.open_unix_connection(short_sock_path)
        resp = await _send_command(reader, writer, _join_payload())
        assert resp["ok"] is True
        assert session_mgr.status().active_session is not None

        writer.close()
        await writer.wait_closed()

        for _ in range(10):
            await asyncio.sleep(0.1)
            if session_mgr.status().active_session is None:
                break

        assert session_mgr.status().active_session is None
        assert fake_gemini.closed is True
    finally:
        await session_mgr.stop()
        await server.stop()


@pytest.mark.asyncio
async def test_bad_join_payload_rejected_cleanly(short_sock_path, base_config, monkeypatch):
    fake_gemini = _FakeGeminiLive()
    monkeypatch.setattr("gem_voice.session._make_gemini_session", lambda api_key: fake_gemini)

    session_mgr = Session(base_config)
    server = IpcServer(socket_path=short_sock_path, session_manager=session_mgr)
    await server.start()

    try:
        reader, writer = await asyncio.open_unix_connection(short_sock_path)

        # Missing required fields
        bad = {"id": "bad-1", "action": "join"}
        resp = await _send_command(reader, writer, bad)
        assert resp["ok"] is False
        assert resp["id"] == "bad-1"

        # Daemon should still be healthy
        resp = await _send_command(reader, writer, {"id": "stat-1", "action": "status"})
        assert resp["ok"] is True
        assert resp["active_session"] is None

        writer.close()
        await writer.wait_closed()
    finally:
        await session_mgr.stop()
        await server.stop()
