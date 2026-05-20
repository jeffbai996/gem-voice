"""End-to-end integration tests.

Runs the daemon's IpcServer + Session against fake voice and fake Gemini Live
backends, exercising the real unix socket and the real NDJSON IPC protocol.
This catches bugs in module wiring that the per-module unit tests don't.

Not a subprocess test — the daemon runs in-process so we can monkeypatch the
factories. The signal/SIGTERM path is covered by the manual smoke test in
the README.

Run: pytest tests/test_integration.py -v -m integration
Skip in default sweep: pytest -m "not integration"
"""
from __future__ import annotations

import asyncio
import json
import os
import uuid
from unittest.mock import AsyncMock, MagicMock

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


# ---------- Fake collaborators -----------------------------------------

class _FakeVoiceClient:
    """In-process stand-in for the Discord voice WS client.

    Records frames passed to send_loop, lets tests push frames into
    recv_loop via the registered sink.
    """

    def __init__(self):
        self.sent_frames: list[bytes] = []
        self.connected = False
        self.disconnected = False
        self._sink = None
        self._summoner_ssrc: int | None = None

    async def connect(self, creds):
        self.connected = True

    async def recv_loop(self, opus_out: asyncio.Queue, summoner_ssrc: int):
        self._summoner_ssrc = summoner_ssrc
        # Sink pattern matches the real voice client — recv loop holds
        # until cancelled; frames arrive via push_inbound_frame().
        loop = asyncio.get_running_loop()

        def push(ssrc: int, data: bytes):
            if ssrc != self._summoner_ssrc:
                return
            loop.call_soon_threadsafe(opus_out.put_nowait, data)

        self._sink = push
        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            return

    async def send_loop(self, opus_in: asyncio.Queue):
        while True:
            frame = await opus_in.get()
            if frame is None:
                return
            self.sent_frames.append(frame)

    async def disconnect(self):
        self.disconnected = True

    # --- Test-side helpers (not on real client) ---
    def push_inbound_frame(self, ssrc: int, opus_data: bytes):
        """Test calls this to simulate Discord delivering a frame."""
        if self._sink is None:
            raise RuntimeError("recv_loop not started yet")
        self._sink(ssrc, opus_data)


class _FakeGeminiLive:
    """In-process stand-in for the Gemini Live session.

    `stream()` reads PCM from pcm_in until None, echoes a canned model
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

    async def stream(self, pcm_in: asyncio.Queue, pcm_out: asyncio.Queue, events: asyncio.Queue):
        while True:
            try:
                frame = await pcm_in.get()
            except asyncio.CancelledError:
                return
            if frame is None:
                # Stream end — emit a final speech-end and exit
                await events.put(SessionEvent(
                    type=SessionEventType.MODEL_SPEECH_END,
                    data={"transcript": "[fake model done]"},
                ))
                return
            self.received_pcm.append(frame)
            # Echo: emit one 20ms PCM chunk per input frame at 24kHz.
            # 24kHz mono int16 20ms = 480 samples * 2 bytes = 960 bytes.
            # After 24k→48k upsample in _encode_loop, that becomes a 1920-byte
            # frame, which is exactly one Opus encode unit.
            await pcm_out.put(b"\xbb" * 960)

    async def close(self):
        self.closed = True


# ---------- IPC helpers ------------------------------------------------

async def _send_command(reader, writer, payload: dict) -> dict:
    writer.write((json.dumps(payload) + "\n").encode())
    await writer.drain()
    line = await reader.readline()
    return json.loads(line.decode())


async def _read_event(reader, timeout: float = 2.0) -> dict:
    line = await asyncio.wait_for(reader.readline(), timeout=timeout)
    return json.loads(line.decode())


def _join_payload(owner_id: str = "42", user_ssrc: str = "42") -> dict:
    return {
        "id": "join-1",
        "action": "join",
        "vc_credentials": {
            "guild_id": "100",
            "channel_id": "200",
            "user_id": user_ssrc,  # session.py reads .user_id as the summoner SSRC
            "session_id": "sess-handshake",
            "endpoint": "fake.discord.media:443",
            "token": "fake-voice-token",
        },
        "owner_user_id": owner_id,
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
async def test_full_join_audio_roundtrip_leave(
    short_sock_path,
    base_config,
    monkeypatch,
):
    """The big one: spin up the whole daemon-in-process stack and verify
    audio flows IPC → voice in → audio decode → gemini → audio encode → voice out.
    """
    fake_voice = _FakeVoiceClient()
    fake_gemini = _FakeGeminiLive()

    monkeypatch.setattr("gem_voice.session._make_voice_client", lambda: fake_voice)
    monkeypatch.setattr("gem_voice.session._make_gemini_session", lambda api_key: fake_gemini)

    session_mgr = Session(base_config)
    server = IpcServer(socket_path=short_sock_path, session_manager=session_mgr)
    await server.start()

    async def _run_body():
        reader, writer = await asyncio.open_unix_connection(short_sock_path)

        # 1. Join
        resp = await _send_command(reader, writer, _join_payload())
        assert resp["ok"] is True
        assert resp["session_id"].startswith("sess-")
        assert fake_voice.connected is True
        assert fake_gemini.connected is True
        assert fake_gemini._persona.name == "TestPersona"
        assert fake_gemini._model_config.voice == "FakeVoice"

        # 2. Give the recv_loop and send_loop a moment to register
        await asyncio.sleep(0.05)

        # 3. Simulate Discord delivering audio frames to gem-voice.
        from pathlib import Path
        from gem_voice.audio import OpusEncoder, resample_pcm16
        sine_16k = (Path(__file__).parent / "fixtures" / "sine_440hz_16k_1s.pcm").read_bytes()
        sine_48k = resample_pcm16(sine_16k, src_rate=16000, dst_rate=48000)
        encoder = OpusEncoder()
        opus_packet = encoder.encode(sine_48k[:1920])  # 20ms at 48kHz mono int16

        # SSRC must match session.py's int(vc_credentials.user_id)
        for _ in range(10):
            fake_voice.push_inbound_frame(ssrc=42, opus_data=opus_packet)

        # 4. Wait for audio to round-trip through the whole pipeline.
        for _ in range(20):
            await asyncio.sleep(0.05)
            if fake_gemini.received_pcm and fake_voice.sent_frames:
                break
        assert len(fake_gemini.received_pcm) > 0, (
            f"PCM never reached gemini; pipeline upstream is broken."
        )
        assert len(fake_voice.sent_frames) > 0, (
            f"Opus never reached voice send_loop; pipeline downstream is broken."
        )

        # 5. Status while active
        resp = await _send_command(reader, writer, {"id": "stat-1", "action": "status"})
        assert resp["ok"] is True
        assert resp["active_session"] is not None

        # 6. Disconnect — IPC server's EOF handler will tear down session
        writer.close()
        await writer.wait_closed()

    try:
        await asyncio.wait_for(_run_body(), timeout=10.0)
        # Give the EOF handler a moment to run teardown
        for _ in range(20):
            await asyncio.sleep(0.1)
            if session_mgr.status().active_session is None:
                break
        assert session_mgr.status().active_session is None, "session should be torn down after disconnect"
        assert fake_voice.disconnected is True
        assert fake_gemini.closed is True
    finally:
        await session_mgr.stop()
        await server.stop()


@pytest.mark.asyncio
async def test_explicit_leave_returns_ack(short_sock_path, base_config, monkeypatch):
    """Sending 'leave' over IPC returns ack and tears down session.

    Separate from the audio roundtrip test — that one uses disconnect-driven
    teardown (writer.close()), which is the architecturally primary path.
    This test verifies the explicit leave command works without exercising
    audio pipeline cleanup ordering edge cases.
    """
    fake_voice = _FakeVoiceClient()
    fake_gemini = _FakeGeminiLive()
    monkeypatch.setattr("gem_voice.session._make_voice_client", lambda: fake_voice)
    monkeypatch.setattr("gem_voice.session._make_gemini_session", lambda api_key: fake_gemini)

    session_mgr = Session(base_config)
    server = IpcServer(socket_path=short_sock_path, session_manager=session_mgr)
    await server.start()

    async def _body():
        reader, writer = await asyncio.open_unix_connection(short_sock_path)
        resp = await _send_command(reader, writer, _join_payload())
        assert resp["ok"] is True

        # Send leave — no audio pushed first, so teardown has no stuck queues
        resp = await _send_command(reader, writer, {"id": "leave-1", "action": "leave"})
        assert resp["ok"] is True
        assert resp["was_active"] is True

        writer.close()
        await writer.wait_closed()

    try:
        await asyncio.wait_for(_body(), timeout=5.0)
        assert fake_voice.disconnected is True
        assert fake_gemini.closed is True
    finally:
        await session_mgr.stop()
        await server.stop()


@pytest.mark.asyncio
async def test_join_twice_returns_error(short_sock_path, base_config, monkeypatch):
    """Second join while session active should be rejected."""
    fake_voice = _FakeVoiceClient()
    fake_gemini = _FakeGeminiLive()
    monkeypatch.setattr("gem_voice.session._make_voice_client", lambda: fake_voice)
    monkeypatch.setattr("gem_voice.session._make_gemini_session", lambda api_key: fake_gemini)

    session_mgr = Session(base_config)
    server = IpcServer(socket_path=short_sock_path, session_manager=session_mgr)
    await server.start()

    try:
        reader, writer = await asyncio.open_unix_connection(short_sock_path)
        resp1 = await _send_command(reader, writer, _join_payload())
        assert resp1["ok"] is True

        # Second join over the same connection
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
    """Parent dropping the IPC connection should trigger session teardown."""
    fake_voice = _FakeVoiceClient()
    fake_gemini = _FakeGeminiLive()
    monkeypatch.setattr("gem_voice.session._make_voice_client", lambda: fake_voice)
    monkeypatch.setattr("gem_voice.session._make_gemini_session", lambda api_key: fake_gemini)

    session_mgr = Session(base_config)
    server = IpcServer(socket_path=short_sock_path, session_manager=session_mgr)
    await server.start()

    try:
        reader, writer = await asyncio.open_unix_connection(short_sock_path)
        resp = await _send_command(reader, writer, _join_payload())
        assert resp["ok"] is True
        assert session_mgr.status().active_session is not None

        # Drop the connection
        writer.close()
        await writer.wait_closed()

        # Give the server's disconnect-cleanup logic time to fire
        for _ in range(10):
            await asyncio.sleep(0.1)
            if session_mgr.status().active_session is None:
                break

        assert session_mgr.status().active_session is None
        assert fake_voice.disconnected is True
        assert fake_gemini.closed is True
    finally:
        await session_mgr.stop()
        await server.stop()


@pytest.mark.asyncio
async def test_bad_join_payload_rejected_cleanly(short_sock_path, base_config, monkeypatch):
    """Malformed join shouldn't crash the daemon — just return an error response."""
    fake_voice = _FakeVoiceClient()
    fake_gemini = _FakeGeminiLive()
    monkeypatch.setattr("gem_voice.session._make_voice_client", lambda: fake_voice)
    monkeypatch.setattr("gem_voice.session._make_gemini_session", lambda api_key: fake_gemini)

    session_mgr = Session(base_config)
    server = IpcServer(socket_path=short_sock_path, session_manager=session_mgr)
    await server.start()

    try:
        reader, writer = await asyncio.open_unix_connection(short_sock_path)

        # Missing required fields
        bad = {"id": "bad-1", "action": "join", "vc_credentials": {"guild_id": "1"}}
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
