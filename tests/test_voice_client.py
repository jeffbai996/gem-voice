"""Tests for the Discord voice client wrapper. Mocks discord.py voice internals."""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from gem_voice.voice_client import VoiceClient, _decode_user_id_from_ssrc_map
from gem_voice.types import VoiceCredentials


def _make_creds() -> VoiceCredentials:
    return VoiceCredentials(
        guild_id="100", channel_id="200", user_id="12345",
        session_id="s", endpoint="e:443", token="t",
    )


@pytest.mark.asyncio
async def test_connect_invokes_discord_handshake(monkeypatch):
    fake_vc = AsyncMock()
    factory = MagicMock(return_value=fake_vc)
    monkeypatch.setattr("gem_voice.voice_client._build_vc", factory)

    creds = _make_creds()
    vc = VoiceClient()
    await vc.connect(creds)

    factory.assert_called_once_with(creds)
    fake_vc.connect_websocket.assert_called_once()


@pytest.mark.asyncio
async def test_recv_loop_filters_to_summoner_ssrc(monkeypatch):
    """Only frames from summoner_ssrc enter the output queue."""
    frames_in = [
        (12345, b"opus-from-summoner-1"),
        (99999, b"opus-from-other-user"),
        (12345, b"opus-from-summoner-2"),
    ]
    sinks_registered = []

    class _FakeVC:
        def listen(self, sink):
            sinks_registered.append(sink)

        async def connect_websocket(self):
            pass

        async def disconnect(self, force=True):
            pass

    fake = _FakeVC()
    monkeypatch.setattr("gem_voice.voice_client._build_vc", lambda creds: fake)

    vc = VoiceClient()
    await vc.connect(_make_creds())

    out: asyncio.Queue = asyncio.Queue()
    task = asyncio.create_task(vc.recv_loop(out, summoner_ssrc=12345))
    await asyncio.sleep(0.05)
    assert len(sinks_registered) == 1
    sink = sinks_registered[0]
    for ssrc, opus in frames_in:
        sink.write(ssrc, opus)
    await asyncio.sleep(0.1)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    got = []
    while not out.empty():
        got.append(out.get_nowait())
    assert got == [b"opus-from-summoner-1", b"opus-from-summoner-2"]


@pytest.mark.asyncio
async def test_send_loop_forwards_opus_to_discord(monkeypatch):
    sent = []

    class _FakeVC:
        def send_audio_packet(self, frame, encode=False):
            sent.append(frame)

        async def connect_websocket(self):
            pass

        async def disconnect(self, force=True):
            pass

    fake = _FakeVC()
    monkeypatch.setattr("gem_voice.voice_client._build_vc", lambda creds: fake)

    vc = VoiceClient()
    await vc.connect(_make_creds())

    opus_in: asyncio.Queue = asyncio.Queue()
    await opus_in.put(b"frame-1")
    await opus_in.put(b"frame-2")
    await opus_in.put(None)
    await vc.send_loop(opus_in)
    assert sent == [b"frame-1", b"frame-2"]


@pytest.mark.asyncio
async def test_disconnect_cleanup(monkeypatch):
    fake = AsyncMock()
    monkeypatch.setattr("gem_voice.voice_client._build_vc", lambda creds: fake)
    vc = VoiceClient()
    await vc.connect(_make_creds())
    await vc.disconnect()
    fake.disconnect.assert_called_once()


def test_decode_user_id_from_ssrc_map_present():
    fake_ws = MagicMock()
    fake_ws.ssrc_map = {12345: {"user_id": "user-abc"}}
    fake_vc = MagicMock()
    fake_vc.ws = fake_ws
    assert _decode_user_id_from_ssrc_map(fake_vc, 12345) == "user-abc"


def test_decode_user_id_from_ssrc_map_missing():
    fake_vc = MagicMock()
    fake_vc.ws = MagicMock()
    fake_vc.ws.ssrc_map = {}
    assert _decode_user_id_from_ssrc_map(fake_vc, 99999) is None
