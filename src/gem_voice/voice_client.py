"""Discord voice WebSocket client.

Wraps discord.py's VoiceClient internals so we can drive it without a main
gateway login. We construct the VoiceClient with the per-call credentials
handed in via IPC and call connect_websocket() directly.

Filters incoming audio to the summoner's SSRC — other speakers in the vc
are ignored, never reach the model.

Status: v0.1 — the integration path for discord.py's voice internals is
inherently version-specific. _StandaloneVoiceClient is a best-effort subclass
that sets the attributes discord.py's VoiceClient.connect_websocket() expects.
If connect_websocket() raises AttributeError at integration time, add the
missing attribute to _StandaloneVoiceClient.__init__ and try again. This is
expected to need iteration during the manual smoke test.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from discord.voice_client import VoiceClient as _DpyVoiceClient

from gem_voice.types import VoiceCredentials

log = logging.getLogger(__name__)


def _decode_user_id_from_ssrc_map(vc: Any, ssrc: int) -> str | None:
    """Look up the Discord user_id behind an SSRC. Returns None if unknown.

    discord.py exposes `vc.ws.ssrc_map` keyed by SSRC → {'user_id': ...}.
    """
    ssrc_map = getattr(getattr(vc, "ws", None), "ssrc_map", None) or {}
    entry = ssrc_map.get(ssrc)
    if entry is None:
        return None
    uid = entry.get("user_id")
    return str(uid) if uid is not None else None


class _StandaloneVoiceClient(_DpyVoiceClient):
    """discord.py VoiceClient that takes credentials directly instead of from a Client.

    discord.py's VoiceClient normally fetches endpoint/token/session_id from the
    main gateway's VOICE_SERVER_UPDATE and VOICE_STATE_UPDATE events. gem-voice
    has no main gateway. So we set the attributes the parent class would have
    set when those events arrived.
    """

    def __init__(self, creds: VoiceCredentials):
        # Skip super().__init__ — it wants a client + channel.
        self.guild_id = int(creds.guild_id)
        self.channel_id = int(creds.channel_id)
        self.user_id = int(creds.user_id)
        self.session_id = creds.session_id
        self.token = creds.token
        self.endpoint = creds.endpoint
        self._connecting = asyncio.Event()
        self._connected = asyncio.Event()


def _build_vc(creds: VoiceCredentials) -> Any:
    """Build a standalone discord.py VoiceClient from our IPC-supplied credentials.

    Wrapped so tests can monkeypatch the factory.
    """
    return _StandaloneVoiceClient(creds)


class _FrameSink:
    """discord.py-compatible audio sink that funnels frames to an asyncio queue."""

    def __init__(self, queue: asyncio.Queue, summoner_ssrc: int, loop: asyncio.AbstractEventLoop):
        self.queue = queue
        self.summoner_ssrc = summoner_ssrc
        self.loop = loop

    def write(self, ssrc: int, data: bytes) -> None:
        if ssrc != self.summoner_ssrc:
            return
        # Sinks may be called from non-asyncio threads; bounce to event loop.
        self.loop.call_soon_threadsafe(self.queue.put_nowait, data)


class VoiceClient:
    """Lifecycle wrapper around discord.py voice. One vc at a time."""

    def __init__(self):
        self._vc: Any = None

    async def connect(self, creds: VoiceCredentials) -> None:
        self._vc = _build_vc(creds)
        await self._vc.connect_websocket()
        log.info("voice_connected", extra={"guild_id": creds.guild_id, "channel_id": creds.channel_id})

    async def recv_loop(self, opus_out: asyncio.Queue, summoner_ssrc: int) -> None:
        """Register a sink that forwards summoner's Opus frames into the queue."""
        loop = asyncio.get_running_loop()
        sink = _FrameSink(opus_out, summoner_ssrc, loop)
        self._vc.listen(sink)
        # Hold until cancelled; discord.py's recv runs on its own thread.
        while True:
            await asyncio.sleep(3600)

    async def send_loop(self, opus_in: asyncio.Queue) -> None:
        """Pull Opus frames from the queue and send to Discord. None = stop."""
        while True:
            frame = await opus_in.get()
            if frame is None:
                return
            try:
                self._vc.send_audio_packet(frame, encode=False)
            except Exception as e:
                log.warning("voice_send_failed", extra={"error": str(e)})
                return

    async def disconnect(self) -> None:
        if self._vc is not None:
            try:
                await self._vc.disconnect(force=True)
            except Exception as e:
                log.warning("voice_disconnect_failed", extra={"error": str(e)})
            finally:
                self._vc = None
