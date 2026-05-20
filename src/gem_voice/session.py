"""Session manager — the integrator.

Wires VoiceClient, audio pipeline, and GeminiLiveSession into one lifecycle.
Holds exactly one active session in v0.1. Exposes start/stop/status and an
asyncio Queue of SessionEvents that the IPC server pushes to the parent.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid

from gem_voice.audio import (
    OpusDecoder,
    OpusEncoder,
    VAD,
    resample_pcm16,
)
from gem_voice.gemini_live import GeminiLiveSession
from gem_voice.memory_client import fetch_context
from gem_voice.types import (
    Config,
    ModelConfig,
    Persona,
    SessionEvent,
    SessionEventType,
    SessionStatus,
    VoiceCredentials,
)
from gem_voice.voice_client import VoiceClient

log = logging.getLogger(__name__)


class SessionAlreadyActiveError(RuntimeError):
    pass


def _make_voice_client() -> VoiceClient:
    return VoiceClient()


def _make_gemini_session(api_key: str) -> GeminiLiveSession:
    return GeminiLiveSession(api_key=api_key)


class Session:
    """Holds one active voice session over its lifetime."""

    def __init__(self, config: Config):
        self._config = config
        self._started_at = time.time()
        self._active_session_id: str | None = None
        self._voice: VoiceClient | None = None
        self._gemini: GeminiLiveSession | None = None
        self._tasks: list[asyncio.Task] = []
        self._events: asyncio.Queue[SessionEvent] = asyncio.Queue()

    @property
    def events(self) -> asyncio.Queue:
        return self._events

    def status(self) -> SessionStatus:
        return SessionStatus(
            active_session=self._active_session_id,
            uptime_s=int(time.time() - self._started_at),
            gemini_connected=self._gemini is not None,
            voice_connected=self._voice is not None,
        )

    async def start(
        self,
        vc_credentials: VoiceCredentials,
        persona: Persona,
        model_config: ModelConfig,
        owner_user_id: str,
    ) -> str:
        if self._active_session_id is not None:
            raise SessionAlreadyActiveError("session already active")

        composed_persona = await self._compose_persona(persona)

        self._voice = _make_voice_client()
        self._gemini = _make_gemini_session(self._config.gemini_api_key)

        try:
            await self._voice.connect(vc_credentials)
            await self._gemini.connect(composed_persona, model_config)
        except Exception as e:
            log.error("session_start_failed", extra={"error": str(e)})
            await self._teardown()
            raise

        sess_id = f"sess-{uuid.uuid4().hex[:8]}"
        self._active_session_id = sess_id

        opus_in: asyncio.Queue = asyncio.Queue(maxsize=200)
        pcm_in: asyncio.Queue = asyncio.Queue(maxsize=200)
        pcm_out: asyncio.Queue = asyncio.Queue(maxsize=200)
        opus_out: asyncio.Queue = asyncio.Queue(maxsize=200)

        summoner_ssrc = int(vc_credentials.user_id) if vc_credentials.user_id.isdigit() else 0
        self._tasks = [
            asyncio.create_task(self._voice.recv_loop(opus_in, summoner_ssrc=summoner_ssrc)),
            asyncio.create_task(self._decode_loop(opus_in, pcm_in)),
            asyncio.create_task(self._gemini.stream(pcm_in, pcm_out, self._events)),
            asyncio.create_task(self._encode_loop(pcm_out, opus_out)),
            asyncio.create_task(self._voice.send_loop(opus_out)),
        ]

        log.info("session_started", extra={"session_id": sess_id})
        return sess_id

    async def stop(self, emit_event: bool = False) -> bool:
        """Stop the active session. Returns True if there was one to stop.

        emit_event: if True, push a SESSION_ENDED event before teardown. Set
        when the session ends for a reason the parent didn't initiate (vc
        disconnect, model error). Leave False when the parent explicitly
        called 'leave' — they already know the session ended, and racing
        the event against the leave ack causes ordering bugs.
        """
        if self._active_session_id is None:
            return False
        log.info("session_stopping", extra={"session_id": self._active_session_id})
        if emit_event:
            await self._events.put(SessionEvent(
                type=SessionEventType.SESSION_ENDED,
                data={"reason": "leave_requested"},
            ))
        await self._teardown()
        return True

    async def _teardown(self) -> None:
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        self._tasks = []
        if self._voice is not None:
            try:
                await self._voice.disconnect()
            except Exception:
                pass
        if self._gemini is not None:
            try:
                await self._gemini.close()
            except Exception:
                pass
        self._voice = None
        self._gemini = None
        self._active_session_id = None

    async def _compose_persona(self, persona: Persona) -> Persona:
        if not persona.memory_query:
            return persona
        snippets = await fetch_context(
            query=persona.memory_query,
            base_url=self._config.memory_store_url,
            timeout_s=self._config.memory_store_timeout_s,
        )
        if not snippets:
            return persona
        context_block = "\n".join(f"- {s}" for s in snippets)
        new_prompt = f"{persona.system_prompt}\n\nRelevant context:\n{context_block}"
        return Persona(
            name=persona.name,
            system_prompt=new_prompt,
            memory_query=persona.memory_query,
        )

    async def _decode_loop(
        self,
        opus_in: asyncio.Queue,
        pcm_in: asyncio.Queue,
    ) -> None:
        """Opus 48kHz → PCM 16kHz mono int16. Apply VAD."""
        decoder = OpusDecoder()
        vad = VAD(threshold_rms=500)
        while True:
            try:
                opus = await opus_in.get()
            except asyncio.CancelledError:
                return
            pcm_48k = decoder.decode(opus)
            pcm_16k = resample_pcm16(pcm_48k, src_rate=48000, dst_rate=16000)
            if not vad.is_speech(pcm_16k):
                continue
            await pcm_in.put(pcm_16k)

    async def _encode_loop(
        self,
        pcm_out: asyncio.Queue,
        opus_out: asyncio.Queue,
    ) -> None:
        """PCM 24kHz → Opus 48kHz. Slice into 20ms frames."""
        encoder = OpusEncoder()
        frame_size = 1920  # 20ms at 48kHz int16 mono
        while True:
            try:
                pcm_24k = await pcm_out.get()
            except asyncio.CancelledError:
                return
            pcm_48k = resample_pcm16(pcm_24k, src_rate=24000, dst_rate=48000)
            for i in range(0, len(pcm_48k), frame_size):
                frame = pcm_48k[i:i + frame_size]
                if len(frame) < frame_size:
                    break
                opus = encoder.encode(frame)
                if opus:
                    await opus_out.put(opus)
