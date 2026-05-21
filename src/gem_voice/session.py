"""Session manager — the integrator.

Wires the audio pipeline and GeminiLiveSession into one lifecycle. Holds
exactly one active session in v0.2.

v0.2 architecture shift: gem-voice no longer talks to Discord's voice
gateway. The parent bot owns that connection (via discord.js / @discordjs/voice
or similar) and streams raw 48kHz Opus frames in over IPC. We decode,
forward to Gemini Live, encode the model's response back to Opus, and
emit it to the parent over the same IPC.

Reasons for the shift: discord.py's voice client requires too much main-gateway
context to drive from arbitrary credentials. @discordjs/voice and similar
Node-side libraries are designed for exactly this use case and are far more
mature.
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
)

log = logging.getLogger(__name__)


class SessionAlreadyActiveError(RuntimeError):
    pass


def _make_gemini_session(api_key: str) -> GeminiLiveSession:
    return GeminiLiveSession(api_key=api_key)


class Session:
    """Holds one active voice session over its lifetime."""

    def __init__(self, config: Config):
        self._config = config
        self._started_at = time.time()
        self._active_session_id: str | None = None
        self._gemini: GeminiLiveSession | None = None
        self._tasks: list[asyncio.Task] = []
        self._events: asyncio.Queue[SessionEvent] = asyncio.Queue()
        # Inbound opus frames from parent — fed via push_opus(), drained by _decode_loop.
        self._opus_in: asyncio.Queue[bytes] = asyncio.Queue(maxsize=200)

    @property
    def events(self) -> asyncio.Queue:
        return self._events

    def status(self) -> SessionStatus:
        return SessionStatus(
            active_session=self._active_session_id,
            uptime_s=int(time.time() - self._started_at),
            gemini_connected=self._gemini is not None,
        )

    def push_opus(self, frame: bytes) -> None:
        """Feed one 48kHz mono Opus packet from the parent into the pipeline.

        Synchronous + non-blocking. Drops oldest frame on queue overflow to keep
        the model fed with the freshest audio (latency > completeness).
        """
        if self._active_session_id is None:
            return
        try:
            self._opus_in.put_nowait(frame)
        except asyncio.QueueFull:
            try:
                self._opus_in.get_nowait()
                self._opus_in.put_nowait(frame)
                log.warning("opus_in_overflow_drop_oldest")
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                pass

    async def start(
        self,
        persona: Persona,
        model_config: ModelConfig,
        owner_user_id: str,
    ) -> str:
        if self._active_session_id is not None:
            raise SessionAlreadyActiveError("session already active")

        composed_persona = await self._compose_persona(persona)

        self._gemini = _make_gemini_session(self._config.gemini_api_key)

        try:
            await self._gemini.connect(composed_persona, model_config)
        except Exception as e:
            log.error("session_start_failed", extra={"error": str(e)})
            await self._teardown()
            raise

        sess_id = f"sess-{uuid.uuid4().hex[:8]}"
        self._active_session_id = sess_id

        # Drain the residual opus_in queue from any previous session.
        while not self._opus_in.empty():
            try:
                self._opus_in.get_nowait()
            except asyncio.QueueEmpty:
                break

        pcm_in: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=200)
        pcm_out: asyncio.Queue[bytes] = asyncio.Queue(maxsize=200)

        self._tasks = [
            asyncio.create_task(self._decode_loop(self._opus_in, pcm_in)),
            asyncio.create_task(self._gemini.stream(pcm_in, pcm_out, self._events)),
            asyncio.create_task(self._encode_loop(pcm_out)),
        ]

        log.info("session_started", extra={"session_id": sess_id})
        return sess_id

    async def stop(self, emit_event: bool = False) -> bool:
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
        if self._gemini is not None:
            try:
                await self._gemini.close()
            except Exception:
                pass
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
    ) -> None:
        """PCM 24kHz from model → Opus 48kHz frames → emit as AUDIO_OUT events."""
        import base64
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
                    await self._events.put(SessionEvent(
                        type=SessionEventType.AUDIO_OUT,
                        data={"b64": base64.b64encode(opus).decode("ascii")},
                    ))
