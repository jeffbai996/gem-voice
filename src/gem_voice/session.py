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
import datetime
import logging
import os
import time
import uuid
from dataclasses import replace

from gem_voice.audio import (
    OpusDecoder,
    OpusEncoder,
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

# Appended to every composed persona. Live calls are conversation, not a
# text channel — any inherited "when to stay silent" rules must not apply.
_VOICE_OVERRIDE = (
    "IMPORTANT — you are on a LIVE VOICE CALL right now. The rules above "
    "about staying silent, opting out of replies, or skipping "
    "acknowledgments apply ONLY to text channels and are suspended for "
    "this call. On a call, silence is a malfunction: ALWAYS respond out "
    "loud to anything the speaker says, including greetings, mic tests, "
    "and small talk. Keep replies short, natural, and conversational — "
    "you are speaking, not writing. No markdown, no lists, no emoji."
)


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
        # Cost guardrails for the Live API (billed per-second of audio in +
        # per-token out). A forgotten session burns money until something
        # explicitly stops it. Two timers:
        #   - idle: if no opus frame arrives for IDLE_TIMEOUT_S, end session
        #     (parent process likely died, network flapped, etc.)
        #   - hard: end session unconditionally after HARD_MAX_DURATION_S
        # Overrides via env. _last_opus_at gets bumped in push_opus().
        self._last_opus_at: float = time.time()
        # Cumulative count of opus frames dropped on queue overflow. Without
        # this, a saturated pipeline silently sheds audio and the only signal
        # is a context-free WARNING — you can't tell one hiccup from sustained
        # loss. Logged as a running total each time a drop happens.
        self._opus_dropped: int = 0
        # Cumulative opus frames dropped because decode/encode RAISED
        # (malformed packet from a lossy Discord voice stream). One bad
        # frame must NOT kill the loop — it used to escape and silently
        # mute the whole session while status() still reported active.
        self._opus_decode_dropped: int = 0
        self._opus_encode_dropped: int = 0
        self._idle_timeout_s = int(os.environ.get("GEM_VOICE_IDLE_TIMEOUT_S", "300"))   # 5 min
        self._hard_max_s = int(os.environ.get("GEM_VOICE_MAX_DURATION_S", "1800"))      # 30 min

    @property
    def events(self) -> asyncio.Queue:
        return self._events

    def status(self) -> SessionStatus:
        return SessionStatus(
            active_session=self._active_session_id,
            uptime_s=int(time.time() - self._started_at),
            gemini_connected=self._gemini is not None,
        )

    async def push_tool_response(self, call_id: str, name: str,
                                 response: dict) -> bool:
        """Forward a parent-executed tool result into the live session."""
        if self._gemini is None:
            return False
        try:
            await self._gemini.send_tool_response(call_id, name, response)
            return True
        except Exception as e:  # noqa: BLE001 — surfaced to caller
            log.warning("tool_response_failed",
                        extra={"tool_name": name, "error": str(e)})
            return False

    async def say(self, text: str) -> None:
        """Speak `text` in gem-voice's configured voice via Gemini TTS, emitting
        it as AUDIO_OUT frames over the SAME encode/broadcast path the live model
        uses — so it sounds identical. Drives /voice speak (text-driven): the
        parent feeds chat-Gemma's reply, we synthesize + stream it back. Needs no
        live Gemini session — the IPC broadcaster runs independently of one."""
        import base64
        text = (text or "").strip()
        if not text:
            return

        def _synthesize() -> bytes:
            # Gemini TTS shares the exact prebuilt voice set as Gemini Live, so
            # voice_name = our configured Live voice => identical voice. The SDK
            # call is blocking, hence asyncio.to_thread below.
            from google import genai
            from google.genai import types as gt
            client = genai.Client(api_key=self._config.gemini_api_key)
            model = os.environ.get("GEM_VOICE_TTS_MODEL", "gemini-2.5-flash-preview-tts")
            resp = client.models.generate_content(
                model=model,
                contents=text,
                config=gt.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=gt.SpeechConfig(
                        voice_config=gt.VoiceConfig(
                            prebuilt_voice_config=gt.PrebuiltVoiceConfig(
                                voice_name=self._config.gemini_voice)))))
            return resp.candidates[0].content.parts[0].inline_data.data  # 24kHz s16le mono

        # The TTS preview API is occasionally flaky (transient 5xx / rate
        # limits) — retry a couple times so a one-off hiccup doesn't silently
        # drop the whole spoken reply (the text reply already posted).
        pcm_24k: bytes | None = None
        for attempt in range(3):
            try:
                pcm_24k = await asyncio.to_thread(_synthesize)
                break
            except Exception as e:  # noqa: BLE001 — a TTS failure must not crash the daemon
                log.warning("say_tts_attempt_failed",
                            extra={"attempt": attempt + 1, "error": str(e)})
                if attempt < 2:
                    await asyncio.sleep(0.4 * (attempt + 1))
        if pcm_24k is None:
            log.warning("say_tts_failed", extra={"chars": len(text)})
            return
        try:
            pcm_48k = resample_pcm16(pcm_24k, src_rate=24000, dst_rate=48000)
        except Exception as e:  # noqa: BLE001
            log.warning("say_resample_failed", extra={"error": str(e)})
            return

        encoder = OpusEncoder()
        frame_size = 1920  # 20ms @ 48kHz int16 mono — matches _encode_loop
        usable = len(pcm_48k) - (len(pcm_48k) % frame_size)
        emitted = 0
        for i in range(0, usable, frame_size):
            try:
                opus = encoder.encode(pcm_48k[i:i + frame_size])
            except Exception:  # noqa: BLE001 — one bad frame must not stop the rest
                continue
            if opus:
                emitted += 1
                await self._events.put(SessionEvent(
                    type=SessionEventType.AUDIO_OUT,
                    data={"b64": base64.b64encode(opus).decode("ascii")},
                ))
        log.info("say_done", extra={"chars": len(text), "opus_frames": emitted,
                                    "pcm24k_bytes": len(pcm_24k)})

    def push_opus(self, frame: bytes) -> None:
        """Feed one 48kHz mono Opus packet from the parent into the pipeline.

        Synchronous + non-blocking. Drops oldest frame on queue overflow to keep
        the model fed with the freshest audio (latency > completeness).
        """
        if self._active_session_id is None:
            return
        self._last_opus_at = time.time()
        try:
            self._opus_in.put_nowait(frame)
        except asyncio.QueueFull:
            try:
                self._opus_in.get_nowait()
                self._opus_in.put_nowait(frame)
                self._opus_dropped += 1
                log.warning("opus_in_overflow_drop_oldest",
                            extra={"dropped_total": self._opus_dropped})
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                pass

    async def start(
        self,
        persona: Persona,
        model_config: ModelConfig,
        owner_user_id: str,
        tools: list[dict] | None = None,
    ) -> str:
        if self._active_session_id is not None:
            raise SessionAlreadyActiveError("session already active")

        # The IPC join payload from the Node side may omit the model (or carry
        # ModelConfig's stale dataclass default, which is a since-deprecated
        # Live model id). The daemon — not the caller — owns its model: fall
        # back to GEMINI_MODEL from config whenever the caller didn't send a
        # real override. Without this, every join used the dead default and
        # Gemini rejected the connection with a 1008 "model not found".
        if not model_config.model or model_config.model == ModelConfig().model:
            model_config = replace(model_config, model=self._config.gemini_model)
            log.info("model_override_from_config", extra={"model": model_config.model})

        composed_persona = await self._compose_persona(persona)

        self._gemini = _make_gemini_session(self._config.gemini_api_key)

        try:
            await self._gemini.connect(composed_persona, model_config, tools)
        except Exception as e:
            log.error("session_start_failed", extra={"error": str(e)})
            await self._teardown()
            raise

        sess_id = f"sess-{uuid.uuid4().hex[:8]}"
        self._active_session_id = sess_id
        self._last_opus_at = time.time()
        session_start_wall = time.time()

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
            asyncio.create_task(self._timeout_watchdog(sess_id, session_start_wall)),
        ]

        log.info(
            "session_started",
            extra={
                "session_id": sess_id,
                "idle_timeout_s": self._idle_timeout_s,
                "hard_max_s": self._hard_max_s,
            },
        )
        return sess_id

    async def _timeout_watchdog(self, sess_id: str, started_at: float) -> None:
        """End the session if it idles too long or runs past the hard cap.

        Both timeouts are cost guardrails — the Live API bills per-second of
        audio. A forgotten or flapping session can quietly rack up cost. The
        watchdog polls once per second; on trigger it puts a SESSION_ENDED
        event with a reason and calls stop().
        """
        try:
            while self._active_session_id == sess_id:
                await asyncio.sleep(1.0)
                if self._active_session_id != sess_id:
                    return
                now = time.time()
                if now - started_at >= self._hard_max_s:
                    log.warning(
                        "session_hard_cap_reached",
                        extra={"session_id": sess_id, "duration_s": int(now - started_at)},
                    )
                    await self._events.put(SessionEvent(
                        type=SessionEventType.SESSION_ENDED,
                        data={"reason": "hard_max_duration", "duration_s": int(now - started_at)},
                    ))
                    await self._teardown()
                    return
                if now - self._last_opus_at >= self._idle_timeout_s:
                    log.warning(
                        "session_idle_timeout",
                        extra={"session_id": sess_id, "idle_s": int(now - self._last_opus_at)},
                    )
                    await self._events.put(SessionEvent(
                        type=SessionEventType.SESSION_ENDED,
                        data={"reason": "idle_timeout", "idle_s": int(now - self._last_opus_at)},
                    ))
                    await self._teardown()
                    return
        except asyncio.CancelledError:
            pass

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
        new_prompt = persona.system_prompt
        # Always tell voice Gemma the current time — she was blind to "now"
        # and hallucinating dates. host-a runs in Springfield (local tz).
        _now = datetime.datetime.now().astimezone().strftime("%A, %B %d %Y, %-I:%M %p %Z")
        new_prompt = (f'{new_prompt}\n\nThe current date and time is {_now}. '
                      'Treat this as "now" — do not guess the date.')
        if persona.memory_query:
            snippets = await fetch_context(
                query=persona.memory_query,
                base_url=self._config.memory_store_url,
                timeout_s=self._config.memory_store_timeout_s,
            )
            if snippets:
                context_block = "\n".join(f"- {s}" for s in snippets)
                new_prompt = (f"{new_prompt}\n\n"
                              f"Relevant context:\n{context_block}")
        # Parent bots hand us their TEXT-channel persona, which usually
        # carries lurking etiquette ("stay silent unless you add value",
        # "opt out of pure acknowledgments"). On a live voice call that
        # etiquette is a bug: the very first smoke test ("testing") made
        # the model conclude silence was the polite reply — perfect
        # pipeline, mute bot. Voice overrides text rules, always.
        new_prompt = f"{new_prompt}\n\n{_VOICE_OVERRIDE}"
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
        """Opus 48kHz → PCM 16kHz mono int16. Pass everything through.

        We deliberately do NOT run a local VAD here. Gemini Live performs its
        own server-side VAD on the continuous audio stream — a redundant local
        VAD silently dropped every frame when the threshold was wrong for the
        actual mic level, and showed up as "frames flowing in over IPC but
        zero send_progress logs on the Gemini side" (the May 21 symptom).
        """
        decoder = OpusDecoder()
        frame_count = 0
        while True:
            try:
                opus = await opus_in.get()
            except asyncio.CancelledError:
                return
            try:
                pcm_48k = decoder.decode(opus)
                pcm_16k = resample_pcm16(pcm_48k, src_rate=48000, dst_rate=16000)
            except Exception:  # noqa: BLE001 — bad opus frame must not kill the loop
                self._opus_decode_dropped += 1
                log.warning("decode_drop",
                            extra={"dropped_total": self._opus_decode_dropped})
                continue
            frame_count += 1
            if frame_count == 1 or frame_count % 100 == 0:
                log.info("decode_loop_progress",
                         extra={"frames_decoded": frame_count,
                                "pcm16k_bytes": len(pcm_16k)})
            await pcm_in.put(pcm_16k)

    async def _encode_loop(
        self,
        pcm_out: asyncio.Queue,
    ) -> None:
        """PCM 24kHz from model → Opus 48kHz frames → emit as AUDIO_OUT events."""
        import base64
        encoder = OpusEncoder()
        frame_size = 1920  # 20ms at 48kHz int16 mono
        emitted = 0
        leftover = b""  # PCM remainder carried between model chunks —
        # chunk boundaries don't align to 20ms frames, and dropping the
        # tail of every chunk shaves audible slivers off the speech
        while True:
            try:
                pcm_24k = await pcm_out.get()
            except asyncio.CancelledError:
                return
            try:
                pcm_48k = leftover + resample_pcm16(
                    pcm_24k, src_rate=24000, dst_rate=48000)
            except Exception:  # noqa: BLE001 — bad model PCM must not kill the loop
                self._opus_encode_dropped += 1
                log.warning("encode_resample_drop",
                            extra={"dropped_total": self._opus_encode_dropped})
                continue
            usable = len(pcm_48k) - (len(pcm_48k) % frame_size)
            leftover = pcm_48k[usable:]
            for i in range(0, usable, frame_size):
                frame = pcm_48k[i:i + frame_size]
                try:
                    opus = encoder.encode(frame)
                except Exception:  # noqa: BLE001 — bad frame must not kill the loop
                    self._opus_encode_dropped += 1
                    log.warning("encode_drop",
                                extra={"dropped_total": self._opus_encode_dropped})
                    continue
                if opus:
                    emitted += 1
                    if emitted == 1 or emitted % 100 == 0:
                        log.info("encode_loop_progress",
                                 extra={"opus_frames_emitted": emitted,
                                        "opus_bytes": len(opus)})
                    await self._events.put(SessionEvent(
                        type=SessionEventType.AUDIO_OUT,
                        data={"b64": base64.b64encode(opus).decode("ascii")},
                    ))
