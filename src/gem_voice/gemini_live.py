"""Gemini Live API session wrapper.

Wraps google-genai's Live API behind a queue-based interface that matches the
gem-voice audio pipeline. The Live API is bidirectional WebSocket — send raw
PCM in, receive raw PCM out plus turn-boundary events.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from google import genai
from google.genai import types as genai_types

from gem_voice.types import (
    ModelConfig,
    Persona,
    SessionEvent,
    SessionEventType,
)

log = logging.getLogger(__name__)


def _make_client(api_key: str):
    """Factory wrapped so tests can monkeypatch it."""
    return genai.Client(api_key=api_key)


def _build_live_config(persona: Persona, model_config: ModelConfig,
                       tools: list[dict] | None = None,
                       resume_handle: str | None = None) -> Any:
    """Construct the LiveConnectConfig for Gemini Live.

    output_audio_transcription / input_audio_transcription give us text-level
    visibility into both sides of the conversation. The model's transcription
    of what it heard tells us whether server-side VAD is working; the
    output transcription tells us what it tried to say.
    """
    # Always give voice web grounding via the built-in google_search tool — text
    # Gemma has it, voice lacked it. On the 3.x live model built-in tools combine
    # with custom function-calling, so google_search grounds alongside the IPC
    # function declarations.
    cfg_tools = [genai_types.Tool(google_search=genai_types.GoogleSearch())]
    if tools:
        # Declarations arrive as plain dicts over IPC (the parent's
        # FunctionDeclaration JSON) — the SDK accepts dict-shaped
        # declarations inside a Tool wrapper.
        cfg_tools.append(genai_types.Tool(function_declarations=tools))
    cfg_kwargs = dict(
        response_modalities=["AUDIO"],
        tools=cfg_tools,
        system_instruction=persona.system_prompt,
        speech_config=genai_types.SpeechConfig(
            voice_config=genai_types.VoiceConfig(
                prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(
                    voice_name=model_config.voice
                )
            ),
            language_code=model_config.language,
        ),
        output_audio_transcription=genai_types.AudioTranscriptionConfig(),
        input_audio_transcription=genai_types.AudioTranscriptionConfig(),
        # Enable session resumption so the server streams us resumption handles.
        # On a goAway/drop we reconnect with the latest handle and the
        # conversation continues. handle=None on first connect = a fresh
        # resumable session.
        session_resumption=genai_types.SessionResumptionConfig(
            handle=resume_handle),
    )
    # Continuous-watch video config — GATED behind GEM_VOICE_VIDEO so audio-only
    # calls are byte-identical to before. When on: cap frame detail (cost/latency)
    # and add sliding-window context compression, which is what lets an
    # audio+video session run past Gemini's ~2-minute uncompressed av-session cap.
    # try/except guards an SDK that predates these fields — falls back to a working
    # audio config rather than failing the connect.
    if os.environ.get("GEM_VOICE_VIDEO", "0") == "1":
        try:
            cfg_kwargs["media_resolution"] = (
                genai_types.MediaResolution.MEDIA_RESOLUTION_LOW)
            cfg_kwargs["context_window_compression"] = (
                genai_types.ContextWindowCompressionConfig(
                    sliding_window=genai_types.SlidingWindow()))
        except AttributeError as e:
            log.warning("video_config_unavailable", extra={"error": str(e)})
    return genai_types.LiveConnectConfig(**cfg_kwargs)


def _synth_thinking_blip(hz: float, gain: float,
                         dur_s: float = 0.18, rate: int = 24000) -> bytes:
    """One soft 'thinking' blip as 24kHz mono s16le PCM (the pcm_out format).

    A Hann-windowed sine — the window fades the blip in and out so it sounds
    like a gentle 'bloop', no clicks. Quiet by design (gain ≪ 1); it's an
    'I'm still here' cue under Gemma's silence, not a notification."""
    import math
    import struct
    n = max(1, int(rate * dur_s))
    amp = max(0.0, min(1.0, gain)) * 32767.0
    out = bytearray()
    for i in range(n):
        window = math.sin(math.pi * i / n) ** 2  # raised-cosine, 0→1→0
        s = amp * window * math.sin(2 * math.pi * hz * i / rate)
        out += struct.pack("<h", int(max(-32768.0, min(32767.0, s))))
    return bytes(out)


class GeminiLiveSession:
    """One Gemini Live session over its lifetime."""

    def __init__(self, api_key: str):
        self._api_key = api_key
        self._client = _make_client(api_key)
        self._ctx = None
        self._session = None
        self._connected = False
        # Session-resumption state. Gemini Live hangs up the WebSocket on a
        # timer (a goAway after ~minutes) — without resuming, the call just
        # dies mid-conversation (Jeff's "doesn't stay alive more than ~3 min").
        # We keep the latest resumption handle the server pushes and reconnect
        # with it; the model context carries over. _stopping guards a
        # deliberate teardown from racing a reconnect.
        self._resume_handle: str | None = None
        self._stopping = False
        # True while the model is "thinking" — the gap between the user's
        # utterance committing (or a tool call) and the model's audio starting.
        # The thinking-tone loop reads this to play a soft cue so a call
        # doesn't feel dead during a pause.
        self._thinking = False
        self._persona: Persona | None = None
        self._model_config: ModelConfig | None = None
        self._tools: list[dict] | None = None

    async def connect(self, persona: Persona, model_config: ModelConfig,
                      tools: list[dict] | None = None) -> None:
        # Fresh start: a new call must never resume a prior conversation's
        # context, and _stopping clears from any earlier teardown.
        self._stopping = False
        self._resume_handle = None
        self._thinking = False
        # Stash config inputs so a mid-call resume rebuilds the same session
        # (system prompt, tools, voice) and just swaps in the handle.
        self._persona = persona
        self._model_config = model_config
        self._tools = tools
        cfg = _build_live_config(persona, model_config, tools,
                                 resume_handle=self._resume_handle)
        self._ctx = self._client.aio.live.connect(model=model_config.model, config=cfg)
        self._session = await self._ctx.__aenter__()
        self._connected = True

    async def _reconnect(self) -> None:
        """Tear down the dropped WS and reconnect using the latest resumption
        handle. The model context carries over; the audio queues (owned by the
        session) are untouched, so mic/playback resume after a brief gap."""
        self._thinking = False  # don't carry a stuck thinking-tone across a drop
        try:
            if self._session is not None:
                await self._session.close()
            if self._ctx is not None:
                await self._ctx.__aexit__(None, None, None)
        except Exception as e:
            log.info("gemini_resume_old_close_failed", extra={"error": str(e)})
        cfg = _build_live_config(self._persona, self._model_config, self._tools,
                                 resume_handle=self._resume_handle)
        self._ctx = self._client.aio.live.connect(
            model=self._model_config.model, config=cfg)
        self._session = await self._ctx.__aenter__()
        self._connected = True
        log.info("gemini_resumed",
                 extra={"handle_tail": (self._resume_handle or "")[-8:]})

    async def stream(
        self,
        pcm_in: asyncio.Queue,
        pcm_out: asyncio.Queue,
        events: asyncio.Queue,
        video_in: asyncio.Queue | None = None,
    ) -> None:
        """Run the bidirectional model stream until input sentinel or error.

        - pcm_in:   bytes frames at 16kHz mono int16. None = stop sentinel.
        - pcm_out:  bytes frames at 24kHz mono int16 from the model.
        - events:   SessionEvent objects (turn boundaries, errors).
        - video_in: optional JPEG frames for continuous-watch video. None (the
                    default) = audio-only, byte-identical to before.
        """
        if not self._connected or self._session is None:
            raise RuntimeError("connect() must be called before stream()")

        async def _send_loop():
            frame_count = 0
            # Frames sent since the last audio_stream_end. Discord stops
            # shipping opus the moment the speaker goes quiet (no comfort
            # noise), so Gemini's server-side VAD never hears trailing
            # silence and waits forever for the utterance to end — session
            # alive, model mute. When the frame stream pauses longer than
            # the gap threshold, tell Gemini the audio stream ended so it
            # commits the turn and replies.
            sent_since_end = 0
            gap_s = float(os.environ.get("GEM_VOICE_UTTERANCE_GAP_S", "0.6"))
            while True:
                try:
                    if sent_since_end:
                        frame = await asyncio.wait_for(pcm_in.get(),
                                                       timeout=gap_s)
                    else:
                        frame = await pcm_in.get()
                except asyncio.TimeoutError:
                    try:
                        await self._session.send_realtime_input(
                            audio_stream_end=True)
                        log.info("gemini_audio_stream_end",
                                 extra={"after_frames": sent_since_end})
                    except Exception as e:
                        log.warning("gemini_stream_end_failed",
                                    extra={"error": str(e)})
                    sent_since_end = 0
                    # Utterance committed → the model is now thinking until its
                    # first audio chunk lands. Drives the thinking-tone cue.
                    self._thinking = True
                    continue
                if frame is None:
                    # Stop sentinel from session.py = deliberate teardown.
                    # Mark it so the stream loop doesn't try to resume.
                    self._stopping = True
                    log.info("gemini_send_stopped", extra={"frames_sent": frame_count})
                    return
                try:
                    # audio= kwarg with Blob is the shape used by the official
                    # gemini-live-api-examples command-line python sample. The
                    # docs show media= too, but audio= is what the working
                    # reference uses and it accepts a plain dict OR a Blob.
                    await self._session.send_realtime_input(
                        audio=genai_types.Blob(data=frame, mime_type="audio/pcm;rate=16000")
                    )
                    frame_count += 1
                    sent_since_end += 1
                    # Was every 100 frames (~16k lines per 10-min session) — far
                    # too chatty. Widen to every 1000 so progress is still
                    # visible without drowning the logs. Keep the frame-1 line as
                    # the "stream is alive" signal.
                    if frame_count == 1 or frame_count % 1000 == 0:
                        log.info("gemini_send_progress",
                                 extra={"frames_sent": frame_count, "frame_bytes": len(frame)})
                except Exception as e:
                    log.warning("gemini_send_failed",
                                extra={"error": str(e), "frames_sent": frame_count})
                    await events.put(SessionEvent(
                        type=SessionEventType.ERROR,
                        data={"fatal": False, "message": f"gemini send failed: {e}"},
                    ))
                    return

        async def _recv_loop():
            # The official example does `while True: turn = session.receive();
            # async for response in turn: ...`. Each call to receive() returns
            # an iterator that completes at end-of-turn — looping it once misses
            # every turn after the first, which matches the symptom we saw
            # (WS closes with code 1000, model silent in our logs).
            msg_count = 0
            audio_chunk_count = 0
            input_transcript_chars = 0
            output_transcript_chars = 0
            try:
                while True:
                    turn = self._session.receive()
                    async for response in turn:
                        msg_count += 1
                        # Resumption handle — the server pushes these
                        # periodically; keep the latest so a drop can resume.
                        sru = getattr(response, "session_resumption_update", None)
                        if sru is not None:
                            if (getattr(sru, "resumable", False)
                                    and getattr(sru, "new_handle", None)):
                                self._resume_handle = sru.new_handle
                            continue
                        tool_call = getattr(response, "tool_call", None)
                        fcs = getattr(tool_call, "function_calls", None) \
                            if tool_call is not None else None
                        if isinstance(fcs, list) and fcs:
                            for fc in fcs:
                                log.info("gemini_tool_call",
                                         extra={"tool_name": fc.name,
                                                "call_id": fc.id})
                                await events.put(SessionEvent(
                                    type=SessionEventType.TOOL_CALL,
                                    data={"call_id": fc.id,
                                          "name": fc.name,
                                          "args": dict(fc.args or {})},
                                ))
                            # Tool round-trip is a "working" gap too — keep the
                            # thinking cue going until the model speaks.
                            self._thinking = True
                            continue
                        server_content = getattr(response, "server_content", None)
                        if (server_content is not None
                                and getattr(server_content, "interrupted",
                                            None) is True):
                            # Barge-in: the speaker talked over the model.
                            # Gemini stops generating; everything already
                            # emitted must die too or "interruption" just
                            # means "she finishes the sentence anyway".
                            drained = 0
                            while not pcm_out.empty():
                                try:
                                    pcm_out.get_nowait()
                                    drained += 1
                                except asyncio.QueueEmpty:
                                    break
                            log.info("gemini_interrupted",
                                     extra={"pcm_chunks_drained": drained})
                            await events.put(SessionEvent(
                                type=SessionEventType.AUDIO_FLUSH,
                                data={},
                            ))
                        if server_content is None:
                            # No content — could be setup_complete, or a goAway
                            # (server-initiated disconnect). Dump the actual payload
                            # so we can see WHY Gemini is hanging up instead of
                            # talking. goAway carries a time_left/reason; a config
                            # rejection shows here too.
                            go_away = getattr(response, "go_away", None)
                            setup_complete = getattr(response, "setup_complete", None)
                            if go_away is not None:
                                log.warning("gemini_go_away",
                                            extra={"msg_count": msg_count,
                                                   "go_away": repr(go_away)[:500]})
                            elif msg_count <= 5:
                                log.info("gemini_recv_no_server_content",
                                         extra={"msg_count": msg_count,
                                                "setup_complete": repr(setup_complete)[:200],
                                                "response_repr": repr(response)[:500]})
                            continue
                        # Audio out from the model.
                        model_turn = getattr(server_content, "model_turn", None)
                        if model_turn is not None:
                            for part in getattr(model_turn, "parts", []) or []:
                                inline = getattr(part, "inline_data", None)
                                if inline is not None and getattr(inline, "data", None):
                                    # Model is speaking now → thinking over.
                                    self._thinking = False
                                    audio_chunk_count += 1
                                    if audio_chunk_count <= 3:
                                        log.info("gemini_audio_chunk",
                                                 extra={"chunk_n": audio_chunk_count,
                                                        "bytes": len(inline.data),
                                                        "mime": getattr(inline, "mime_type", None)})
                                    await pcm_out.put(inline.data)
                        # Server-side ASR of *our* audio. If this stays empty
                        # while frames are flowing, server VAD never accepted
                        # them as speech (mic too quiet, mime wrong, etc).
                        in_tx = getattr(server_content, "input_transcription", None)
                        if in_tx is not None and getattr(in_tx, "text", None):
                            input_transcript_chars += len(in_tx.text)
                            log.info("gemini_input_transcript",
                                     extra={"chars": input_transcript_chars,
                                            "text": in_tx.text[:200]})
                        # ASR of the *model's* speech.
                        out_tx = getattr(server_content, "output_transcription", None)
                        if out_tx is not None and getattr(out_tx, "text", None):
                            output_transcript_chars += len(out_tx.text)
                            log.info("gemini_output_transcript",
                                     extra={"chars": output_transcript_chars,
                                            "text": out_tx.text[:200]})
                        if getattr(server_content, "turn_complete", False):
                            self._thinking = False
                            log.info("gemini_turn_complete",
                                     extra={"msgs": msg_count,
                                            "audio_chunks": audio_chunk_count,
                                            "in_chars": input_transcript_chars,
                                            "out_chars": output_transcript_chars})
                            await events.put(SessionEvent(
                                type=SessionEventType.MODEL_SPEECH_END,
                                data={},
                            ))
                    # Turn iterator ended naturally — loop to receive the next
                    # turn. The connection stays open until close() is called.
            except Exception as e:
                if self._resume_handle and not self._stopping:
                    # goAway / dropped WS but we hold a resumption handle — not
                    # fatal. Return quietly; stream() reconnects and resumes.
                    log.info("gemini_recv_dropped_resumable",
                             extra={"error": str(e)[:200],
                                    "msgs_received": msg_count})
                    return
                log.warning("gemini_recv_failed",
                            extra={"error": str(e), "msgs_received": msg_count,
                                   "audio_chunks": audio_chunk_count,
                                   "in_chars": input_transcript_chars,
                                   "out_chars": output_transcript_chars})
                await events.put(SessionEvent(
                    type=SessionEventType.ERROR,
                    data={"fatal": True, "message": f"gemini recv failed: {e}"},
                ))

        async def _thinking_tone_loop():
            """Soft periodic cue while the model is thinking, so a call doesn't
            feel dead during a pause. Call-only by construction — only the Live
            session (this stream) runs it; /voice speak has no Live session. The
            blip rides pcm_out through the same encode/playback path as the
            model's voice. Knobs are env-tunable for tuning by ear."""
            if os.environ.get("GEM_VOICE_THINKING_TONE", "1") != "1":
                return
            hz = float(os.environ.get("GEM_VOICE_THINKING_TONE_HZ", "396"))
            grace_s = float(os.environ.get("GEM_VOICE_THINKING_TONE_GRACE_S", "0.7"))
            gap_s = float(os.environ.get("GEM_VOICE_THINKING_TONE_GAP_S", "1.3"))
            gain = float(os.environ.get("GEM_VOICE_THINKING_TONE_GAIN", "0.12"))
            blip = _synth_thinking_blip(hz, gain)
            while not self._stopping:
                if not self._thinking:
                    await asyncio.sleep(0.1)
                    continue
                # Thinking just started — hold a grace period so quick replies
                # don't trigger a blip; abort if it clears in the meantime.
                waited = 0.0
                while waited < grace_s and self._thinking and not self._stopping:
                    await asyncio.sleep(0.1)
                    waited += 0.1
                # Then blip on a steady cadence until the model speaks.
                while self._thinking and not self._stopping:
                    await pcm_out.put(blip)
                    slept = 0.0
                    while slept < gap_s and self._thinking and not self._stopping:
                        await asyncio.sleep(0.1)
                        slept += 0.1

        async def _video_send_loop():
            """Stream inbound JPEG frames to the Live session at <=1fps (the model
            caps video at ~1 frame/sec; sending faster only burns tokens). Runs for
            the whole session like the thinking tone — it reads the session-owned
            video_in queue and sends on whatever self._session currently is, so it
            survives WS reconnects (a frame lost in the reconnect gap is fine; video
            is lossy-tolerant). No-op when no video queue is wired (audio-only)."""
            if video_in is None:
                return
            fps = float(os.environ.get("GEM_VOICE_VIDEO_FPS", "1"))
            min_interval = 1.0 / fps if fps > 0 else 1.0
            loop = asyncio.get_running_loop()
            last_sent = 0.0
            frame_count = 0
            while not self._stopping:
                try:
                    frame = await video_in.get()
                except asyncio.CancelledError:
                    return
                if frame is None:   # optional stop sentinel
                    return
                # Freshest-frame-wins: if newer frames queued while we waited,
                # skip to the latest and drop the stale ones.
                while not video_in.empty():
                    try:
                        frame = video_in.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                # Rate-limit to <= target fps.
                wait = min_interval - (loop.time() - last_sent)
                if wait > 0:
                    await asyncio.sleep(wait)
                if self._stopping:
                    return
                if not self._connected or self._session is None:
                    continue   # mid-reconnect — drop this frame, the next one rides
                try:
                    await self._session.send_realtime_input(
                        video=genai_types.Blob(data=frame, mime_type="image/jpeg"))
                    last_sent = loop.time()
                    frame_count += 1
                    if frame_count == 1 or frame_count % 60 == 0:
                        log.info("gemini_video_progress",
                                 extra={"frames_sent": frame_count,
                                        "frame_bytes": len(frame)})
                except Exception as e:  # noqa: BLE001 — a bad/late frame mustn't kill video
                    log.warning("gemini_video_send_failed",
                                extra={"error": str(e), "frames_sent": frame_count})
                    continue

        # The thinking cue + video send loop run for the whole session,
        # independent of the WS so reconnects don't disturb them.
        tone_task = asyncio.create_task(_thinking_tone_loop())
        video_task = asyncio.create_task(_video_send_loop())
        try:
            # Run send+recv until one ends, then decide: a clean stop (the
            # sentinel set self._stopping) ends the session; a recv drop while
            # we hold a resumption handle is a goAway/timeout → reconnect and
            # keep going. The audio queues persist across reconnects (session.py
            # owns them), so the call survives the WS hangup with a brief gap.
            while True:
                send_task = asyncio.create_task(_send_loop())
                recv_task = asyncio.create_task(_recv_loop())
                _, pending = await asyncio.wait(
                    {send_task, recv_task}, return_when=asyncio.FIRST_COMPLETED)
                for tk in pending:
                    tk.cancel()
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)
                if self._stopping:
                    return
                if not self._resume_handle:
                    # Nothing to resume from (e.g. died before the first handle,
                    # a config rejection) — recv already emitted the fatal error.
                    return
                log.info("gemini_resuming",
                         extra={"handle_tail": self._resume_handle[-8:]})
                try:
                    await self._reconnect()
                except Exception as e:
                    log.warning("gemini_resume_failed", extra={"error": str(e)})
                    await events.put(SessionEvent(
                        type=SessionEventType.ERROR,
                        data={"fatal": True,
                              "message": f"gemini resume failed: {e}"}))
                    return
        finally:
            for bg in (tone_task, video_task):
                bg.cancel()
            for bg in (tone_task, video_task):
                try:
                    await bg
                except BaseException:
                    pass

    async def send_tool_response(self, call_id: str, name: str,
                                 response: dict) -> None:
        """Return a function result to the live session. Gemini holds the
        conversational turn open while waiting, then speaks the answer."""
        if self._session is None:
            raise RuntimeError("no live session")
        await self._session.send_tool_response(
            function_responses=[genai_types.FunctionResponse(
                id=call_id, name=name, response=response)])
        log.info("gemini_tool_response_sent",
                 extra={"tool_name": name, "call_id": call_id})

    async def close(self) -> None:
        # Mark stopping first so any in-flight stream() loop won't try to resume
        # a session we're deliberately tearing down.
        self._stopping = True
        if not self._connected:
            return
        try:
            if self._session is not None:
                await self._session.close()
            if self._ctx is not None:
                await self._ctx.__aexit__(None, None, None)
        except Exception as e:
            log.warning("gemini_close_failed", extra={"error": str(e)})
        finally:
            self._connected = False
            self._session = None
            self._ctx = None
