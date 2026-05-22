"""Gemini Live API session wrapper.

Wraps google-genai's Live API behind a queue-based interface that matches the
gem-voice audio pipeline. The Live API is bidirectional WebSocket — send raw
PCM in, receive raw PCM out plus turn-boundary events.
"""
from __future__ import annotations

import asyncio
import logging
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


def _build_live_config(persona: Persona, model_config: ModelConfig) -> Any:
    """Construct the LiveConnectConfig for Gemini Live."""
    return genai_types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=persona.system_prompt,
        speech_config=genai_types.SpeechConfig(
            voice_config=genai_types.VoiceConfig(
                prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(
                    voice_name=model_config.voice
                )
            ),
            language_code=model_config.language,
        ),
    )


class GeminiLiveSession:
    """One Gemini Live session over its lifetime."""

    def __init__(self, api_key: str):
        self._api_key = api_key
        self._client = _make_client(api_key)
        self._ctx = None
        self._session = None
        self._connected = False

    async def connect(self, persona: Persona, model_config: ModelConfig) -> None:
        cfg = _build_live_config(persona, model_config)
        self._ctx = self._client.aio.live.connect(model=model_config.model, config=cfg)
        self._session = await self._ctx.__aenter__()
        self._connected = True

    async def stream(
        self,
        pcm_in: asyncio.Queue,
        pcm_out: asyncio.Queue,
        events: asyncio.Queue,
    ) -> None:
        """Run the bidirectional model stream until input sentinel or error.

        - pcm_in:   bytes frames at 16kHz mono int16. None = stop sentinel.
        - pcm_out:  bytes frames at 24kHz mono int16 from the model.
        - events:   SessionEvent objects (turn boundaries, errors).
        """
        if not self._connected or self._session is None:
            raise RuntimeError("connect() must be called before stream()")

        async def _send_loop():
            frame_count = 0
            while True:
                frame = await pcm_in.get()
                if frame is None:
                    log.info("gemini_send_stopped", extra={"frames_sent": frame_count})
                    return
                try:
                    await self._session.send_realtime_input(
                        media=genai_types.Blob(data=frame, mime_type="audio/pcm;rate=16000")
                    )
                    frame_count += 1
                    if frame_count == 1 or frame_count % 100 == 0:
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
            msg_count = 0
            audio_chunk_count = 0
            try:
                async for response in self._session.receive():
                    msg_count += 1
                    server_content = getattr(response, "server_content", None)
                    if server_content is None:
                        # Sample-log unusual responses so we can see what we're
                        # getting from Gemini Live when things look wrong.
                        if msg_count <= 5:
                            log.info("gemini_recv_no_server_content",
                                     extra={"msg_count": msg_count,
                                            "response_attrs": [a for a in dir(response) if not a.startswith("_")][:10]})
                        continue
                    model_turn = getattr(server_content, "model_turn", None)
                    if model_turn is not None:
                        for part in getattr(model_turn, "parts", []) or []:
                            inline = getattr(part, "inline_data", None)
                            if inline is not None and getattr(inline, "data", None):
                                audio_chunk_count += 1
                                if audio_chunk_count <= 3:
                                    log.info("gemini_audio_chunk",
                                             extra={"chunk_n": audio_chunk_count,
                                                    "bytes": len(inline.data),
                                                    "mime": getattr(inline, "mime_type", None)})
                                await pcm_out.put(inline.data)
                    if getattr(server_content, "turn_complete", False):
                        log.info("gemini_turn_complete",
                                 extra={"msgs": msg_count, "audio_chunks": audio_chunk_count})
                        await events.put(SessionEvent(
                            type=SessionEventType.MODEL_SPEECH_END,
                            data={},
                        ))
            except Exception as e:
                log.warning("gemini_recv_failed",
                            extra={"error": str(e), "msgs_received": msg_count,
                                   "audio_chunks": audio_chunk_count})
                await events.put(SessionEvent(
                    type=SessionEventType.ERROR,
                    data={"fatal": True, "message": f"gemini recv failed: {e}"},
                ))

        send_task = asyncio.create_task(_send_loop())
        recv_task = asyncio.create_task(_recv_loop())

        await send_task
        recv_task.cancel()
        try:
            await recv_task
        except asyncio.CancelledError:
            pass

    async def close(self) -> None:
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
