"""Unix-socket NDJSON IPC server.

Protocol:
- Parent connects to IPC_SOCKET_PATH.
- Parent sends JSON commands, one per line.
- Server responds with one JSON line per command.
- Server pushes unsolicited event JSON lines while session active.

One client connection at a time in v0.1. Disconnect = stop active session.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Protocol

from gem_voice.types import (
    ModelConfig,
    Persona,
    SessionEvent,
    SessionStatus,
)

log = logging.getLogger(__name__)


class _SessionManagerProto(Protocol):
    async def start(
        self,
        persona: Persona,
        model_config: ModelConfig,
        owner_user_id: str,
        tools: list[dict] | None = None,
    ) -> str: ...
    async def stop(self) -> bool: ...
    def status(self) -> SessionStatus: ...
    def push_opus(self, frame: bytes) -> None: ...
    async def push_tool_response(self, call_id: str, name: str,
                                 response: dict) -> bool: ...
    async def say(self, text: str, voice: str | None = None) -> None: ...
    @property
    def events(self) -> asyncio.Queue: ...


class IpcServer:
    def __init__(self, socket_path: str, session_manager: _SessionManagerProto):
        self.socket_path = socket_path
        self.sm = session_manager
        self._server: asyncio.AbstractServer | None = None
        self._broadcaster_task: asyncio.Task | None = None
        self._active_writer: asyncio.StreamWriter | None = None

    async def start(self) -> None:
        try:
            os.unlink(self.socket_path)
        except FileNotFoundError:
            pass
        Path(self.socket_path).parent.mkdir(parents=True, exist_ok=True)
        self._server = await asyncio.start_unix_server(
            self._handle_client, path=self.socket_path,
        )
        self._broadcaster_task = asyncio.create_task(self._broadcaster())
        log.info("ipc_listening", extra={"socket": self.socket_path})

    async def stop(self) -> None:
        if self._broadcaster_task is not None:
            self._broadcaster_task.cancel()
            try:
                await self._broadcaster_task
            except asyncio.CancelledError:
                pass
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        try:
            os.unlink(self.socket_path)
        except FileNotFoundError:
            pass

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        peer = writer.get_extra_info("peername") or "<unix>"
        log.info("ipc_client_connected", extra={"peer": str(peer)})
        self._active_writer = writer
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break  # EOF
                resp = await self._dispatch(line)
                writer.write((json.dumps(resp) + "\n").encode())
                await writer.drain()
        except (asyncio.CancelledError, ConnectionResetError):
            pass
        finally:
            log.info("ipc_client_disconnected")
            self._active_writer = None
            try:
                await self.sm.stop()
            except Exception as e:
                log.warning("session_stop_on_disconnect_failed", extra={"error": str(e)})
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _dispatch(self, line: bytes) -> dict[str, Any]:
        try:
            msg = json.loads(line.decode())
        except json.JSONDecodeError as e:
            return {"ok": False, "error": f"json parse error: {e}"}

        req_id = msg.get("id", "")
        action = msg.get("action")
        if action == "join":
            return await self._handle_join(req_id, msg)
        if action == "leave":
            return await self._handle_leave(req_id)
        if action == "status":
            return self._handle_status(req_id)
        if action == "audio_in":
            return self._handle_audio_in(req_id, msg)
        if action == "tool_response":
            return await self._handle_tool_response(req_id, msg)
        if action == "say":
            return self._handle_say(req_id, msg)
        return {"id": req_id, "ok": False, "error": f"unknown action: {action!r}"}

    def _handle_say(self, req_id: str, msg: dict[str, Any]) -> dict[str, Any]:
        """/voice speak: TTS `text` and stream it as audio_out. Fire-and-forget
        so a multi-second synthesis never blocks the IPC dispatch loop."""
        text = msg.get("text")
        if not isinstance(text, str) or not text.strip():
            return {"id": req_id, "ok": False, "error": "say requires non-empty 'text'"}
        # Optional per-utterance voice override (the /voice type pick). Ignore a
        # non-string; say() falls back to the configured default when None.
        voice = msg.get("voice")
        if not isinstance(voice, str) or not voice.strip():
            voice = None
        asyncio.create_task(self.sm.say(text, voice))
        return {"id": req_id, "ok": True}

    async def _handle_join(self, req_id: str, msg: dict[str, Any]) -> dict[str, Any]:
        try:
            persona = Persona(**msg["persona"])
            model_config = ModelConfig(**msg.get("model_config", {}))
            owner_user_id = msg["owner_user_id"]
            tools = msg.get("tools") or None
            if tools is not None and not isinstance(tools, list):
                raise TypeError("tools must be a list of declarations")
        except (KeyError, TypeError) as e:
            return {"id": req_id, "ok": False, "error": f"bad join payload: {e}"}

        try:
            session_id = await self.sm.start(persona, model_config,
                                             owner_user_id, tools)
        except RuntimeError as e:
            return {"id": req_id, "ok": False, "error": str(e)}
        return {"id": req_id, "ok": True, "session_id": session_id}

    def _handle_audio_in(self, req_id: str, msg: dict[str, Any]) -> dict[str, Any]:
        """Inbound Opus frame from parent. base64-encoded in msg['b64']."""
        import base64
        b64 = msg.get("b64")
        if not isinstance(b64, str):
            return {"id": req_id, "ok": False, "error": "audio_in requires 'b64' string"}
        try:
            opus = base64.b64decode(b64)
        except (ValueError, TypeError) as e:
            return {"id": req_id, "ok": False, "error": f"bad b64: {e}"}
        self.sm.push_opus(opus)
        # Don't bother acking each audio_in; that's per-frame chatter. Parent
        # doesn't wait for an ack. Return a minimal ok response anyway so the
        # NDJSON parser stays happy if the parent ever does listen.
        return {"id": req_id, "ok": True}

    async def _handle_tool_response(self, req_id: str,
                                    msg: dict[str, Any]) -> dict[str, Any]:
        """Parent finished executing a tool call — feed the result back."""
        call_id = msg.get("call_id")
        name = msg.get("name")
        response = msg.get("response")
        if not isinstance(call_id, str) or not isinstance(name, str):
            return {"id": req_id, "ok": False,
                    "error": "tool_response requires call_id + name"}
        if not isinstance(response, dict):
            # tolerate plain-string results from the parent's dispatcher
            response = {"result": response}
        ok = await self.sm.push_tool_response(call_id, name, response)
        return {"id": req_id, "ok": ok}

    async def _handle_leave(self, req_id: str) -> dict[str, Any]:
        was_active = await self.sm.stop()
        return {"id": req_id, "ok": True, "was_active": was_active}

    def _handle_status(self, req_id: str) -> dict[str, Any]:
        s = self.sm.status()
        return {
            "id": req_id,
            "ok": True,
            "active_session": s.active_session,
            "uptime_s": s.uptime_s,
            "gemini_connected": s.gemini_connected,
        }

    async def _broadcaster(self) -> None:
        """Pull events from session manager and push to the active client."""
        audio_out_sent = 0
        while True:
            try:
                event: SessionEvent = await self.sm.events.get()
            except asyncio.CancelledError:
                return
            writer = self._active_writer
            if writer is None:
                if event.type.value == "audio_out":
                    log.warning("audio_out_dropped_no_client")
                continue
            try:
                writer.write((json.dumps(event.to_dict()) + "\n").encode())
                await writer.drain()
                if event.type.value == "audio_out":
                    audio_out_sent += 1
                    if audio_out_sent == 1 or audio_out_sent % 100 == 0:
                        log.info("audio_out_broadcast",
                                 extra={"frames_sent": audio_out_sent})
            except (ConnectionResetError, BrokenPipeError) as e:
                log.warning("ipc_broadcast_failed", extra={"error": str(e)})
                self._active_writer = None
