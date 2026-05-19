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
    VoiceCredentials,
)

log = logging.getLogger(__name__)


class _SessionManagerProto(Protocol):
    async def start(
        self,
        vc_credentials: VoiceCredentials,
        persona: Persona,
        model_config: ModelConfig,
        owner_user_id: str,
    ) -> str: ...
    async def stop(self) -> bool: ...
    def status(self) -> SessionStatus: ...
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
        return {"id": req_id, "ok": False, "error": f"unknown action: {action!r}"}

    async def _handle_join(self, req_id: str, msg: dict[str, Any]) -> dict[str, Any]:
        try:
            vc = VoiceCredentials(**msg["vc_credentials"])
            persona = Persona(**msg["persona"])
            model_config = ModelConfig(**msg.get("model_config", {}))
            owner_user_id = msg["owner_user_id"]
        except (KeyError, TypeError) as e:
            return {"id": req_id, "ok": False, "error": f"bad join payload: {e}"}

        try:
            session_id = await self.sm.start(vc, persona, model_config, owner_user_id)
        except RuntimeError as e:
            return {"id": req_id, "ok": False, "error": str(e)}
        return {"id": req_id, "ok": True, "session_id": session_id}

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
            "voice_connected": s.voice_connected,
        }

    async def _broadcaster(self) -> None:
        """Pull events from session manager and push to the active client."""
        while True:
            try:
                event: SessionEvent = await self.sm.events.get()
            except asyncio.CancelledError:
                return
            writer = self._active_writer
            if writer is None:
                continue
            try:
                writer.write((json.dumps(event.to_dict()) + "\n").encode())
                await writer.drain()
            except (ConnectionResetError, BrokenPipeError) as e:
                log.warning("ipc_broadcast_failed", extra={"error": str(e)})
                self._active_writer = None
