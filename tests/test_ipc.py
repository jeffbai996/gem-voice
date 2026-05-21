"""Tests for the IPC server. Uses real unix sockets for end-to-end roundtrip."""
import asyncio
import json
import os
import uuid

import pytest

from gem_voice.ipc_server import IpcServer
from gem_voice.types import SessionStatus, SessionEvent, SessionEventType


@pytest.fixture
def short_sock_path():
    """AF_UNIX paths are capped at ~104 bytes on macOS; tmp_path is too deep."""
    p = f"/tmp/gv-test-{uuid.uuid4().hex[:8]}.sock"
    yield p
    try:
        os.unlink(p)
    except FileNotFoundError:
        pass


class _FakeSessionManager:
    """Stand-in for session.py — records calls so IPC tests don't need real audio loop."""

    def __init__(self):
        self.join_called_with = None
        self.leave_called = False
        self._active = None
        self._events: asyncio.Queue = asyncio.Queue()

    async def start(self, persona, model_config, owner_user_id):
        self.join_called_with = {
            "persona": persona,
            "model_config": model_config,
            "owner_user_id": owner_user_id,
        }
        self._active = "sess-test"
        return "sess-test"

    async def stop(self) -> bool:
        self.leave_called = True
        was_active = self._active is not None
        self._active = None
        return was_active

    def status(self) -> SessionStatus:
        return SessionStatus(
            active_session=self._active,
            uptime_s=42,
            gemini_connected=self._active is not None,
        )

    def push_opus(self, frame: bytes) -> None:
        self.pushed_opus = getattr(self, 'pushed_opus', [])
        self.pushed_opus.append(frame)

    @property
    def events(self) -> asyncio.Queue:
        return self._events


async def _send_recv(sock_path: str, payload: dict) -> dict:
    reader, writer = await asyncio.open_unix_connection(sock_path)
    writer.write((json.dumps(payload) + "\n").encode())
    await writer.drain()
    line = await reader.readline()
    writer.close()
    await writer.wait_closed()
    return json.loads(line.decode())


@pytest.mark.asyncio
async def test_join_dispatches_to_session_manager(short_sock_path):
    sock = short_sock_path
    sm = _FakeSessionManager()
    server = IpcServer(socket_path=sock, session_manager=sm)
    await server.start()
    try:
        resp = await _send_recv(sock, {
            "id": "req-1",
            "action": "join",
            "owner_user_id": "owner",
            "persona": {"name": "P", "system_prompt": "be P"},
            "model_config": {"model": "m", "voice": "v", "language": "en-US"},
        })
        assert resp == {"id": "req-1", "ok": True, "session_id": "sess-test"}
        assert sm.join_called_with is not None
        assert sm.join_called_with["owner_user_id"] == "owner"
        assert sm.join_called_with["persona"].name == "P"
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_leave_when_active(short_sock_path):
    sock = short_sock_path
    sm = _FakeSessionManager()
    sm._active = "sess-x"
    server = IpcServer(socket_path=sock, session_manager=sm)
    await server.start()
    try:
        resp = await _send_recv(sock, {"id": "req-2", "action": "leave"})
        assert resp == {"id": "req-2", "ok": True, "was_active": True}
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_leave_when_idle(short_sock_path):
    sock = short_sock_path
    sm = _FakeSessionManager()
    server = IpcServer(socket_path=sock, session_manager=sm)
    await server.start()
    try:
        resp = await _send_recv(sock, {"id": "req-3", "action": "leave"})
        assert resp == {"id": "req-3", "ok": True, "was_active": False}
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_status(short_sock_path):
    sock = short_sock_path
    sm = _FakeSessionManager()
    server = IpcServer(socket_path=sock, session_manager=sm)
    await server.start()
    try:
        resp = await _send_recv(sock, {"id": "req-4", "action": "status"})
        assert resp["id"] == "req-4"
        assert resp["ok"] is True
        assert resp["active_session"] is None
        assert resp["uptime_s"] == 42
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_malformed_json_returns_error(short_sock_path):
    sock = short_sock_path
    sm = _FakeSessionManager()
    server = IpcServer(socket_path=sock, session_manager=sm)
    await server.start()
    try:
        reader, writer = await asyncio.open_unix_connection(sock)
        writer.write(b"{not json}\n")
        await writer.drain()
        line = await reader.readline()
        resp = json.loads(line.decode())
        assert resp["ok"] is False
        assert "parse" in resp["error"].lower() or "json" in resp["error"].lower()
        writer.close()
        await writer.wait_closed()
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_unknown_action(short_sock_path):
    sock = short_sock_path
    sm = _FakeSessionManager()
    server = IpcServer(socket_path=sock, session_manager=sm)
    await server.start()
    try:
        resp = await _send_recv(sock, {"id": "req-5", "action": "wat"})
        assert resp["ok"] is False
        assert "unknown" in resp["error"].lower()
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_events_pushed_to_connected_client(short_sock_path):
    sock = short_sock_path
    sm = _FakeSessionManager()
    server = IpcServer(socket_path=sock, session_manager=sm)
    await server.start()
    try:
        reader, writer = await asyncio.open_unix_connection(sock)
        writer.write((json.dumps({"id": "req-6", "action": "status"}) + "\n").encode())
        await writer.drain()
        await reader.readline()  # consume status response

        await sm.events.put(SessionEvent(
            type=SessionEventType.USER_SPEECH_END,
            data={"transcript": "hello"},
        ))
        line = await asyncio.wait_for(reader.readline(), timeout=2.0)
        event = json.loads(line.decode())
        assert event == {"event": "user_speech_end", "transcript": "hello"}

        writer.close()
        await writer.wait_closed()
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_audio_in_pushes_opus_to_session(short_sock_path):
    """audio_in action decodes b64 and forwards bytes to session.push_opus."""
    import base64
    sock = short_sock_path
    sm = _FakeSessionManager()
    server = IpcServer(socket_path=sock, session_manager=sm)
    await server.start()
    try:
        opus_bytes = b"\xde\xad\xbe\xef"
        b64 = base64.b64encode(opus_bytes).decode("ascii")
        resp = await _send_recv(sock, {"id": "au-1", "action": "audio_in", "b64": b64})
        assert resp["ok"] is True
        assert sm.pushed_opus == [opus_bytes]
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_audio_in_rejects_missing_b64(short_sock_path):
    sock = short_sock_path
    sm = _FakeSessionManager()
    server = IpcServer(socket_path=sock, session_manager=sm)
    await server.start()
    try:
        resp = await _send_recv(sock, {"id": "au-2", "action": "audio_in"})
        assert resp["ok"] is False
        assert "b64" in resp["error"]
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_disconnect_triggers_session_stop(short_sock_path):
    sock = short_sock_path
    sm = _FakeSessionManager()
    sm._active = "sess-z"
    server = IpcServer(socket_path=sock, session_manager=sm)
    await server.start()
    try:
        reader, writer = await asyncio.open_unix_connection(sock)
        writer.write((json.dumps({"id": "x", "action": "status"}) + "\n").encode())
        await writer.drain()
        await reader.readline()
        writer.close()
        await writer.wait_closed()
        await asyncio.sleep(0.2)
        assert sm.leave_called is True
    finally:
        await server.stop()
