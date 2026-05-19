"""gem-voice daemon entrypoint: python -m gem_voice"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys

from gem_voice.config import ConfigError, load_config
from gem_voice.ipc_server import IpcServer
from gem_voice.logging_setup import setup_logging
from gem_voice.session import Session

log = logging.getLogger(__name__)


def _default_socket_path() -> str:
    xdg = os.getenv("XDG_RUNTIME_DIR")
    if xdg:
        return os.path.join(xdg, "gem-voice.sock")
    return "/tmp/gem-voice.sock"


async def _run() -> int:
    try:
        cfg = load_config()
    except ConfigError as e:
        sys.stderr.write(f"config error: {e}\n")
        return 2

    setup_logging(level=cfg.log_level)
    log.info("daemon_starting", extra={"pid": os.getpid()})

    session_mgr = Session(cfg)
    socket_path = cfg.ipc_socket_path or _default_socket_path()
    server = IpcServer(socket_path=socket_path, session_manager=session_mgr)

    await server.start()
    log.info("daemon_ready", extra={"socket": socket_path})

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _signal_handler(sig: signal.Signals) -> None:
        log.info("signal_received", extra={"signal": sig.name})
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler, sig)

    await stop_event.wait()
    log.info("daemon_stopping")
    await session_mgr.stop()
    await server.stop()
    log.info("daemon_stopped")
    return 0


def main() -> None:
    try:
        sys.exit(asyncio.run(_run()))
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
