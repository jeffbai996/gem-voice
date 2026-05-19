"""Configure stdlib logging with JSON output for journald."""
from __future__ import annotations

import json
import logging
import sys
from typing import Any


class _JsonFormatter(logging.Formatter):
    """Minimal JSON formatter — one record per line, stderr-bound."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Surface extra={} kwargs that aren't stdlib LogRecord attrs
        skip = {
            "args", "asctime", "created", "exc_info", "exc_text", "filename",
            "funcName", "levelname", "levelno", "lineno", "module", "msecs",
            "message", "msg", "name", "pathname", "process", "processName",
            "relativeCreated", "stack_info", "thread", "threadName",
            "taskName",
        }
        for k, v in record.__dict__.items():
            if k not in skip and not k.startswith("_"):
                payload[k] = v
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(level: str = "INFO") -> None:
    """Install the JSON formatter on the root logger. Idempotent.

    Unknown level strings fall back to INFO.
    """
    numeric = getattr(logging, level.upper(), None)
    if not isinstance(numeric, int):
        numeric = logging.INFO

    root = logging.getLogger()
    root.setLevel(numeric)

    # Remove existing handlers (idempotency — tests call this repeatedly)
    for h in list(root.handlers):
        root.removeHandler(h)

    h = logging.StreamHandler(stream=sys.stderr)
    h.setFormatter(_JsonFormatter())
    root.addHandler(h)
