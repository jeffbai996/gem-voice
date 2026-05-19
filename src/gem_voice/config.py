"""Load and validate daemon config from environment."""
from __future__ import annotations

import os

from dotenv import load_dotenv

from gem_voice.types import Config


class ConfigError(Exception):
    """Raised when required config is missing or malformed."""


def _required(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise ConfigError(f"{key} not set in environment")
    return val


def _optional(key: str, default: str | None = None) -> str | None:
    val = os.getenv(key)
    if val is None or val == "":
        return default
    return val


def _optional_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as e:
        raise ConfigError(f"{key} must be a float, got: {raw!r}") from e


def load_config() -> Config:
    """Read env (and .env if present), return validated Config.

    Raises ConfigError on missing/malformed required values.
    """
    load_dotenv()

    return Config(
        gemini_api_key=_required("GEMINI_API_KEY"),
        discord_owner_user_id=_required("DISCORD_OWNER_USER_ID"),
        gemini_model=_optional("GEMINI_MODEL") or "gemini-3.1-flash-live-preview",
        gemini_voice=_optional("GEMINI_VOICE") or "Aoede",
        gemini_language=_optional("GEMINI_LANGUAGE") or "en-US",
        log_level=_optional("LOG_LEVEL") or "INFO",
        ipc_socket_path=_optional("IPC_SOCKET_PATH"),
        memory_store_url=_optional("MEMORY_STORE_URL"),
        memory_store_timeout_s=_optional_float("MEMORY_STORE_TIMEOUT_S", 2.0),
    )
