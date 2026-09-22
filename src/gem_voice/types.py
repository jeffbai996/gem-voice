"""Shared dataclasses and enums for gem-voice.

All cross-module data shapes live here. Modules import from this; they do not
define shared types of their own.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


@dataclass(frozen=True)
class Config:
    """Daemon-wide configuration, loaded once from env at startup."""
    gemini_api_key: str
    discord_owner_user_id: str
    gemini_model: str = "gemini-3.1-flash-live-preview"
    gemini_voice: str = "Aoede"
    gemini_language: str = "en-US"
    log_level: str = "INFO"
    ipc_socket_path: str | None = None
    memory_store_url: str | None = None
    memory_store_timeout_s: float = 2.0


@dataclass(frozen=True)
class Persona:
    """Per-call persona definition. Comes from parent over IPC."""
    name: str
    system_prompt: str
    memory_query: str | None = None


@dataclass(frozen=True)
class ModelConfig:
    """Per-call selection; an omitted model inherits the daemon configuration."""
    model: str | None = None
    voice: str = "Aoede"
    language: str = "en-US"


    silence_ms: int = 700
    prefix_ms: int = 100
    start_sensitivity: str = "low"
    end_sensitivity: str = "low"
    interrupt: bool = True
    thinking: str = "minimal"
    temperature: float | None = None
    max_tokens: int | None = None

    def __post_init__(self) -> None:
        for key, low, high in [("silence_ms", 100, 3000), ("prefix_ms", 0, 1000),
                                ("temperature", 0, 2), ("max_tokens", 1, 8192)]:
            value = getattr(self, key)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not low <= value <= high
                                      or (key != "temperature" and not isinstance(value, int))):
                raise ValueError(f"invalid {key}")
        if self.thinking not in {"minimal", "low", "medium", "high"}:
            raise ValueError("invalid thinking level")
        if self.start_sensitivity not in {"low", "high"} or self.end_sensitivity not in {"low", "high"}:
            raise ValueError("invalid speech sensitivity")
        if not isinstance(self.interrupt, bool):
            raise ValueError("invalid interruption setting")
        if self.model == "gemini-3.8-live-extended-thinking" and self.thinking == "minimal":
            raise ValueError("extended thinking requires low, medium, or high")


class SessionEventType(str, Enum):
    USER_SPEECH_START = "user_speech_start"
    USER_SPEECH_END = "user_speech_end"
    MODEL_SPEECH_START = "model_speech_start"
    MODEL_SPEECH_END = "model_speech_end"
    SESSION_ENDED = "session_ended"
    ERROR = "error"
    # Audio frames flowing back to parent (model speech). Each event carries
    # a base64-encoded 48kHz mono Opus packet in data['b64'].
    AUDIO_OUT = "audio_out"
    # Gemini requested a function call mid-conversation. data carries
    # {id, name, args}; the parent executes and replies with a
    # `tool_response` IPC action.
    TOOL_CALL = "tool_call"
    # The speaker interrupted the model mid-reply (barge-in). The parent
    # must drop all buffered playback immediately — the daemon has already
    # flushed its own outbound queue.
    AUDIO_FLUSH = "audio_flush"


@dataclass(frozen=True)
class SessionEvent:
    """Event emitted by gem-voice to parent over IPC during an active session."""
    type: SessionEventType
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"event": self.type.value}
        out.update(self.data)
        return out


@dataclass(frozen=True)
class SessionStatus:
    """Snapshot of daemon state for the IPC `status` command."""
    active_session: str | None
    uptime_s: int
    gemini_connected: bool
