"""Tests for the shared types module."""
import pytest

from gem_voice.types import (
    Config,
    VoiceCredentials,
    Persona,
    ModelConfig,
    SessionEvent,
    SessionEventType,
    SessionStatus,
)


def test_config_required_fields_only():
    c = Config(
        gemini_api_key="key123",
        discord_owner_user_id="user456",
    )
    assert c.gemini_api_key == "key123"
    assert c.discord_owner_user_id == "user456"
    assert c.gemini_model == "gemini-3.1-flash-live-preview"
    assert c.gemini_voice == "Aoede"
    assert c.gemini_language == "en-US"
    assert c.log_level == "INFO"
    assert c.ipc_socket_path is None
    assert c.memory_store_url is None
    assert c.memory_store_timeout_s == 2.0


def test_voice_credentials_all_fields():
    creds = VoiceCredentials(
        guild_id="g1",
        channel_id="c1",
        user_id="u1",
        session_id="s1",
        endpoint="example.discord.media:443",
        token="vc-token-abc",
    )
    assert creds.guild_id == "g1"
    assert creds.endpoint == "example.discord.media:443"


def test_persona_with_optional_memory_query():
    p = Persona(name="TestBot", system_prompt="You are a test bot.")
    assert p.name == "TestBot"
    assert p.system_prompt == "You are a test bot."
    assert p.memory_query is None

    p2 = Persona(name="X", system_prompt="Y", memory_query="recent context")
    assert p2.memory_query == "recent context"


def test_model_config_defaults():
    m = ModelConfig()
    assert m.model == "gemini-3.1-flash-live-preview"
    assert m.voice == "Aoede"
    assert m.language == "en-US"


def test_session_event_types_are_distinct():
    assert SessionEventType.USER_SPEECH_START != SessionEventType.USER_SPEECH_END
    assert SessionEventType.MODEL_SPEECH_START != SessionEventType.MODEL_SPEECH_END
    assert SessionEventType.SESSION_ENDED != SessionEventType.ERROR


def test_session_event_serialization():
    e = SessionEvent(type=SessionEventType.USER_SPEECH_END, data={"transcript": "hello"})
    d = e.to_dict()
    assert d == {"event": "user_speech_end", "transcript": "hello"}


def test_session_event_no_data():
    e = SessionEvent(type=SessionEventType.MODEL_SPEECH_START)
    d = e.to_dict()
    assert d == {"event": "model_speech_start"}


def test_session_status_idle():
    s = SessionStatus(
        active_session=None,
        uptime_s=10,
        gemini_connected=False,
        voice_connected=False,
    )
    assert s.active_session is None
    assert s.uptime_s == 10
