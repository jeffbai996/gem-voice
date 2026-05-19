"""Tests for env-based config loading."""
import pytest

from gem_voice.config import load_config, ConfigError
from gem_voice.types import Config


def _clear_all_env(monkeypatch):
    for k in (
        "GEMINI_API_KEY", "DISCORD_OWNER_USER_ID",
        "GEMINI_MODEL", "GEMINI_VOICE", "GEMINI_LANGUAGE",
        "LOG_LEVEL", "IPC_SOCKET_PATH",
        "MEMORY_STORE_URL", "MEMORY_STORE_TIMEOUT_S",
    ):
        monkeypatch.delenv(k, raising=False)


def test_load_with_required_only(monkeypatch):
    _clear_all_env(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "key-abc")
    monkeypatch.setenv("DISCORD_OWNER_USER_ID", "user-xyz")

    c = load_config()
    assert isinstance(c, Config)
    assert c.gemini_api_key == "key-abc"
    assert c.discord_owner_user_id == "user-xyz"
    assert c.gemini_model == "gemini-3.1-flash-live-preview"
    assert c.log_level == "INFO"
    assert c.ipc_socket_path is None
    assert c.memory_store_url is None
    assert c.memory_store_timeout_s == 2.0


def test_load_missing_required_raises(monkeypatch):
    _clear_all_env(monkeypatch)
    monkeypatch.setenv("DISCORD_OWNER_USER_ID", "user-xyz")
    with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
        load_config()


def test_load_missing_owner_id_raises(monkeypatch):
    _clear_all_env(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "key-abc")
    with pytest.raises(ConfigError, match="DISCORD_OWNER_USER_ID"):
        load_config()


def test_load_all_overrides(monkeypatch):
    _clear_all_env(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("DISCORD_OWNER_USER_ID", "u")
    monkeypatch.setenv("GEMINI_MODEL", "custom-model")
    monkeypatch.setenv("GEMINI_VOICE", "Charon")
    monkeypatch.setenv("GEMINI_LANGUAGE", "zh-CN")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("IPC_SOCKET_PATH", "/tmp/custom.sock")
    monkeypatch.setenv("MEMORY_STORE_URL", "http://example.com/memory")
    monkeypatch.setenv("MEMORY_STORE_TIMEOUT_S", "5.5")

    c = load_config()
    assert c.gemini_model == "custom-model"
    assert c.gemini_voice == "Charon"
    assert c.gemini_language == "zh-CN"
    assert c.log_level == "DEBUG"
    assert c.ipc_socket_path == "/tmp/custom.sock"
    assert c.memory_store_url == "http://example.com/memory"
    assert c.memory_store_timeout_s == 5.5


def test_timeout_invalid_float_raises(monkeypatch):
    _clear_all_env(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("DISCORD_OWNER_USER_ID", "u")
    monkeypatch.setenv("MEMORY_STORE_TIMEOUT_S", "not-a-number")
    with pytest.raises(ConfigError, match="MEMORY_STORE_TIMEOUT_S"):
        load_config()


def test_empty_optional_treated_as_unset(monkeypatch):
    """MEMORY_STORE_URL='' should be None, not the empty string."""
    _clear_all_env(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("DISCORD_OWNER_USER_ID", "u")
    monkeypatch.setenv("MEMORY_STORE_URL", "")
    monkeypatch.setenv("IPC_SOCKET_PATH", "")
    c = load_config()
    assert c.memory_store_url is None
    assert c.ipc_socket_path is None
