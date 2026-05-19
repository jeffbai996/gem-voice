"""Shared pytest fixtures and global setup."""
import pytest


@pytest.fixture(scope="session", autouse=True)
def _load_opus():
    """Load libopus once per test session.

    discord.py needs an explicit path on macOS where libopus isn't on the
    default ctypes search path. Linux/CI typically auto-loads — discord.opus
    falls back gracefully if load_opus() is called with a name that doesn't
    resolve.
    """
    import discord.opus as opus
    if opus.is_loaded():
        return
    for candidate in (
        "/opt/homebrew/lib/libopus.dylib",  # macOS arm64 homebrew
        "/usr/local/lib/libopus.dylib",      # macOS x86_64 homebrew
        "libopus.so.0",                       # most Linux
        "libopus",                            # last resort
    ):
        try:
            opus.load_opus(candidate)
            if opus.is_loaded():
                return
        except OSError:
            continue
    # Don't fail collection — individual tests that need opus will fail clearly.
