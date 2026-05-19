"""Tests for logging setup."""
import json
import logging

import pytest

from gem_voice.logging_setup import setup_logging


@pytest.fixture(autouse=True)
def _reset_root_logger():
    """Reset root logger between tests so handlers don't accumulate."""
    yield
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)


def test_setup_emits_json_to_stderr(capsys):
    setup_logging(level="INFO")
    log = logging.getLogger("gem_voice.test")
    log.info("hello", extra={"foo": "bar"})

    captured = capsys.readouterr()
    assert captured.err.strip() != ""
    last_line = captured.err.strip().split("\n")[-1]
    parsed = json.loads(last_line)
    assert parsed.get("message") == "hello"
    assert parsed["level"] == "info"
    assert parsed["foo"] == "bar"


def test_setup_respects_level(capsys):
    setup_logging(level="WARNING")
    log = logging.getLogger("gem_voice.test2")
    log.info("should-not-appear")
    log.warning("should-appear")

    captured = capsys.readouterr()
    assert "should-not-appear" not in captured.err
    assert "should-appear" in captured.err


def test_invalid_level_falls_back_to_info(capsys):
    setup_logging(level="BOGUS")
    log = logging.getLogger("gem_voice.test3")
    log.info("info-line")
    captured = capsys.readouterr()
    assert "info-line" in captured.err
