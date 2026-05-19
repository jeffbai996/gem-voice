"""Tests for the optional memory store HTTP client."""
import pytest
import httpx
from unittest.mock import patch

from gem_voice.memory_client import fetch_context


@pytest.mark.asyncio
async def test_no_url_returns_empty():
    result = await fetch_context("anything", base_url=None, timeout_s=1.0)
    assert result == []


@pytest.mark.asyncio
async def test_empty_url_returns_empty():
    result = await fetch_context("anything", base_url="", timeout_s=1.0)
    assert result == []


@pytest.mark.asyncio
async def test_fetches_and_parses():
    captured_urls = []

    async def fake_get(self, url, **kwargs):
        captured_urls.append(url)
        return httpx.Response(200, json={"snippets": ["one", "two", "three"]})

    with patch.object(httpx.AsyncClient, "get", fake_get):
        result = await fetch_context("hello", base_url="http://example.com/m", timeout_s=1.0)
    assert result == ["one", "two", "three"]
    assert any("q=hello" in u for u in captured_urls)


@pytest.mark.asyncio
async def test_http_error_returns_empty():
    async def fake_get(self, url, **kwargs):
        return httpx.Response(500, text="server error")

    with patch.object(httpx.AsyncClient, "get", fake_get):
        result = await fetch_context("q", base_url="http://example.com/m", timeout_s=1.0)
    assert result == []


@pytest.mark.asyncio
async def test_timeout_returns_empty():
    async def fake_get(self, url, **kwargs):
        raise httpx.TimeoutException("timed out")

    with patch.object(httpx.AsyncClient, "get", fake_get):
        result = await fetch_context("q", base_url="http://example.com/m", timeout_s=0.1)
    assert result == []


@pytest.mark.asyncio
async def test_malformed_response_returns_empty():
    async def fake_get(self, url, **kwargs):
        return httpx.Response(200, json={"unexpected_key": "x"})

    with patch.object(httpx.AsyncClient, "get", fake_get):
        result = await fetch_context("q", base_url="http://example.com/m", timeout_s=1.0)
    assert result == []
