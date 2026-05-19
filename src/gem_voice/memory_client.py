"""Optional HTTP client for fetching context snippets from an external memory store.

The store is a black box that exposes GET ?q=<query> returning JSON
{"snippets": ["...", "..."]}. If the URL is unset, all fetches no-op.

Failure mode: any error (network, parse, status) returns []. We don't want
the model to wait on the memory store, and starting a session with no
context is always a valid fallback.
"""
from __future__ import annotations

import logging
from urllib.parse import quote

import httpx

log = logging.getLogger(__name__)


async def fetch_context(
    query: str,
    base_url: str | None,
    timeout_s: float,
    top_k: int = 5,
) -> list[str]:
    """Fetch up to top_k context snippets for the query. Returns [] on any failure."""
    if not base_url:
        return []

    url = f"{base_url}?q={quote(query)}&top_k={top_k}"
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.get(url)
    except (httpx.TimeoutException, httpx.HTTPError) as e:
        log.warning("memory_store_request_failed", extra={"error": str(e)})
        return []

    if resp.status_code != 200:
        log.warning("memory_store_non_200", extra={"status": resp.status_code})
        return []

    try:
        data = resp.json()
        snippets = data.get("snippets")
        if not isinstance(snippets, list):
            return []
        return [str(s) for s in snippets[:top_k]]
    except (ValueError, KeyError, TypeError) as e:
        log.warning("memory_store_parse_failed", extra={"error": str(e)})
        return []
