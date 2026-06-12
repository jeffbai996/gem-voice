#!/usr/bin/env python3
"""Probe which Live-capable Gemini models actually grant a bidi session.

The Live API (bidiGenerateContent) has per-model quota buckets separate
from regular generateContent. This lists every model the key exposes
with bidi support, then attempts a minimal live.connect against each
and records the outcome — accepted, quota-refused (429/goAway), or
otherwise rejected. Read-only diagnostics; each successful session is
closed immediately.

Run from the repo root: venv/bin/python scripts/probe_live_models.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import urllib.request

from dotenv import dotenv_values

from google import genai
from google.genai import types as gtypes

TIMEOUT_S = 12


def list_live_models(key: str) -> list[str]:
    url = ("https://generativelanguage.googleapis.com/v1beta/models"
           f"?pageSize=100&key={key}")
    with urllib.request.urlopen(url, timeout=15) as r:
        data = json.load(r)
    return [m["name"].removeprefix("models/")
            for m in data.get("models", [])
            if "bidiGenerateContent" in m.get("supportedGenerationMethods", [])]


async def probe(client: genai.Client, model: str) -> str:
    config = gtypes.LiveConnectConfig(
        response_modalities=["AUDIO"],
    )
    try:
        async with asyncio.timeout(TIMEOUT_S):
            async with client.aio.live.connect(
                    model=model, config=config) as session:
                # ask for a one-word reply so the server must actually
                # run the session, not just accept the socket
                await session.send_client_content(
                    turns=gtypes.Content(
                        role="user",
                        parts=[gtypes.Part(text="Say OK.")]),
                    turn_complete=True)
                async for response in session.receive():
                    sc = getattr(response, "server_content", None)
                    if getattr(response, "go_away", None) is not None:
                        return "GOAWAY: " + repr(response.go_away)[:120]
                    if sc is not None:
                        mt = getattr(sc, "model_turn", None)
                        if mt and getattr(mt, "parts", None):
                            return "WORKS (model responded)"
                        if getattr(sc, "turn_complete", False):
                            return "WORKS (turn completed)"
                return "closed without content"
    except TimeoutError:
        return "timeout (accepted socket, no reply)"
    except Exception as e:  # noqa: BLE001 — classify by message
        msg = str(e)
        if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
            return "QUOTA (429 RESOURCE_EXHAUSTED)"
        if "1008" in msg or "not found" in msg.lower():
            return "rejected (model not found for bidi)"
        return f"error: {msg[:140]}"


async def main() -> None:
    env = dotenv_values(".env")
    key = env.get("GEMINI_API_KEY")
    if not key:
        sys.exit("no GEMINI_API_KEY in .env")
    models = list_live_models(key)
    print(f"{len(models)} live-capable models exposed by the key:\n")
    client = genai.Client(api_key=key)
    for model in models:
        verdict = await probe(client, model)
        print(f"  {model:55} {verdict}")
        await asyncio.sleep(2)  # don't hammer the quota endpoint


if __name__ == "__main__":
    asyncio.run(main())
