# gem-voice

Token-agnostic Discord voice subprocess. Bring your own bot identity; gem-voice handles the audio loop and LLM session.

## Status

v0.1 — under development.

## Quick start

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -e .
cp .env.example .env  # fill in GEMINI_API_KEY and DISCORD_OWNER_USER_ID
python -m gem_voice
```

## Architecture

gem-voice runs as a long-lived daemon. Your existing Discord bot connects to it over a unix socket, hands over per-call voice credentials, and gem-voice does the rest — Discord voice WebSocket, Opus codec, Gemini Live session.

## License

MIT
