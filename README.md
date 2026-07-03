# gem-voice

Token-agnostic Discord voice subprocess. Bring your own bot identity; gem-voice handles the audio loop and LLM session.

## What it does

gem-voice is a long-running daemon that any Discord bot can delegate voice work to. The parent bot owns the Discord identity and the main gateway connection; gem-voice opens the voice WebSocket using per-call credentials handed over a unix-socket IPC, plumbs audio bidirectionally between Discord and a realtime LLM, and emits events back to the parent.

Two modes: a **live call** (`join`/`leave`), full duplex audio+video with the model in real time, and **speak** (`say`/`cancel_say`), a lighter one-shot TTS path for `/voice speak` — the parent hands over text, gem-voice streams it back as audio without opening a full Live session.

One audio process, many parent bots.

## Status

79 unit tests passing across all modules. Live-call mode has been smoke-tested against real Discord voice channels; speak-mode ships with pipelined TTS (parallel chunk synthesis, sentence-level chunking, realtime pacing) and barge-in cancellation (a new message cuts off an in-flight utterance). Session resumption survives Gemini `goAway`/timeout events. Tool calls can be bridged over IPC so the parent's tools are reachable mid-call.

## Requirements

- Python 3.12+
- A Discord bot you already control (gem-voice does not authenticate to Discord on its own)
- A Gemini API key with Live API access
- libopus on the system. macOS: `brew install opus`. Debian/Ubuntu: `apt install libopus0`.

## Install

```bash
git clone https://github.com/<your-fork>/gem-voice
cd gem-voice
python3.12 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env  # fill in GEMINI_API_KEY and DISCORD_OWNER_USER_ID
```

## Run

```bash
source venv/bin/activate
python -m gem_voice
```

The daemon listens on a unix socket (default `$XDG_RUNTIME_DIR/gem-voice.sock`, fallback `/tmp/gem-voice.sock`). Override with `IPC_SOCKET_PATH` in `.env`.

### Session cost guardrails

The Gemini Live API bills per-second of audio in plus per-token out, so a
forgotten session quietly racks up cost. Two timeouts cap session
lifetime; both emit a `SESSION_ENDED` event with a `reason` field when
they fire (`idle_timeout` or `hard_max_duration`):

- `GEM_VOICE_IDLE_TIMEOUT_S` — end session after no opus frame received
  for this long (default `300`, i.e. 5 minutes). Catches dead parent
  processes and network flaps.
- `GEM_VOICE_MAX_DURATION_S` — hard ceiling on session length regardless
  of activity (default `1800`, i.e. 30 minutes). Backstop against
  unexpectedly long sessions.

These guardrails apply to `join`/`leave` live sessions. `say` (speak-mode) is one-shot TTS with no open session to leak — its cost is bounded by the text length per call.

## IPC protocol

Newline-delimited JSON over unix socket. Nine actions:

**`join`** — start a live voice session

```json
{
  "id": "req-001",
  "action": "join",
  "vc_credentials": {
    "guild_id": "...",
    "channel_id": "...",
    "user_id": "...",
    "session_id": "...",
    "endpoint": "us-east-1234.discord.media:443",
    "token": "..."
  },
  "owner_user_id": "...",
  "persona": {"name": "MyBot", "system_prompt": "You are MyBot."},
  "model_config": {"model": "gemini-3.1-flash-live-preview", "voice": "Aoede", "language": "en-US"}
}
```

**`leave`** — end the active session

```json
{"id": "req-002", "action": "leave"}
```

**`status`** — daemon health

```json
{"id": "req-003", "action": "status"}
```

**`audio_in`** / **`video_in`** — stream a frame into an active live session (opus audio frame / video frame for continuous-watch mode). Sent repeatedly while the parent is forwarding Discord voice/video to gem-voice.

**`tool_response`** — return the result of a tool call gem-voice dispatched over IPC mid-session (see tool-call bridge below).

**`say`** — one-shot TTS, independent of `join`/`leave`. Synthesizes `text` and streams it back as `audio_out`, fire-and-forget so a multi-second synthesis never blocks the IPC loop. Powers `/voice speak`.

```json
{"id": "req-004", "action": "say", "text": "it's sunny out", "voice": "Aoede"}
```

`voice` is optional — a per-utterance override of the configured default (backs `/voice type`).

**`cancel_say`** — barge-in: cut off an in-flight `say` synthesis/playback, e.g. because a new message superseded it.

**`think`** — emit a soft thinking-tone audio cue while the model is generating, so speak-mode doesn't sit in dead silence.

While a session is active, gem-voice pushes events on the same socket:

```json
{"event": "user_speech_end", "transcript": "what's the weather"}
{"event": "model_speech_end", "transcript": "it's sunny..."}
{"event": "session_ended", "reason": "leave_requested"}
```

## Parent bot integration example

Your bot intercepts a `/voice join` slash command, captures the voice credentials from `VOICE_STATE_UPDATE` and `VOICE_SERVER_UPDATE` events on its main gateway, and sends them to gem-voice:

```python
import asyncio, json

async def delegate_to_gem_voice(creds, persona):
    reader, writer = await asyncio.open_unix_connection("/tmp/gem-voice.sock")
    payload = {
        "id": "1",
        "action": "join",
        "vc_credentials": creds,
        "owner_user_id": "...",
        "persona": persona,
        "model_config": {},
    }
    writer.write((json.dumps(payload) + "\n").encode())
    await writer.drain()
    ack = json.loads((await reader.readline()).decode())
    # Keep the connection open; events stream while session active.
    async for line in reader:
        event = json.loads(line.decode())
        print(event)
```

## Test

```bash
pytest -q                       # all unit tests
pytest -m integration -q        # integration tests
pytest -m slow                  # real network tests (rare)
```

## systemd

See `systemd/gem-voice.service` for an example unit file with resource limits (`MemoryMax=1G`, `CPUQuota=50%`).

## License

MIT
