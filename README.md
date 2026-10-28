# gem-voice

Token-agnostic Discord voice daemon backed by the Gemini Live and Gemini TTS APIs. A parent Discord bot keeps its own bot token, gateway and voice connection, and forwards 48 kHz Opus frames over a Unix socket in newline-delimited JSON; `gem-voice` runs the audio codec and resample pipeline, owns the Gemini session, and streams outbound Opus packets and turn events back over the same socket.

```text
Parent Discord Bot                                  gem-voice daemon
(@discordjs/voice, discord.py, etc.)                (python -m gem_voice)
┌──────────────────────────────┐                    ┌──────────────────────────────────────┐
│ Discord Gateway + Voice UDP  │   NDJSON over      │ Opus 48 kHz stereo <-> mono PCM      │
│                              │   Unix socket      │   in:  48 kHz -> 16 kHz PCM          │
│  join / leave / say ─────────┼───────────────────►│   out: 24 kHz -> 48 kHz PCM -> Opus  │
│  audio_in / video_in ────────┼───────────────────►│                                      │
│  tool_response ──────────────┼───────────────────►│ GeminiLiveSession (WebSocket)        │
│                              │                    │   - server VAD + silence pacing      │
│  ◄───────────────────────────┼─── audio_out       │   - goAway session resumption        │
│  ◄───────────────────────────┼─── audio_flush     │   - GoogleSearch + IPC tool bridge   │
│  ◄───────────────────────────┼─── tool_call       │                                      │
│  ◄───────────────────────────┼─── session_ended   │ Speak mode (Gemini TTS)              │
└──────────────────────────────┘                    │   - sentence chunking + parallel TTS │
                                                    │   - 20 ms paced playback + barge-in  │
                                                    └──────────────────────────────────────┘
```

### Wire session over `$XDG_RUNTIME_DIR/gem-voice.sock`

```jsonc
// -> Parent starts a full-duplex live call
{"id": "1", "action": "join", "owner_user_id": "123456789012345678",
 "persona": {"name": "Bot", "system_prompt": "You are a concise voice assistant."},
 "model_config": {"model": "gemini-3.1-flash-live-preview", "voice": "Aoede", "silence_ms": 700},
 "tools": [{"name": "get_quote", "description": "Latest stock price",
            "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}}}}]}
// <- Daemon acknowledges with the session ID
{"id": "1", "ok": true, "session_id": "sess-9f3a1c02"}

// -> Parent streams 20 ms 48 kHz Opus packets from Discord voice
{"id": "2", "action": "audio_in", "b64": "T2dnUwAC..."}
{"id": "2", "ok": true}

// <- Model calls a parent tool mid-turn; parent returns the result
{"event": "tool_call", "call_id": "call-1", "name": "get_quote", "args": {"symbol": "AAPL"}}
{"id": "3", "action": "tool_response", "session_id": "sess-9f3a1c02",
 "call_id": "call-1", "name": "get_quote", "response": {"symbol": "AAPL", "price": 231.40}}
{"id": "3", "ok": true}

// <- Daemon streams 20 ms 48 kHz Opus packets back as the model speaks
{"event": "audio_out", "b64": "f8fffe..."}
{"event": "model_speech_end"}

// -> Or one-shot TTS without opening a Live session (barge-in preempts any active utterance)
{"id": "4", "action": "say", "text": "AAPL is trading at 231 dollars and 40 cents.", "voice": "Aoede"}
{"id": "4", "ok": true}
```

## How it is built

- **Why a separate process:** Discord voice libraries (especially `@discordjs/voice` on Node) need the parent bot's main gateway state to join a voice channel, while Python has the cleanest `google-genai` Live API and audio-processing tooling. Putting the audio loop behind a Unix socket lets any Discord bot delegate voice without sharing its Discord token.
- **Two modes sharing one output encoder:**
  - **Live call (`join` / `leave`):** Full-duplex audio (and optional 1 fps JPEG video) over a bidirectional Gemini Live WebSocket. Inbound 48 kHz stereo Opus frames are decoded via `libopus`, downmixed to mono, resampled to 16 kHz PCM, and streamed to the model. Outbound 24 kHz PCM from the model is resampled to 48 kHz with a byte-remainder carry buffer across chunks (so chunk boundaries that do not land on a 20 ms frame boundary never clip syllables), upmixed to stereo, and encoded into 20 ms Opus packets.
  - **Speak mode (`say` / `cancel_say`):** One-shot TTS for text replies without a Live session. Input text is split on sentence boundaries: the first sentence flushes on its own for low time-to-first-audio, subsequent sentences accumulate to at least 30 characters, and trailing fragments under 15 characters attach to the previous chunk. Chunks are synthesized in parallel up to `GEM_VOICE_SAY_SYNTH_CONCURRENCY` (default `4`, with 3 retries on transient errors), awaited in order, resampled from 24 kHz to 48 kHz, and emitted on a single 20 ms playback clock with an initial burst lead (`GEM_VOICE_SAY_LEAD_S`, default `0.3` s) to prime the parent's jitter buffer.
- **Barge-in and `audio_flush`:**
  - During a live call, when Gemini's server-side VAD detects the speaker talking over the model (`server_content.interrupted`), `gem-voice` drains its outbound PCM queue and emits `{"event": "audio_flush"}` so the parent drops buffered playback frames immediately instead of finishing the sentence.
  - During speak mode, sending a new `say` or `cancel_say` cancels the active synthesis and pacing tasks and emits `{"event": "audio_flush"}`.
- **Server-side VAD and silence pacing:** `gem-voice` does not gate microphone audio with a local VAD before sending to Gemini. Because Discord stops sending Opus packets when a speaker pauses, the send loop feeds 20 ms frames of 16 kHz silence (`640` zero bytes) at wall-clock pace whenever inbound frames stop arriving, and sends `audio_stream_end=True` once quiet time reaches `max(1200, silence_ms + 500)` ms so server-side turn detection triggers reliably.
- **Session resumption across `goAway` disconnects:** Gemini Live closes WebSockets after several minutes with a `goAway` frame. `GeminiLiveSession` enables `SessionResumptionConfig`, records each `session_resumption_update.new_handle` pushed by the server, and reconnects with the latest handle when the socket drops while preserving the session-owned audio and video queues.
- **Persona composition for voice:** Text-channel bot prompts often include rules to stay silent when a message needs no reply; on a voice call, those rules make the model stay mute on greetings or mic checks. On `join`, `gem-voice` stamps the host's local date and time, optionally fetches up to 5 context snippets from `MEMORY_STORE_URL`, and appends a live-call override suspending text-channel silence rules.
- **Single-owner connection and cost guardrails:**
  - One IPC client connection owns the daemon at a time. A second connection while an owner is connected receives `{"id": "", "ok": false, "error": "another IPC client owns voice"}` and closes without stopping the owner or draining its events. Disconnecting the owner stops its active session and any in-flight `say`.
  - Because Gemini Live bills per second of audio, a watchdog ends an active live session if no audio or video frame arrives for `GEM_VOICE_IDLE_TIMEOUT_S` (`300` s, reason `idle_timeout`) or when total duration reaches `GEM_VOICE_MAX_DURATION_S` (`1800` s, reason `hard_max_duration`).

## IPC protocol

Requests and responses are single-line JSON objects over the Unix socket. Every request carries an `id` and `action`.

### Actions (parent -> `gem-voice`)

| Action | Fields | Response / Behavior |
|---|---|---|
| `join` | `owner_user_id`, `persona` (`name`, `system_prompt`, optional `memory_query`), optional `model_config`, optional `tools` (list of function declaration dicts) | `{"id", "ok": true, "session_id": "sess-..."}` |
| `leave` | none | Stops the live session and any in-flight `say`; returns `{"id", "ok": true, "was_active": bool}` |
| `status` | none | `{"id", "ok": true, "active_session", "uptime_s", "gemini_connected"}` |
| `audio_in` | `b64` (base64 48 kHz Opus frame) | Queues frame (drops oldest on overflow at 200 frames); returns `{"id", "ok": true}` |
| `video_in` | `b64` (base64 JPEG frame) | Queues frame (freshest-frame-wins, sent at `<= GEM_VOICE_VIDEO_FPS`); returns `{"id", "ok": true}` |
| `tool_response` | `call_id`, `name`, `response` (dict or value), optional `session_id` | Forwards tool output to Gemini. A `session_id` that does not match the active session is rejected with `voice session no longer active` |
| `say` | `text`, optional `voice`, optional `tts_config` (`model`, `style` <= 500 chars) | Preempts any active `say`, starts background synthesis, and returns `{"id", "ok": true}` immediately while `audio_out` events stream |
| `cancel_say` | none | Cancels in-flight `say`, emits `audio_flush` if one was active, and returns `{"id", "ok": true, "cancelled": bool}` |
| `think` | none | No-op returning `{"id", "ok": true}` (kept for wire compatibility) |

#### `model_config` options on `join`

All fields are optional; out-of-range values reject the `join` with `bad join payload`:

| Field | Default | Allowed values |
|---|---|---|
| `model` | daemon `GEMINI_MODEL` | Gemini Live model ID |
| `voice` | `"Aoede"` | Gemini prebuilt voice name |
| `language` | `"en-US"` | BCP-47 language code |
| `silence_ms` | `700` | integer `100`–`3000` |
| `prefix_ms` | `100` | integer `0`–`1000` |
| `start_sensitivity` | `"low"` | `"low"` or `"high"` |
| `end_sensitivity` | `"low"` | `"low"` or `"high"` |
| `interrupt` | `true` | boolean |
| `thinking` | `"minimal"` | `"minimal"`, `"low"`, `"medium"`, `"high"` (`gemini-3.8-live-extended-thinking` requires non-minimal; omitted for `gemini-3.8-live`) |
| `temperature` | `null` | number `0`–`2` |
| `max_tokens` | `null` | integer `1`–`8192` |

#### `tts_config` options on `say`

- `model`: one of `gemini-2.5-flash-preview-tts`, `gemini-2.5-pro-preview-tts`, `gemini-3.1-flash-tts-preview` (defaults to `GEM_VOICE_TTS_MODEL`).
- `style`: delivery instructions string, up to 500 characters.

### Events (`gem-voice` -> parent)

| Event | Payload fields | Meaning |
|---|---|---|
| `audio_out` | `b64` | One base64-encoded 20 ms 48 kHz Opus packet for Discord voice playback |
| `audio_flush` | none | Speaker interrupted the model or `say` was preempted/cancelled; drop buffered playback immediately |
| `tool_call` | `call_id`, `name`, `args` | Model requested a function call; reply with `tool_response` |
| `model_speech_end` | none | Model finished its current spoken turn |
| `session_ended` | `session_id`, `reason` (`idle_timeout`, `hard_max_duration`, `leave_requested`), optional `idle_s` / `duration_s` | Live session ended |
| `error` | `fatal` (bool), `message` | Send, receive or session-resumption error |

### Parent bot connection example

```python
import asyncio
import json

async def delegate_to_gem_voice():
    reader, writer = await asyncio.open_unix_connection("/tmp/gem-voice.sock")
    join_req = {
        "id": "1",
        "action": "join",
        "owner_user_id": "123456789012345678",
        "persona": {"name": "MyBot", "system_prompt": "You are MyBot."},
        "model_config": {},
    }
    writer.write((json.dumps(join_req) + "\n").encode())
    await writer.drain()

    ack = json.loads((await reader.readline()).decode())
    # Keep the connection open; events stream on the same socket while active.
    async for line in reader:
        event = json.loads(line.decode())
        print(event)
```

## Install and run

Requires Python 3.12+, system `libopus` (`apt install libopus0` on Debian/Ubuntu, `brew install opus` on macOS), a Gemini API key with Live API access, and a Discord bot you control.

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # set GEMINI_API_KEY and DISCORD_OWNER_USER_ID
python -m gem_voice
```

See `systemd/gem-voice.service` for an example user service unit (`MemoryMax=1G`, `CPUQuota=50%`).

## Configuration

Loaded from the environment and `.env`:

| Variable | Default | Meaning |
|---|---|---|
| `GEMINI_API_KEY` | required | Gemini API key |
| `DISCORD_OWNER_USER_ID` | required | Discord user ID of the call owner |
| `GEMINI_MODEL` | `gemini-3.1-flash-live-preview` | Default Live model when `join` omits `model_config.model` (`.env.example` sets `gemini-live-2.5-flash-preview`) |
| `GEMINI_VOICE` | `Aoede` | Default voice for Live sessions and `say` |
| `GEMINI_LANGUAGE` | `en-US` | Default language code |
| `IPC_SOCKET_PATH` | `$XDG_RUNTIME_DIR/gem-voice.sock` | Unix socket path (falls back to `/tmp/gem-voice.sock` when `XDG_RUNTIME_DIR` is unset) |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `MEMORY_STORE_URL` | unset | Optional HTTP endpoint (`GET <url>?q=<query>&top_k=5` -> `{"snippets": [...]}`) queried when `persona.memory_query` is set |
| `MEMORY_STORE_TIMEOUT_S` | `2.0` | Timeout in seconds for `MEMORY_STORE_URL` |
| `GEM_VOICE_IDLE_TIMEOUT_S` | `300` | End live session after this many seconds with no inbound audio or video frame |
| `GEM_VOICE_MAX_DURATION_S` | `1800` | Hard maximum duration in seconds for a live session |
| `GEM_VOICE_VIDEO` | `0` | `1` enables low-resolution media mode and sliding-window context compression for continuous video |
| `GEM_VOICE_VIDEO_FPS` | `1` | Maximum video frames per second forwarded to Gemini Live |
| `GEM_VOICE_VIDEO_QUEUE` | `30` | Inbound video frames buffered before dropping the oldest |
| `GEM_VOICE_TTS_MODEL` | `gemini-2.5-flash-preview-tts` | Default TTS model for `say` |
| `GEM_VOICE_SAY_SYNTH_CONCURRENCY` | `4` | Maximum TTS sentence chunks synthesized in parallel |
| `GEM_VOICE_SAY_LEAD_S` | `0.3` | Initial burst lead in seconds before 20 ms frame pacing begins in `say` |

## Tests

```bash
pytest -q -m "not slow"    # unit and fake-WebSocket integration suite
pytest -m integration -q   # fake-WebSocket integration tests only
pytest -m slow             # live network tests
```

## License

MIT. See [LICENSE](LICENSE).
