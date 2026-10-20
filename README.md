# gem-voice

Token-agnostic Discord voice subprocess: a parent bot keeps its Discord identity and forwards Opus frames over a unix socket; gem-voice owns the audio loop and the Gemini realtime session.

## Install

Requires Python 3.12+, libopus (`brew install opus` / `apt install libopus0`), a Gemini API key with Live API access, and a Discord bot you control.

```bash
python3.12 -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # set GEMINI_API_KEY and DISCORD_OWNER_USER_ID
```

## Usage

```bash
python -m gem_voice
```

Listens on a unix socket: `IPC_SOCKET_PATH`, else `$XDG_RUNTIME_DIR/gem-voice.sock`, else `/tmp/gem-voice.sock`. Example unit: `systemd/gem-voice.service`.

Modes: live call (`join`/`leave`, full-duplex audio, optional video) and speak (`say`/`cancel_say`, one-shot TTS without a Live session).

One owning parent connection at a time. A second client is rejected while the owner is connected, and its disconnect cannot stop the owner. Send `status` on the owning connection. Disconnecting the owner stops its live session and pending speak playback.

## IPC

Newline-delimited JSON; every request has `id` and `action`.

| action | purpose |
|---|---|
| `join` | start a live session: `owner_user_id`, `persona` (`name`, `system_prompt`), optional `model_config` |
| `leave` | end the live session; also cancels speak playback when no call is open |
| `status` | daemon health |
| `audio_in` / `video_in` | stream an Opus audio / video frame into the live session |
| `tool_response` | result of a tool call gem-voice dispatched over IPC; may carry the `session_id` from `join` |
| `say` | one-shot TTS, fire-and-forget (no ack; never blocks the IPC loop): `text`, optional `voice`, optional `tts_config` (`model`, `style` ≤500 chars); streams back asynchronously as `audio_out` |
| `cancel_say` | cut off an in-flight `say` |
| `think` | accepted and ignored |

`model_config` (all optional): `model`, `voice`, `language`, `silence_ms`, `prefix_ms`, `start_sensitivity`, `end_sensitivity`, `interrupt`, `thinking`, `temperature`, `max_tokens`. Ranges, defaults and valid values: `src/gem_voice/types.py`. Out-of-range values reject the join with `bad join payload`. `tts_config.model` must be a TTS model listed in `src/gem_voice/ipc_server.py`.

Events pushed on the same socket: `user_speech_end`, `model_speech_end`, `session_ended` (with `reason`; terminal events include `session_id`).

```python
reader, writer = await asyncio.open_unix_connection("/tmp/gem-voice.sock")
writer.write((json.dumps({"id": "1", "action": "join", "owner_user_id": "...",
                          "persona": {"name": "MyBot", "system_prompt": "..."},
                          "model_config": {}}) + "\n").encode())
await writer.drain()
async for line in reader:      # ack, then events while the session is active
    print(json.loads(line))
```

## Configuration

Environment, loaded from `.env`. Defaults live in `src/gem_voice/config.py` and `.env.example`. Required: `GEMINI_API_KEY`, `DISCORD_OWNER_USER_ID`.

Tuning (defaults in `src/gem_voice/session.py` and `gemini_live.py`):

- `GEM_VOICE_IDLE_TIMEOUT_S`, `GEM_VOICE_MAX_DURATION_S`: end a live session after no frames / a hard ceiling; emit `session_ended` with reason `idle_timeout` / `hard_max_duration`. Live sessions bill per second, so keep them set. They do not apply to `say`.
- `GEM_VOICE_VIDEO`, `GEM_VOICE_VIDEO_FPS`, `GEM_VOICE_VIDEO_QUEUE`: continuous-watch video (off unless `GEM_VOICE_VIDEO=1`).
- `GEM_VOICE_TTS_MODEL`, `GEM_VOICE_SAY_SYNTH_CONCURRENCY`, `GEM_VOICE_SAY_LEAD_S`: speak mode.

## Test

```bash
pytest -q -m "not slow"   # default gate (.cc-land.json)
pytest -m integration -q  # fake-WS-server integration tests, ~1-5s each
pytest -m slow            # real network
```

## Notes

- `cancel_say`/`leave` cancel playback and queued synthesis, not a provider request already running in a synthesis thread.
- `tool_response` with a `session_id` that does not match the current session is rejected, so a delayed completion cannot enter a replacement call. Clients omitting it still work; send it.
- Older daemon versions transfer ownership or stop the call when another client connects. Do not probe a running daemon with a new connection; use the owning parent or service logs, and restart only when idle.

## License

MIT
