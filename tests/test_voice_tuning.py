import pytest
from gem_voice.gemini_live import _build_live_config
from gem_voice.types import ModelConfig, Persona


def test_native_config_uses_selected_vad_and_omits_language():
    cfg = _build_live_config(Persona("Test", "Speak naturally"), ModelConfig(
        silence_ms=850, prefix_ms=80, start_sensitivity="low", end_sensitivity="low",
        interrupt=False, thinking="low", temperature=0.7, max_tokens=1024))
    vad = cfg.realtime_input_config.automatic_activity_detection
    assert vad.silence_duration_ms == 850
    assert vad.prefix_padding_ms == 80
    assert vad.start_of_speech_sensitivity.value == "START_SENSITIVITY_LOW"
    assert cfg.realtime_input_config.activity_handling.value == "NO_INTERRUPTION"
    assert cfg.speech_config.language_code is None
    assert cfg.thinking_config.thinking_level.value.lower() == "low"
    assert cfg.temperature == 0.7
    assert cfg.max_output_tokens == 1024


def test_plain_live_model_omits_unsupported_thinking():
    cfg = _build_live_config(Persona("Test", "Speak"), ModelConfig(model="gemini-3.8-live"))
    assert cfg.thinking_config is None


@pytest.mark.parametrize("kwargs", [{"silence_ms": -1}, {"prefix_ms": 9000},
    {"temperature": 3}, {"thinking": "bogus"}, {"start_sensitivity": "bogus"},
    {"max_tokens": 0}])
def test_invalid_tuning_rejected(kwargs):
    with pytest.raises(ValueError):
        ModelConfig(**kwargs)

@pytest.mark.asyncio
async def test_pause_sends_paced_silence_and_flushes_only_after_selected_duration(monkeypatch):
    import asyncio
    from gem_voice.gemini_live import GeminiLiveSession
    from tests.test_gemini_live import _FakeLiveSession, _fake_client_factory
    fake = _FakeLiveSession()
    observed = []
    async def send(**kwargs):
        observed.append(kwargs)
    fake.send_realtime_input = send
    monkeypatch.setattr('gem_voice.gemini_live._make_client', lambda _: _fake_client_factory(fake))
    session = GeminiLiveSession('test')
    await session.connect(Persona('Test', 'Speak'), ModelConfig(silence_ms=1500))
    queue = asyncio.Queue()
    await queue.put(b'\x01' * 640)
    ticks = 0
    async def timeout_step(awaitable, timeout):
        nonlocal ticks
        # Exercise production send loop without a two-second wall-clock test.
        awaitable.close()
        ticks += 1
        if ticks == 100:
            await queue.put(None)
        raise asyncio.TimeoutError
    monkeypatch.setattr(asyncio, 'wait_for', timeout_step)
    await session.stream(queue, asyncio.Queue(), asyncio.Queue())
    silence = [x for x in observed if x.get('audio') and x['audio'].data == bytes(640)]
    assert len(silence) == 99
    assert observed[-1] == {'audio_stream_end': True}
    assert ticks == 100  # 2000 ms, greater than the selected 1500 ms silence.


@pytest.mark.asyncio
async def test_invalid_ipc_tuning_returns_error_instead_of_crashing():
    from gem_voice.ipc_server import IpcServer
    server = IpcServer('/tmp/example-voice.sock', None)
    result = await server._handle_join('test', {'persona': {'name': 'Test', 'system_prompt': 'Speak'},
        'owner_user_id': 'example', 'model_config': {'silence_ms': -1}})
    assert result['ok'] is False
