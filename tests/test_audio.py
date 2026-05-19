"""Tests for the audio module: VAD, ring buffer, resample, Opus codec."""
from pathlib import Path

import numpy as np
import pytest

from gem_voice.audio import (
    VAD,
    RingBuffer,
    resample_pcm16,
    mono_to_stereo,
    stereo_to_mono,
    OpusEncoder,
    OpusDecoder,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _read_pcm(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


# --- VAD ----------------------------------------------------------------

def test_vad_detects_silence():
    vad = VAD(threshold_rms=500)
    silence = _read_pcm("silence_16k_1s.pcm")
    frame = silence[:640]  # 20ms at 16kHz int16 mono
    assert vad.is_speech(frame) is False


def test_vad_detects_sine_as_speech():
    vad = VAD(threshold_rms=500)
    sine = _read_pcm("sine_440hz_16k_1s.pcm")
    frame = sine[:640]
    assert vad.is_speech(frame) is True


def test_vad_threshold_tunable():
    sine = _read_pcm("sine_440hz_16k_1s.pcm")
    frame = sine[:640]
    permissive = VAD(threshold_rms=100)
    strict = VAD(threshold_rms=50000)
    assert permissive.is_speech(frame) is True
    assert strict.is_speech(frame) is False


# --- RingBuffer ---------------------------------------------------------

def test_ringbuffer_basic():
    rb = RingBuffer(max_frames=3)
    assert rb.push(b"a") is True
    assert rb.push(b"b") is True
    assert rb.push(b"c") is True
    assert len(rb) == 3
    assert rb.pop() == b"a"
    assert rb.pop() == b"b"
    assert rb.pop() == b"c"
    assert rb.pop() is None


def test_ringbuffer_drop_oldest():
    rb = RingBuffer(max_frames=2, on_overflow="drop_oldest")
    rb.push(b"a")
    rb.push(b"b")
    assert rb.push(b"c") is False  # overflow
    assert rb.pop() == b"b"  # 'a' was evicted
    assert rb.pop() == b"c"


def test_ringbuffer_drop_newest():
    rb = RingBuffer(max_frames=2, on_overflow="drop_newest")
    rb.push(b"a")
    rb.push(b"b")
    assert rb.push(b"c") is False  # 'c' dropped
    assert rb.pop() == b"a"
    assert rb.pop() == b"b"


def test_ringbuffer_invalid_policy():
    with pytest.raises(ValueError):
        RingBuffer(max_frames=1, on_overflow="bogus")


# --- Resample -----------------------------------------------------------

def test_resample_16k_to_48k_length():
    samples = np.zeros(16000, dtype=np.int16).tobytes()
    out = resample_pcm16(samples, src_rate=16000, dst_rate=48000)
    assert len(out) == 48000 * 2


def test_resample_48k_to_16k_length():
    samples = np.zeros(48000, dtype=np.int16).tobytes()
    out = resample_pcm16(samples, src_rate=48000, dst_rate=16000)
    assert len(out) == 16000 * 2


def test_resample_24k_to_48k_length():
    samples = np.zeros(24000, dtype=np.int16).tobytes()
    out = resample_pcm16(samples, src_rate=24000, dst_rate=48000)
    assert len(out) == 48000 * 2


def test_resample_identity():
    src = np.arange(1000, dtype=np.int16).tobytes()
    out = resample_pcm16(src, src_rate=16000, dst_rate=16000)
    assert out == src


def test_resample_sine_preserves_signal():
    """440Hz sine resampled 16k→48k→16k should still be detected as speech by VAD."""
    sine = _read_pcm("sine_440hz_16k_1s.pcm")
    up = resample_pcm16(sine, src_rate=16000, dst_rate=48000)
    down = resample_pcm16(up, src_rate=48000, dst_rate=16000)
    assert len(down) == len(sine)
    vad = VAD(threshold_rms=500)
    assert vad.is_speech(down[:640]) is True


# --- Mono <-> Stereo ----------------------------------------------------

def test_mono_to_stereo_doubles_bytes():
    mono = np.array([1, 2, 3], dtype=np.int16).tobytes()
    stereo = mono_to_stereo(mono)
    assert len(stereo) == len(mono) * 2
    # Each mono sample duplicated into L+R
    arr = np.frombuffer(stereo, dtype=np.int16)
    assert list(arr) == [1, 1, 2, 2, 3, 3]


def test_stereo_to_mono_halves_bytes():
    stereo = np.array([10, 20, 30, 40], dtype=np.int16).tobytes()
    mono = stereo_to_mono(stereo)
    assert len(mono) == len(stereo) // 2
    # Average of L+R per sample pair
    arr = np.frombuffer(mono, dtype=np.int16)
    assert list(arr) == [15, 35]


# --- Opus codec round-trip ----------------------------------------------

def test_opus_roundtrip_mono_48k():
    """Encode 20ms of 48kHz mono sine, decode it back, verify VAD still detects speech.

    discord.opus is fixed at 48kHz stereo internally. Our wrapper upmixes
    mono→stereo before encode and downmixes stereo→mono after decode.
    """
    sine_16k = _read_pcm("sine_440hz_16k_1s.pcm")
    sine_48k = resample_pcm16(sine_16k, src_rate=16000, dst_rate=48000)
    frame_48k_mono = sine_48k[:1920]  # 20ms at 48kHz int16 mono

    encoder = OpusEncoder()
    decoder = OpusDecoder()

    opus_bytes = encoder.encode(frame_48k_mono)
    assert isinstance(opus_bytes, bytes)
    assert len(opus_bytes) > 0
    assert len(opus_bytes) < len(frame_48k_mono)  # Opus compresses

    decoded_mono = decoder.decode(opus_bytes)
    assert len(decoded_mono) == len(frame_48k_mono)

    decoded_16k = resample_pcm16(decoded_mono, src_rate=48000, dst_rate=16000)
    vad = VAD(threshold_rms=500)
    assert vad.is_speech(decoded_16k[:640]) is True


def test_opus_decode_invalid_returns_frame_not_crash():
    """Garbage input → bytes back, no crash.

    discord.opus.Decoder is forgiving — invalid packets get decoded as
    silence-or-noise at the libopus layer, returning a full stereo frame.
    Our wrapper downmixes to mono, so we expect 960 bytes (20ms @ 48kHz).
    """
    decoder = OpusDecoder()
    result = decoder.decode(b"\x00\x01\x02")
    assert isinstance(result, bytes)
    assert len(result) == 960  # 480 mono samples * 2 bytes (int16)
