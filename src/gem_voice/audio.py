"""Audio pipeline: PCM resample, VAD, ring buffers, Opus codec.

Sample rates in flight:
- Discord voice: 48kHz Opus
- Gemini Live input:  16kHz PCM mono int16
- Gemini Live output: 24kHz PCM mono int16

Frame sizes (20ms standard):
- 48kHz: 960 samples = 1920 bytes (int16)
- 24kHz: 480 samples = 960 bytes
- 16kHz: 320 samples = 640 bytes

Opus codec is handled via discord.py's bundled libopus binding
(discord.opus). See OpusEncoder and OpusDecoder.
"""
from __future__ import annotations

import struct
from collections import deque

import numpy as np


SAMPLE_RATE_DISCORD = 48000
SAMPLE_RATE_MODEL_IN = 16000
SAMPLE_RATE_MODEL_OUT = 24000

FRAME_MS = 20
FRAME_SAMPLES_DISCORD = SAMPLE_RATE_DISCORD * FRAME_MS // 1000   # 960
FRAME_SAMPLES_MODEL_IN = SAMPLE_RATE_MODEL_IN * FRAME_MS // 1000  # 320
FRAME_SAMPLES_MODEL_OUT = SAMPLE_RATE_MODEL_OUT * FRAME_MS // 1000  # 480


class VAD:
    """Energy-based voice activity detector.

    RMS over 20ms PCM int16 mono frames. Threshold tuned for quiet
    headset audio (~200) vs loud room mic (~2000). Default 500 is a
    reasonable starting point for headset-quality input.

    Not a state-of-the-art VAD (no spectral analysis, no frequency
    masking). It exists to suppress silence streaming, not to make
    interruption decisions. Gemini Live does its own turn-detection
    on the server side.
    """

    def __init__(self, threshold_rms: int = 500):
        self.threshold_rms = threshold_rms

    def is_speech(self, pcm_frame: bytes, sample_rate: int = SAMPLE_RATE_MODEL_IN) -> bool:
        if len(pcm_frame) < 2:
            return False
        samples = struct.unpack(f"<{len(pcm_frame) // 2}h", pcm_frame)
        sum_sq = sum(s * s for s in samples)
        rms = (sum_sq / len(samples)) ** 0.5
        return rms >= self.threshold_rms


class RingBuffer:
    """Bounded FIFO byte buffer for audio frames.

    On overflow: drop_oldest (input path) or drop_newest (output path).
    The pipeline prefers losing data over blocking.
    """

    def __init__(self, max_frames: int, on_overflow: str = "drop_oldest"):
        if on_overflow not in ("drop_oldest", "drop_newest"):
            raise ValueError("on_overflow must be drop_oldest or drop_newest")
        self.max_frames = max_frames
        self.on_overflow = on_overflow
        self._frames: deque[bytes] = deque()

    def push(self, frame: bytes) -> bool:
        """Append a frame. Returns True if accepted, False if overflow."""
        if len(self._frames) < self.max_frames:
            self._frames.append(frame)
            return True
        if self.on_overflow == "drop_oldest":
            self._frames.popleft()
            self._frames.append(frame)
            return False
        # drop_newest
        return False

    def pop(self) -> bytes | None:
        if not self._frames:
            return None
        return self._frames.popleft()

    def __len__(self) -> int:
        return len(self._frames)


def resample_pcm16(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Linear-interpolate resample int16 mono PCM. Identity if rates equal.

    Linear interp is cheap and good-enough for voice (the model and Discord
    both apply their own anti-alias filtering). If quality matters more
    than CPU later, swap for scipy.signal.resample_poly.
    """
    if src_rate == dst_rate:
        return pcm
    src = np.frombuffer(pcm, dtype=np.int16)
    n_src = len(src)
    n_dst = int(round(n_src * dst_rate / src_rate))
    src_idx = np.linspace(0, n_src - 1, num=n_dst, endpoint=True, dtype=np.float64)
    dst_f = np.interp(src_idx, np.arange(n_src, dtype=np.float64), src.astype(np.float64))
    return dst_f.astype(np.int16).tobytes()


def mono_to_stereo(mono_pcm: bytes) -> bytes:
    """Duplicate each mono int16 sample into L+R for stereo."""
    arr = np.frombuffer(mono_pcm, dtype=np.int16)
    stereo = np.repeat(arr, 2)
    return stereo.tobytes()


def stereo_to_mono(stereo_pcm: bytes) -> bytes:
    """Average L+R int16 channels into mono."""
    arr = np.frombuffer(stereo_pcm, dtype=np.int16).reshape(-1, 2)
    # Convert to int32 before averaging to avoid overflow, then cast back.
    mono = arr.astype(np.int32).mean(axis=1).astype(np.int16)
    return mono.tobytes()


class OpusEncoder:
    """Wraps discord.opus.Encoder.

    discord.opus is fixed at 48kHz stereo. Our wrapper accepts 48kHz mono
    int16 input and upmixes to stereo before encoding. Output is one Opus
    packet per 20ms frame.
    """

    SAMPLES_PER_FRAME = 960  # 20ms at 48kHz

    def __init__(self):
        from discord.opus import Encoder
        self._enc = Encoder()

    def encode(self, pcm_48k_mono: bytes) -> bytes:
        """Encode one 20ms mono frame (1920 bytes) to one Opus packet."""
        stereo = mono_to_stereo(pcm_48k_mono)
        return self._enc.encode(stereo, self.SAMPLES_PER_FRAME)


class OpusDecoder:
    """Wraps discord.opus.Decoder. Returns mono PCM from stereo Opus output.

    libopus is forgiving — invalid packets decode to silence-shaped frames
    at the C layer, so we don't need a try/except for "garbage in." The
    decode is at fixed 48kHz stereo internally; we downmix to mono for the
    rest of the pipeline.
    """

    SAMPLES_PER_FRAME = 960

    def __init__(self):
        from discord.opus import Decoder
        self._dec = Decoder()

    def decode(self, opus_packet: bytes) -> bytes:
        """Decode one Opus packet to 48kHz mono int16."""
        stereo = self._dec.decode(opus_packet)
        return stereo_to_mono(stereo)
