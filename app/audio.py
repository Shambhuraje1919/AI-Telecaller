import io
import math
import struct
import wave
from typing import List

try:
    import audioop
except ImportError:
    audioop = None

# Exact ITU-T G.711 mu-law decode lookup table (matches audioop.ulaw2lin 256/256)
_MULAW_DECODE_TABLE = []
for i in range(256):
    val = ~i & 0xFF
    sign = i & 0x80
    exponent = (val >> 4) & 0x07
    mantissa = val & 0x0F
    sample = ((mantissa << 3) + 0x84) << exponent
    sample -= 0x84
    if not sign:
        sample = -sample
    _MULAW_DECODE_TABLE.append(max(-32768, min(32767, sample)))


def mulaw_to_pcm16(mulaw_bytes: bytes) -> bytes:
    """Decode 8-bit mu-law audio bytes into 16-bit linear PCM (little-endian)."""
    if not mulaw_bytes:
        return b""
    if audioop is not None:
        return audioop.ulaw2lin(mulaw_bytes, 2)
    samples = [_MULAW_DECODE_TABLE[b] for b in mulaw_bytes]
    return struct.pack(f"<{len(samples)}h", *samples)


def pcm16_to_mulaw(pcm_bytes: bytes) -> bytes:
    """Encode 16-bit linear PCM audio into 8-bit mu-law audio bytes."""
    if not pcm_bytes:
        return b""
    if audioop is not None:
        return audioop.lin2ulaw(pcm_bytes, 2)
    num_samples = len(pcm_bytes) // 2
    samples = struct.unpack(f"<{num_samples}h", pcm_bytes)
    out = bytearray(num_samples)
    for idx, sample in enumerate(samples):
        sign = 0x80 if sample >= 0 else 0x00
        if sample < 0:
            sample = -sample
        sample = min(sample, 32767)
        sample += 0x84

        exponent = 7
        for exp in range(7):
            if sample <= (0x84 << exp) + (0x7C << exp):
                exponent = exp
                break

        mantissa = (sample >> (exponent + 3)) & 0x0F
        mulaw_byte = ~(sign | (exponent << 4) | mantissa) & 0xFF
        out[idx] = mulaw_byte
    return bytes(out)


def calculate_rms(pcm_bytes: bytes) -> float:
    """Calculate the Root Mean Square (RMS) energy of 16-bit PCM audio."""
    if not pcm_bytes:
        return 0.0
    if audioop is not None:
        return float(audioop.rms(pcm_bytes, 2))
    num_samples = len(pcm_bytes) // 2
    if num_samples == 0:
        return 0.0
    samples = struct.unpack(f"<{num_samples}h", pcm_bytes)
    sum_squares = sum(s * s for s in samples)
    return math.sqrt(sum_squares / num_samples)


def resample_linear(pcm_bytes: bytes, orig_rate: int, target_rate: int) -> bytes:
    """Resample 16-bit mono PCM."""
    if orig_rate == target_rate or not pcm_bytes:
        return pcm_bytes
    if audioop is not None:
        resampled, _ = audioop.ratecv(pcm_bytes, 2, 1, orig_rate, target_rate, None)
        return resampled

    num_samples = len(pcm_bytes) // 2
    samples = struct.unpack(f"<{num_samples}h", pcm_bytes)
    ratio = target_rate / orig_rate
    target_len = int(num_samples * ratio)
    resampled = [0] * target_len

    for i in range(target_len):
        orig_idx = i / ratio
        idx0 = int(orig_idx)
        idx1 = min(idx0 + 1, num_samples - 1)
        weight = orig_idx - idx0
        val = int(samples[idx0] * (1.0 - weight) + samples[idx1] * weight)
        resampled[i] = max(-32768, min(32767, val))

    return struct.pack(f"<{target_len}h", *resampled)


def create_wav_buffer(pcm_bytes: bytes, sample_rate: int = 16000) -> io.BytesIO:
    """Pack raw 16-bit mono PCM into an in-memory WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)       # Mono
        wf.setsampwidth(2)       # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    buf.seek(0)
    return buf


def wav_to_mulaw_8k(wav_bytes: bytes) -> bytes:
    """Convert WAV container audio into 8kHz mu-law for Twilio."""
    if not wav_bytes:
        return b""
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())

        if audioop is not None:
            if n_channels == 2:
                frames = audioop.tomono(frames, sampwidth, 0.5, 0.5)
            if sampwidth != 2:
                frames = audioop.lin2lin(frames, sampwidth, 2)
                sampwidth = 2
            if framerate != 8000:
                frames, _ = audioop.ratecv(frames, sampwidth, 1, framerate, 8000, None)
            return audioop.lin2ulaw(frames, 2)
        else:
            num_samples = len(frames) // (sampwidth * n_channels)
            if sampwidth == 2 and n_channels == 1:
                pcm16 = frames
            else:
                raw_samples = struct.unpack(f"<{num_samples * n_channels}h", frames)
                if n_channels > 1:
                    mono_samples = [
                        (raw_samples[i * n_channels] + raw_samples[i * n_channels + 1]) // 2
                        for i in range(num_samples)
                    ]
                else:
                    mono_samples = list(raw_samples)
                pcm16 = struct.pack(f"<{len(mono_samples)}h", *mono_samples)

            if framerate != 8000:
                pcm16 = resample_linear(pcm16, orig_rate=framerate, target_rate=8000)

            return pcm16_to_mulaw(pcm16)
    except Exception:
        return b""


def chunk_mulaw_frames(mulaw_bytes: bytes, chunk_size: int = 160) -> List[bytes]:
    """Split mu-law audio into 20ms frames (160 bytes at 8000Hz) for Twilio."""
    return [
        mulaw_bytes[i: i + chunk_size]
        for i in range(0, len(mulaw_bytes), chunk_size)
    ]

