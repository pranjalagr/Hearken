"""Converting between float samples and 16-bit PCM bytes, and cutting sound
into fixed pieces."""

from __future__ import annotations

import numpy as np

# One scale both ways, so a value that came from PCM goes back to exactly the
# same PCM. (Scaling by 32767 one way and 32768 the other would drift by a bit.)
_SCALE = 32768.0


def float_to_pcm16(samples: np.ndarray) -> bytes:
    """Float samples in [-1, 1] -> 16-bit signed little-endian bytes.

    Values outside [-1, 1] are clipped, like a real sound card does, instead
    of wrapping around into a loud crack.
    """
    scaled = np.rint(np.asarray(samples, dtype=np.float64) * _SCALE)
    return np.clip(scaled, -32768, 32767).astype("<i2").tobytes()


def pcm16_to_float(data: bytes) -> np.ndarray:
    if len(data) % 2:
        raise ValueError("16-bit PCM must have an even number of bytes")
    return (np.frombuffer(data, dtype="<i2").astype(np.float32) / _SCALE).astype(np.float32)


def split_pieces(samples: np.ndarray, piece_samples: int) -> list[np.ndarray]:
    """Cut sound into pieces of exactly `piece_samples`, padding the last one
    with silence: the wire only carries whole pieces."""
    if piece_samples <= 0:
        raise ValueError("piece_samples must be positive")
    samples = np.asarray(samples, dtype=np.float32)
    count = -(-len(samples) // piece_samples)  # ceiling division
    padded = np.zeros(count * piece_samples, dtype=np.float32)
    padded[: len(samples)] = samples
    return [padded[i * piece_samples : (i + 1) * piece_samples] for i in range(count)]
