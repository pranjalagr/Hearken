"""Reading and writing sound files (WAV, FLAC), always as mono float32."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile


def read_sound(path: Path) -> tuple[np.ndarray, int]:
    """Returns (mono samples in [-1, 1], sample rate).

    Stereo is averaged to mono: callers and robots both have one microphone.
    """
    data, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    return data.mean(axis=1).astype(np.float32), int(rate)


def write_sound(path: Path, samples: np.ndarray, rate: int) -> None:
    """16-bit, which is what the wire carries; the format follows the file
    name (.wav or .flac). FLAC is lossless and about half the size (C16)."""
    samples = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    soundfile.write(str(path), samples, rate, subtype="PCM_16")
