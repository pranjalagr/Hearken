"""A stand-in voice: a tone pattern as long as the line would take to say.

Until the real voices are picked (week 1) and their lines are made once, the
dummy robot and the director's tests still need sound with a realistic
length and a clear difference from silence. This is that sound. It is never
used for anything a person is asked to judge.
"""

from __future__ import annotations

import numpy as np

from common.audio.pcm import float_to_pcm16, split_pieces
from common.wire.protocol import PIECE_SAMPLES, SAMPLE_RATE

# Roughly how long Hinglish takes per character when spoken at a calm pace.
SECONDS_PER_CHARACTER = 0.06
MIN_SECONDS = 0.3


def synthetic_line(line: str) -> list[bytes]:
    seconds = max(MIN_SECONDS, SECONDS_PER_CHARACTER * len(line))
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    # Two tones inside the phone band, at a normal speech level (about -20 dBFS).
    wave = 0.07 * np.sin(2 * np.pi * 300 * t) + 0.07 * np.sin(2 * np.pi * 620 * t)
    return [float_to_pcm16(piece) for piece in split_pieces(wave.astype(np.float32), PIECE_SAMPLES)]


class SyntheticRecordings:
    """Recordings for the director, made from the stand-in voice."""

    def pieces(self, line: str) -> list[bytes]:
        return synthetic_line(line)


def is_speech(piece: bytes, threshold: float = 0.01) -> bool:
    """True if a piece is louder than near-silence (about -40 dBFS).

    Enough to tell the stand-in voice, or a person on the web phone, from a
    quiet line. A real voice-activity detector comes with the robot's ears.
    """
    samples = np.frombuffer(piece, dtype="<i2").astype(np.float64) / 32768.0
    return float(np.sqrt(np.mean(samples**2))) > threshold
