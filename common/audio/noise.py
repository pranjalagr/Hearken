"""Background noise under the caller's voice (Part 3, C5).

The noise plays for the whole call, not only under speech: real background
noise doesn't stop when you stop talking, and if it did, the robot could cheat
("noise means the caller is talking").

Levels are set against a fixed speech level instead of each sentence's own
loudness, so the noise stays steady all call long, as it would in a real room.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ITU-T P.56: the usual active speech level on a phone line is about -26 dB
# below full scale. Every caller voice is set to this, so "loud noise" means
# the same thing for every voice.
SPEECH_LEVEL_DBFS = -26.0

# Speech-to-noise ratios in dB: a bigger number means quieter noise.
# Starting guesses (C5); they live here, in one place, to be tuned.
NOISE_LEVELS_DB = {"quiet": 20.0, "medium": 10.0, "loud": 5.0}

# Frames quieter than this (below the loudest frame) count as pauses, not speech.
_ACTIVE_RANGE_DB = 30.0


def _db_to_gain(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def active_rms(samples: np.ndarray, rate: int, frame_ms: int = 20) -> float:
    """Loudness of the speech itself, ignoring the pauses between words.

    A plain average over the whole clip would call a sentence with long
    pauses "quiet". Frames within 30 dB of the loudest one count as speech:
    a simple stand-in for ITU-T P.56, good enough to set levels.
    """
    samples = np.asarray(samples, dtype=np.float64)
    frame = max(1, rate * frame_ms // 1000)
    usable = len(samples) - len(samples) % frame
    if usable == 0:
        return float(np.sqrt(np.mean(samples**2))) if len(samples) else 0.0
    frames = samples[:usable].reshape(-1, frame)
    power = np.mean(frames**2, axis=1)
    loudest = power.max()
    if loudest == 0.0:
        return 0.0
    active = power >= loudest * 10.0 ** (-_ACTIVE_RANGE_DB / 10.0)
    return float(np.sqrt(power[active].mean()))


def set_speech_level(samples: np.ndarray, rate: int, level_dbfs: float = SPEECH_LEVEL_DBFS) -> np.ndarray:
    level = active_rms(samples, rate)
    if level == 0.0:
        return np.asarray(samples, dtype=np.float32).copy()
    return (np.asarray(samples, dtype=np.float64) * (_db_to_gain(level_dbfs) / level)).astype(np.float32)


@dataclass
class NoiseBed:
    """A noise recording looped for the whole call, at a fixed level.

    `start_sample` comes from the task (C13), so A, B and C hear exactly the
    same stretch of noise.
    """

    noise: np.ndarray  # already at the call's sample rate
    speech_to_noise_db: float
    start_sample: int = 0
    speech_level_dbfs: float = SPEECH_LEVEL_DBFS

    def __post_init__(self) -> None:
        noise = np.asarray(self.noise, dtype=np.float64)
        if len(noise) == 0:
            raise ValueError("noise recording is empty")
        rms = float(np.sqrt(np.mean(noise**2)))
        if rms == 0.0:
            raise ValueError("noise recording is silent")
        target = _db_to_gain(self.speech_level_dbfs - self.speech_to_noise_db)
        # The whole recording's RMS, not "active" frames: noise has no pauses
        # to ignore, and the level must be the same wherever the loop starts.
        self._noise = (noise * (target / rms)).astype(np.float32)
        self._position = self.start_sample % len(self._noise)

    def next(self, count: int) -> np.ndarray:
        """The next `count` samples, wrapping around the end of the recording."""
        if count < 0:
            raise ValueError("count must be >= 0")
        out = np.empty(count, dtype=np.float32)
        filled = 0
        while filled < count:
            take = min(count - filled, len(self._noise) - self._position)
            out[filled : filled + take] = self._noise[self._position : self._position + take]
            filled += take
            self._position = (self._position + take) % len(self._noise)
        return out
