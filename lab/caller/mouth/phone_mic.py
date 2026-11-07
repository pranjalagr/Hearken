"""The last step before the wire: noise, then the phone's codec (Part 3, C5).

Noise enters the microphone before the phone line squashes the sound, and it plays
for the whole call, also in the caller's silences. The codec isn't linear, so it
runs once, on the sum (common/audio/phone_line.py). It works sample by sample, so
doing it piece by piece gives the same sound as doing it on the whole call.

Runs inside the 20 ms beat, on one 160-sample piece: cheap numpy work only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from common.audio.noise import NOISE_LEVELS_DB, NoiseBed
from common.audio.pcm import float_to_pcm16, pcm16_to_float
from common.audio.phone_line import band_limit, mu_law_round_trip
from common.audio.resample import PHONE_RATE, resample
from common.audio.sound_files import read_sound
from common.wire.protocol import PIECE_SAMPLES


class Mic(Protocol):
    def next_piece(self, tick: int) -> bytes: ...


class PhoneMic:
    def __init__(self, inner: Mic, noise: NoiseBed | None = None) -> None:
        self._inner = inner
        self._noise = noise

    def next_piece(self, tick: int) -> bytes:
        voice = pcm16_to_float(self._inner.next_piece(tick))
        mixed = voice if self._noise is None else voice + self._noise.next(PIECE_SAMPLES)
        return float_to_pcm16(mu_law_round_trip(mixed))


def load_noise(folder: Path, kind: str) -> np.ndarray:
    """A noise recording at the phone's rate and band. The band is applied here
    because it's linear, so voice and noise can each have it before they're added."""
    files = sorted(folder.glob(f"{kind}.*"))
    if not files:
        raise FileNotFoundError(f"no {kind} noise file in {folder}: run `python -m cli.caller_sounds`")
    samples, rate = read_sound(files[0])
    return band_limit(resample(samples, rate, PHONE_RATE))


def make_noise_bed(folder: Path, kind: str, level: str, start_sample: int) -> NoiseBed:
    """C13: the same file, loudness and start point for A, B and C."""
    return NoiseBed(load_noise(folder, kind), NOISE_LEVELS_DB[level], start_sample=start_sample)
