"""The model's view of the sound, and the rough phone lines it learns on (Part 4, step 5).

Features: a log-mel picture of the last 2 s of the caller's sound, at the phone's 8 kHz.
Rough lines: every clip is made new each time it's used, so the model never learns one
clean voice or one noise: a little faster or slower, louder or quieter, sometimes echo,
background noise also before the voice, and always the phone's band and codec.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from common.audio.noise import NOISE_LEVELS_DB, NoiseBed, set_speech_level
from common.audio.phone_line import band_limit, mu_law_round_trip
from common.audio.resample import PHONE_RATE

RATE = PHONE_RATE  # 8 kHz: what a phone line carries
WIN, HOP, N_FFT = 200, 80, 256  # 25 ms frames every 10 ms
N_MELS = 40
WINDOW_S = 2.0  # the model hears the last 2 s; a live robot keeps that much of the caller's line
FRAMES = int(WINDOW_S * RATE / HOP)  # 200
# Background before the voice, so every window of WINDOW_S is real line sound, never padding:
# a live robot's buffer holds the line's noise from before the caller spoke, not silence.
LEAD_S = WINDOW_S + 0.2
TAIL_S = 0.5
FLOOR = 1e-6
NOISE_KINDS = ("traffic", "kitchen", "crowd")


def _mel_filters() -> np.ndarray:
    """Triangular filters spaced evenly on the mel scale, from 0 to 4 kHz."""
    to_mel = lambda hz: 2595 * np.log10(1 + hz / 700)  # noqa: E731
    to_hz = lambda mel: 700 * (10 ** (mel / 2595) - 1)  # noqa: E731
    edges = to_hz(np.linspace(to_mel(0), to_mel(RATE / 2), N_MELS + 2))
    bins = np.fft.rfftfreq(N_FFT, 1 / RATE)
    filters = np.zeros((N_MELS, len(bins)), dtype=np.float32)
    for m in range(N_MELS):
        left, mid, right = edges[m], edges[m + 1], edges[m + 2]
        filters[m] = np.clip(np.minimum((bins - left) / (mid - left), (right - bins) / (right - mid)), 0, None)
    return filters


MEL = _mel_filters()
HANN = np.hanning(WIN).astype(np.float32)


def log_mel(audio: np.ndarray) -> np.ndarray:
    """[N_MELS, frames]. Frame i covers samples i*HOP to i*HOP+WIN."""
    audio = np.asarray(audio, dtype=np.float32)
    count = 1 + (len(audio) - WIN) // HOP
    if count <= 0:
        return np.full((N_MELS, 0), np.log(FLOOR), dtype=np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(audio, WIN)[::HOP][:count] * HANN
    power = np.abs(np.fft.rfft(frames, n=N_FFT)) ** 2
    return np.log(MEL @ power.T + FLOOR).astype(np.float32)


def frame_ending_at(seconds: float) -> int:
    """The last frame that ends by `seconds`: what the model has heard at that moment."""
    return int((seconds * RATE - WIN) // HOP)


def window(features: np.ndarray, seconds: float) -> np.ndarray:
    """The last FRAMES frames heard by `seconds`. Raises if there isn't that much sound."""
    last = frame_ending_at(seconds)
    first = last - FRAMES + 1
    if first < 0 or last >= features.shape[1]:
        raise ValueError(f"no full {WINDOW_S} s window ends at {seconds:.2f} s")
    return features[:, first:last + 1]


@dataclass(frozen=True)
class Roughness:
    speed: float = 1.0  # above 1 is slower, as the clip is stretched to that many times its length
    gain_db: float = 0.0
    echo: bool = False
    noise: str | None = None
    noise_level: str = "medium"
    noise_start: int = 0


def random_roughness(rng: np.random.Generator) -> Roughness:
    noisy = rng.random() > 0.2
    return Roughness(speed=float(rng.uniform(0.9, 1.1)), gain_db=float(rng.uniform(-10, 3)), echo=bool(rng.random() < 0.3),
                     noise=str(rng.choice(NOISE_KINDS)) if noisy else None, noise_level=str(rng.choice(list(NOISE_LEVELS_DB))),
                     noise_start=int(rng.integers(0, 1_000_000)))


# The fixed lines every test is marked on, so two models are compared on the same sound.
TEST_LINES = {
    "phone": Roughness(),
    "noisy": Roughness(noise="traffic", noise_level="medium", noise_start=12_345),
    "rough": Roughness(gain_db=-6, echo=True, noise="crowd", noise_level="loud", noise_start=54_321),
}


def room_echo(voice: np.ndarray, seed: int = 0, seconds: float = 0.35) -> np.ndarray:
    """A small room: the voice plus fading reflections, as when a phone is held away."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(RATE * seconds)) / RATE
    tail = rng.normal(0, 1, len(t)) * np.exp(-t * 6.9 / seconds)
    tail[0] = 1.0
    from scipy.signal import fftconvolve

    out = fftconvolve(voice, tail / np.abs(tail).sum() * 4)[: len(voice)]
    return (out / max(1e-6, float(np.abs(out).max())) * float(np.abs(voice).max())).astype(np.float32)


class Noises:
    """The noise recordings, loaded and set to each loudness once. Doing that per clip took
    most of a training round (57 s for 10,000 windows, 10 Nov 2026)."""

    def __init__(self, folder: Path) -> None:
        from lab.caller.mouth.phone_mic import load_noise

        recordings = {kind: load_noise(folder, kind) for kind in NOISE_KINDS}
        # The lab's own NoiseBed sets the loudness, so these match the pretend caller's noise.
        self.scaled = {(kind, level): NoiseBed(rec, db).next(len(rec))
                       for kind, rec in recordings.items() for level, db in NOISE_LEVELS_DB.items()}

    def stretch(self, kind: str, level: str, start: int, count: int) -> np.ndarray:
        """`count` samples of that noise from `start`, looping round the recording's end."""
        noise = self.scaled[(kind, level)]
        return noise[(start + np.arange(count)) % len(noise)]


def as_float(voice: np.ndarray) -> np.ndarray:
    """Float sound. Real pieces are kept in memory as 16-bit, as saved, which is half the size
    (14 Nov 2026); divided by 32768 they are exactly what reading the file as float gives."""
    voice = np.asarray(voice)
    if voice.dtype == np.int16:
        return voice.astype(np.float32) / np.float32(32768)
    return voice.astype(np.float32, copy=False)


def on_a_rough_line(voice: np.ndarray, rough: Roughness, noises: Noises | None, dither_seed: int = 0) -> tuple[np.ndarray, float]:
    """(the clip on a phone line, how far its times moved): a time t in the clean clip is at
    LEAD_S + t * rough.speed in the result."""
    voice = as_float(voice)
    if rough.speed != 1.0:
        # Straight-line stretching, not the lab's resampler: a random speed like 1.059 makes a
        # ratio with no small common factor, which that resampler handles very slowly.
        count = int(round(len(voice) * rough.speed))
        voice = np.interp(np.linspace(0, len(voice) - 1, count), np.arange(len(voice)), voice).astype(np.float32)
    voice = np.asarray(band_limit(set_speech_level(voice, RATE)), dtype=np.float32) * np.float32(10 ** (rough.gain_db / 20))
    if rough.echo:
        voice = room_echo(voice, seed=dither_seed)
    line = np.concatenate([np.zeros(int(LEAD_S * RATE), np.float32), voice, np.zeros(int(TAIL_S * RATE), np.float32)])
    # A real line is never digital silence; a faint hiss keeps quiet parts from all looking alike.
    line += np.random.default_rng(dither_seed).normal(0, 3e-4, len(line)).astype(np.float32)
    if rough.noise is not None and noises is not None:
        line = line + noises.stretch(rough.noise, rough.noise_level, rough.noise_start, len(line))
    return np.asarray(mu_law_round_trip(line), dtype=np.float32), float(rough.speed)
