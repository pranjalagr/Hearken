"""What a phone line does to sound (Part 3, C5).

Two effects, kept apart on purpose:
- the band limit (300-3400 Hz) is linear, so it can be applied to the voice
  and to the noise separately, before they are added;
- the codec (G.711 mu-law) is NOT linear, so it must run once, on the sum.
That order is what "noise enters the microphone before the phone line squashes
the sound" means in code.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

from common.audio.resample import PHONE_RATE

# ITU-T G.712: a telephone channel passes roughly 300 to 3400 Hz.
BAND_LOW_HZ = 300.0
BAND_HIGH_HZ = 3400.0

# G.711 mu-law: mu = 255, 8 bits per sample (1 sign bit + 7 magnitude bits).
_MU = 255.0
_MAGNITUDE_STEPS = 127

_BAND_SOS = butter(4, [BAND_LOW_HZ, BAND_HIGH_HZ], btype="bandpass", fs=PHONE_RATE, output="sos")
# How far sosfiltfilt pads each edge, given explicitly so a short block works
# too (the padding must be shorter than the block).
_EDGE_PAD = 3 * (2 * len(_BAND_SOS) + 1)


def band_limit(samples_8k: np.ndarray) -> np.ndarray:
    """The phone's frequency band. Zero-phase (filtered forwards and back), so
    speech is not smeared in time; fine because it runs on whole blocks."""
    samples = np.asarray(samples_8k, dtype=np.float64)
    if len(samples) < 2:
        return samples.astype(np.float32)
    return sosfiltfilt(_BAND_SOS, samples, padlen=min(_EDGE_PAD, len(samples) - 1)).astype(np.float32)


def mu_law_round_trip(samples: np.ndarray) -> np.ndarray:
    """Encode to 8-bit mu-law and decode back, keeping the codec's rounding.

    This uses the smooth mu-law curve; real G.711 uses 8 straight segments
    that follow the same curve closely, so the sound is the same to the ear.
    Works sample by sample, so cutting a stream into pieces changes nothing.
    """
    x = np.clip(np.asarray(samples, dtype=np.float64), -1.0, 1.0)
    magnitude = np.log1p(_MU * np.abs(x)) / np.log1p(_MU)
    code = np.rint(magnitude * _MAGNITUDE_STEPS) / _MAGNITUDE_STEPS
    decoded = np.expm1(code * np.log1p(_MU)) / _MU
    return (np.sign(x) * decoded).astype(np.float32)
