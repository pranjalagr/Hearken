"""Changing a sound's sample rate, for example a 24 kHz voice down to the
wire's 8 kHz."""

from __future__ import annotations

from math import gcd

import numpy as np
from scipy.signal import resample_poly

# The wire and every phone line carry 8 kHz sound (Part 2 decision 4).
PHONE_RATE = 8000


def resample(samples: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Polyphase resampling with scipy's built-in anti-aliasing filter.

    Simply dropping samples would fold everything above 4 kHz back into the
    hearing range as hiss and whistles; the filter removes it first.
    """
    if from_rate <= 0 or to_rate <= 0:
        raise ValueError("sample rates must be positive")
    samples = np.asarray(samples, dtype=np.float32)
    if from_rate == to_rate:
        return samples.copy()
    common = gcd(from_rate, to_rate)
    return resample_poly(samples, to_rate // common, from_rate // common).astype(np.float32)
