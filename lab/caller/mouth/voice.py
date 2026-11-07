"""The callers' voice (Part 3, C4): text to phone-ready pieces, on Smallest.ai.

Every clip is saved by its voice and text and never made twice (C17), so a line
said again in another call costs nothing and arrives at once. The saved file is
what the company sent; levelling and the phone's band happen each time it's
loaded, so changing those needs no new requests.

The codec is not applied here: it isn't linear, so it must run once on voice and
noise together (common/audio/phone_line.py), which is the phone mic's job.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import soundfile

from common.audio.noise import set_speech_level
from common.audio.pcm import float_to_pcm16, split_pieces
from common.audio.phone_line import band_limit
from common.audio.resample import PHONE_RATE
from common.wire.protocol import PIECE_SAMPLES
from lab.week1.flash_lite_check import with_retries

SMALLEST_TTS = "https://api.smallest.ai/waves/v1/tts"
# A clip of a sentence or two took about 1 s when measured (4 Nov 2026): past this
# it's stuck, and a caller can't keep a robot waiting much longer.
REQUEST_TIMEOUT_S = 15.0
# Short waits: this runs during a live call. A daily limit fails at once and the
# call becomes a lab problem to rerun (decision 32), never a robot failure.
RETRY_WAITS_S = (1.0, 3.0)


class Voice(Protocol):
    async def pieces(self, text: str) -> list[bytes]:
        """The text as wire pieces, ready for the phone mic."""


def clip_to_pieces(wav: bytes) -> list[bytes]:
    """A clip as phone-ready pieces, at the fixed speech level, so that "loud
    noise" means the same for every voice (C5)."""
    samples, rate = soundfile.read(io.BytesIO(wav), dtype="float32", always_2d=True)
    if rate != PHONE_RATE or not len(samples):
        raise RuntimeError(f"expected 8 kHz audio, got {rate} Hz and {len(samples)} samples")
    mono = samples.mean(axis=1)
    level = np.asarray(band_limit(set_speech_level(mono, rate)), dtype=np.float32)
    return [float_to_pcm16(piece) for piece in split_pieces(level, PIECE_SAMPLES)]


class SavedClips:
    """Clips on disk by voice and text: wav files stay out of git (C18)."""

    def __init__(self, folder: Path) -> None:
        self._folder = folder

    def path(self, voice: str, text: str) -> Path:
        # A hash, not the text: Hindi letters and punctuation make poor file names.
        return self._folder / voice / f"{hashlib.sha256(text.encode('utf-8')).hexdigest()[:24]}.wav"


class SmallestVoice:
    def __init__(self, key: str, voice_id: str, model: str, clips: SavedClips,
                 opener: Callable[..., Any] = urllib.request.urlopen,
                 on_cost: Callable[[int], None] = lambda characters: None) -> None:
        self._key = key
        self.voice_id = voice_id
        self.model = model
        self._clips = clips
        self._opener = opener
        self._on_cost = on_cost

    async def pieces(self, text: str) -> list[bytes]:
        # Off the loop, which is beating 20 ms pieces for every live call.
        path = self._clips.path(self.voice_id, text)
        wav = await asyncio.to_thread(_read_if_saved, path)
        if wav is None:
            wav = await with_retries(lambda: asyncio.to_thread(self._make, text), RETRY_WAITS_S)
            await asyncio.to_thread(_save, path, wav)
            self._on_cost(len(text))
        return await asyncio.to_thread(clip_to_pieces, wav)

    def _make(self, text: str) -> bytes:
        body = {"text": text, "voice_id": self.voice_id, "model": self.model, "sample_rate": PHONE_RATE,
                "speed": 1.0, "language": "hi", "output_format": "wav"}
        request = urllib.request.Request(SMALLEST_TTS, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                         method="POST", headers={"Authorization": f"Bearer {self._key}",
                                                                 "Content-Type": "application/json", "Accept": "audio/wav"})
        with self._opener(request, timeout=REQUEST_TIMEOUT_S) as response:
            return response.read()


def _read_if_saved(path: Path) -> bytes | None:
    return path.read_bytes() if path.exists() else None


def _save(path: Path, wav: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_bytes(wav)
    # Whole or not at all: another call may ask for the same line at any moment.
    temp.replace(path)
