"""The ears test's lines and sound clips (Part 2 decision 24, Part 3 C5).

A person's recording goes in as it was made, and comes out as two phone
versions, so every service hears exactly the same bytes:
- phone: what the line itself does to a voice;
- noise: the same, with background noise added first, because noise enters the
  microphone before the phone line squashes the sound (C5).

Real voices stay under lab/sounds/, which git ignores (C16, plan risk 9).
"""

from __future__ import annotations

import json
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from common.audio.noise import NOISE_LEVELS_DB, NoiseBed, set_speech_level
from common.audio.phone_line import band_limit, mu_law_round_trip
from common.audio.resample import PHONE_RATE, resample
from common.audio.sound_files import read_sound, write_sound

ROOT = Path(__file__).resolve().parents[2]
LINES = Path(__file__).with_name("ears_lines.yaml")
SOUNDS = ROOT / "lab" / "sounds" / "ears"
NOISE_DIR = ROOT / "lab" / "sounds" / "noise"
KINDS = ("number", "name", "amount", "haan", "nahi", "ruko", "word")
VERSIONS = ("phone", "noise")

# The rate noise and voice are mixed at: above the phone's, so reducing to 8 kHz
# still happens after the mix, and fixed, so every recording is treated alike.
WORK_RATE = 16000
# Silence around the speech. The tail matters most: a service often waits for a
# pause before it gives its final words.
LEAD_S = 0.3
TAIL_S = 1.5
# C5's middle loudness: hard, but the words are still there for a person.
NOISE_DB = NOISE_LEVELS_DB["medium"]
# Used when no noise files are provided: several people talking at once, built
# from the other recordings. It stands in for C5's "crowd" only.
BABBLE_VOICES = 6
BABBLE_S = 30.0
# A sound counts as speech within this much of the clip's loudest moment.
SPEECH_RANGE_DB = 30.0


@dataclass(frozen=True)
class Detail:
    kind: str
    forms: tuple[str, ...]


@dataclass(frozen=True)
class EarsLine:
    id: str
    say: str
    expect: tuple[Detail, ...]


def load_lines(path: Path = LINES) -> list[EarsLine]:
    """Strict: a typo must fail here, not quietly score a service wrong."""
    lines = []
    for item in yaml.safe_load(path.read_text(encoding="utf-8")):
        if set(item) != {"id", "say", "expect"} or not str(item["say"]).strip() or not item["expect"]:
            raise ValueError(f"line {item.get('id')}: needs id, say and at least one expected detail")
        details = []
        for detail in item["expect"]:
            if detail.get("kind") not in KINDS or not detail.get("any"):
                raise ValueError(f"line {item['id']}: a detail needs a known kind and some forms")
            details.append(Detail(detail["kind"], tuple(str(form) for form in detail["any"])))
        lines.append(EarsLine(str(item["id"]), str(item["say"]), tuple(details)))
    ids = [line.id for line in lines]
    if len(ids) != len(set(ids)):
        raise ValueError("line ids must be unique")
    return lines


def raw_path(speaker: str, line_id: str, sounds: Path = SOUNDS) -> Path:
    return sounds / "raw" / speaker / f"{line_id}.wav"


def clip_path(version: str, speaker: str, line_id: str, sounds: Path = SOUNDS) -> Path:
    return sounds / version / speaker / f"{line_id}.wav"


def speech_span(samples: np.ndarray, rate: int, frame_ms: int = 20) -> tuple[float, float]:
    """When the speech starts and ends, in seconds, for the speed numbers.
    Taken from the clean phone version, since noise has no pauses to find."""
    frame = max(1, rate * frame_ms // 1000)
    usable = len(samples) - len(samples) % frame
    if usable == 0:
        return 0.0, 0.0
    power = np.mean(np.asarray(samples[:usable], dtype=np.float64).reshape(-1, frame) ** 2, axis=1)
    if power.max() == 0.0:
        return 0.0, 0.0
    active = np.flatnonzero(power >= power.max() * 10.0 ** (-SPEECH_RANGE_DB / 10.0))
    return round(active[0] * frame / rate, 3), round((active[-1] + 1) * frame / rate, 3)


def phone_version(samples: np.ndarray, noise: np.ndarray | None = None) -> np.ndarray:
    """A clip at WORK_RATE as the phone line delivers it. The codec runs once,
    on the sum, because it isn't linear (common/audio/phone_line.py)."""
    mixed = samples if noise is None else samples + noise
    return mu_law_round_trip(band_limit(resample(mixed, WORK_RATE, PHONE_RATE)))


def make_babble(voices: list[np.ndarray], seed: int = 1) -> np.ndarray:
    """Several people talking at once: each voice repeated, from its own random
    start, so no single sentence can be picked out. The seed keeps it the same
    from run to run, so every service is tested on the same noise."""
    if len(voices) < BABBLE_VOICES:
        raise ValueError(f"babble needs {BABBLE_VOICES} recordings; got {len(voices)}. Record more, or add noise files.")
    rng = np.random.default_rng(seed)
    length = int(BABBLE_S * WORK_RATE)
    mix = np.zeros(length, dtype=np.float64)
    # A few voices, not all of them: dozens would average into a plain hiss,
    # and a crowd is meant to sound like people.
    for pick in rng.choice(len(voices), BABBLE_VOICES, replace=False):
        looped = np.resize(voices[pick], length)
        mix += np.roll(looped, int(rng.integers(length)))
    return mix.astype(np.float32)


def load_noises(noise_dir: Path, recordings: list[np.ndarray]) -> tuple[list[np.ndarray], str]:
    """(noise recordings at WORK_RATE, what they are). Real files win."""
    files = sorted(noise_dir.glob("*.wav")) + sorted(noise_dir.glob("*.flac")) if noise_dir.exists() else []
    if files:
        noises = []
        for path in files:
            samples, rate = read_sound(path)
            noises.append(resample(samples, rate, WORK_RATE))
        return noises, f"{len(files)} noise files"
    return [make_babble(recordings)], "babble made from the recordings (no noise files given)"


def prepare(sounds: Path = SOUNDS, noise_dir: Path = NOISE_DIR) -> dict[str, Any]:
    """Make both phone versions of every recording, and clips.json. Safe to
    run again: the same recordings give the same clips."""
    raw_paths = sorted((sounds / "raw").glob("*/*.wav"))
    if not raw_paths:
        raise ValueError(f"no recordings under {sounds / 'raw'}")
    voices: dict[Path, np.ndarray] = {}
    for path in raw_paths:
        samples, rate = read_sound(path)
        voices[path] = set_speech_level(resample(samples, rate, WORK_RATE), WORK_RATE)
    noises, noise_kind = load_noises(noise_dir, list(voices.values()))
    lead, tail = np.zeros(int(LEAD_S * WORK_RATE), np.float32), np.zeros(int(TAIL_S * WORK_RATE), np.float32)
    clips = {}
    for path, voice in voices.items():
        speaker, line_id = path.parent.name, path.stem
        padded = np.concatenate([lead, voice, tail])
        # The line picks its noise and start, so re-runs make the very same clip.
        pick = zlib.crc32(f"{speaker}/{line_id}".encode())
        bed = NoiseBed(noises[pick % len(noises)], NOISE_DB, start_sample=pick)
        clean = phone_version(padded)
        noisy = phone_version(padded, bed.next(len(padded)))
        for version, samples in (("phone", clean), ("noise", noisy)):
            target = clip_path(version, speaker, line_id, sounds)
            target.parent.mkdir(parents=True, exist_ok=True)
            write_sound(target, samples, PHONE_RATE)
        start, end = speech_span(clean, PHONE_RATE)
        clips[f"{speaker}/{line_id}"] = {"speech_start_s": start, "speech_end_s": end,
                                         "seconds": round(len(clean) / PHONE_RATE, 3)}
    manifest = {"noise": noise_kind, "noise_db": NOISE_DB, "clips": clips}
    (sounds / "clips.json").write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    return manifest
