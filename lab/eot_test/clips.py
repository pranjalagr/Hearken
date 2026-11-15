"""The end-of-turn test's clips (Part 2, decision 25): machine voices with real-length
pauses and background sound, each sentence as a finished and an unfinished clip.

The unfinished clip is the same recording cut at its pause, never the first half said
alone: a voice saying only "chaar paanch" says it as a whole sentence, with the tone
falling at the end, and that clip would sound finished under an "unfinished" label.

A person approves every unfinished clip by ear (review_server.py) before it counts.
These pick a model for now; machine voices flatter these models (97% on machine voices
against 60% on real speech for one of them), so real voices confirm it in week 11.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile
import yaml

from common.audio.noise import NOISE_LEVELS_DB, NoiseBed, set_speech_level
from common.audio.phone_line import band_limit, mu_law_round_trip
from lab.caller.mouth.phone_mic import load_noise

ROOT = Path(__file__).resolve().parents[2]
SOUNDS = ROOT / "lab" / "sounds" / "eot"
NOISE_FOLDER = ROOT / "lab" / "sounds" / "noise"
LINES_FILE = Path(__file__).with_name("lines.yaml")

# Two per sentence, so a model can't do well by knowing one voice.
VOICES = ("aviraj", "rhea", "sanjay", "nirupma", "inaaya")
# (kind, loudness): an even spread, one clean in ten. Loudness as in the lab's calls (C5).
MIXES = ((None, None), ("traffic", "quiet"), ("kitchen", "quiet"), ("crowd", "quiet"), ("traffic", "medium"),
         ("kitchen", "medium"), ("crowd", "medium"), ("traffic", "loud"), ("kitchen", "loud"), ("crowd", "loud"))
# How long the caller stays quiet in the finished clip: a voice's own pause is 0.3-0.7 s,
# shorter than a person looking for a number.
PAUSES_S = (0.5, 0.7, 0.9, 1.1, 1.3, 0.6, 0.8, 1.0)
LEAD_S = 0.3
# Longer than any moment the test asks at, so it can trim each clip to the same quiet.
TRAIL_S = 0.6
# Quiet after the last loud frame is kept this long, so a soft last sound isn't clipped.
GUARD_S = 0.04
FRAME_S = 0.02
QUIET_RATIO = 0.08  # of the voice's loud level
MIN_PAUSE_S = 0.15
MIN_SIDE_S = 0.3  # speech needed on each side of the cut


@dataclass(frozen=True)
class ClipPlan:
    id: str
    line: int
    voice: str
    text: str
    noise: str | None
    level: str | None
    pause_s: float
    noise_start: int


@dataclass(frozen=True)
class Clip:
    """A made clip's facts. The ends are where speech stops in each file, so the test
    can give every clip the same amount of quiet at the end."""

    plan: ClipPlan
    rate: int
    unfinished_end_s: float
    finished_end_s: float


def load_lines(path: Path = LINES_FILE) -> list[str]:
    lines = yaml.safe_load(path.read_text(encoding="utf-8"))
    bad = [line for line in lines if line.count("...") != 1 or "," in line]
    if bad:
        raise ValueError(f"each line needs one '...' and no commas: {bad}")
    return lines


def plan_clips(lines: list[str]) -> list[ClipPlan]:
    """Each line in two voices. Fixed, not random, so a rerun makes the same clips."""
    plans: list[ClipPlan] = []
    for n, text in enumerate(lines):
        for voice in (VOICES[n % len(VOICES)], VOICES[(n + 2) % len(VOICES)]):
            k = len(plans)
            noise, level = MIXES[k % len(MIXES)]
            plans.append(ClipPlan(f"L{n + 1:02d}_{voice}", n + 1, voice, text, noise, level,
                                  PAUSES_S[k % len(PAUSES_S)], 24_000 * (k + 1)))
    return plans


def find_pause(voice: np.ndarray, rate: int) -> tuple[int, int, int, int] | None:
    """(first sound, pause start, pause end, last sound) in samples, for the longest quiet
    stretch inside the speech, or None if there's no usable pause."""
    frame = int(rate * FRAME_S)
    usable = len(voice) - len(voice) % frame
    if usable == 0:
        return None
    rms = np.sqrt(np.mean(voice[:usable].reshape(-1, frame) ** 2, axis=1))
    speaking = rms > QUIET_RATIO * np.percentile(rms, 95)
    loud = np.flatnonzero(speaking)
    if len(loud) == 0:
        return None
    first, last = int(loud[0]), int(loud[-1]) + 1
    best: tuple[int, int] | None = None
    start: int | None = None
    for n in range(first, last):
        if not speaking[n] and start is None:
            start = n
        elif speaking[n] and start is not None:
            long_enough = (n - start) * FRAME_S >= MIN_PAUSE_S
            room = min(start - first, last - n) * FRAME_S >= MIN_SIDE_S
            if long_enough and room and (best is None or n - start > best[1] - best[0]):
                best = (start, n)
            start = None
    if best is None:
        return None
    return first * frame, best[0] * frame, best[1] * frame, last * frame


def build_pair(voice: np.ndarray, rate: int, pause_s: float) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    """(unfinished, finished, where speech ends in each, in seconds), before any noise."""
    found = find_pause(voice, rate)
    if found is None:
        return None
    first, pause_start, pause_end, last = found
    guard = int(rate * GUARD_S)
    lead = np.zeros(int(rate * LEAD_S), dtype=np.float32)
    trail = np.zeros(int(rate * TRAIL_S), dtype=np.float32)
    hold = np.zeros(int(rate * pause_s), dtype=np.float32)
    before = voice[first:pause_start + guard]
    after = voice[pause_end:min(last + guard, len(voice))]
    unfinished_speech = np.concatenate([lead, before])
    finished_speech = np.concatenate([lead, before, hold, after])
    return (np.concatenate([unfinished_speech, trail]), np.concatenate([finished_speech, trail]),
            len(unfinished_speech) / rate, len(finished_speech) / rate)


def on_the_phone(voice: np.ndarray, plan: ClipPlan, noise_folder: Path) -> np.ndarray:
    """Noise first and over the whole clip, pauses included, then the codec on the sum:
    the order the lab's phone mic uses (C5)."""
    if plan.noise is None:
        return mu_law_round_trip(voice)
    bed = NoiseBed(load_noise(noise_folder, plan.noise), NOISE_LEVELS_DB[plan.level], start_sample=plan.noise_start)
    return mu_law_round_trip(voice + bed.next(len(voice)))


def clip_path(sounds: Path, clip_id: str, kind: str) -> Path:
    return sounds / "clips" / f"{clip_id}_{kind}.wav"


async def make_clips(plans: list[ClipPlan], raw: Callable[[ClipPlan], Awaitable[tuple[np.ndarray, int]]],
                     sounds: Path = SOUNDS, noise_folder: Path = NOISE_FOLDER,
                     log: Callable[[str], None] = print, give_up_after: int = 5) -> list[Clip]:
    """Every plan into two files and one manifest. `raw` gives a plan's recording. A failed
    or pause-less clip is reported and left out, and a run of failures stops the spending."""
    (sounds / "clips").mkdir(parents=True, exist_ok=True)
    made: list[Clip] = []
    in_a_row = 0
    for plan in plans:
        try:
            samples, rate = await raw(plan)
            in_a_row = 0
        except Exception as error:  # noqa: BLE001 - reported per clip; a rerun fills the gaps
            in_a_row += 1
            log(f"{plan.id}: no recording: {type(error).__name__}: {str(error)[:120]}")
            if in_a_row >= give_up_after:
                log(f"{in_a_row} failures in a row: stopping")
                break
            continue
        voice = np.asarray(band_limit(set_speech_level(samples, rate)), dtype=np.float32)
        pair = build_pair(voice, rate, plan.pause_s)
        if pair is None:
            log(f"{plan.id}: no usable pause: left out")
            continue
        unfinished, finished, unfinished_end, finished_end = pair
        for kind, audio in (("unfinished", unfinished), ("finished", finished)):
            soundfile.write(clip_path(sounds, plan.id, kind), on_the_phone(audio, plan, noise_folder), rate, subtype="PCM_16")
        made.append(Clip(plan, rate, round(unfinished_end, 3), round(finished_end, 3)))
        if len(made) % 20 == 0:
            log(f"{len(made)} of {len(plans)} made")
    (sounds / "clips.json").write_text(json.dumps([asdict(clip) for clip in made], ensure_ascii=False, indent=1), encoding="utf-8")
    return made


def load_manifest(sounds: Path = SOUNDS) -> list[dict]:
    return json.loads((sounds / "clips.json").read_text(encoding="utf-8"))
