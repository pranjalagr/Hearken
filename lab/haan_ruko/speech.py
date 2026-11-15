"""Where the speech is in each clip, and when its right answer is known (Part 4, step 3).

The voice detector (Silero, the same one the end-of-turn test uses) finds the stretches of
speech. From them come three times per clip, in seconds from the clip's start:

- start: the caller's first sound, where a live robot's detector would wake the model
- switch: in a mixed clip ("haan, lekin..."), where the second part begins; keep before it,
  stop once enough of it is heard
- ignore_until: in a stop clip that opens with a listening word ("Haan bhaiya, mera order..."),
  the opening sounds exactly like a plain "haan", so no answer is taught there at all
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from common.audio.resample import resample
from lab.haan_ruko.phrases import KEEP, MIXED

Runs = list[tuple[float, float]]
MIN_SPEECH_S = 0.08  # shorter is a click or a breath, not a word
MIN_GAP_S = 0.06  # the shortest pause that can separate "haan" from "lekin"
# A listening sound longer than this is something the voice did wrong (a long silence, or a
# phrase read out letter by letter): it would teach "long speech can mean keep".
LONGEST_LISTENING_S = 2.0
NEAR_THE_COMMA = 0.35  # share of the speech either side of where the comma falls in the text
OPENING_MARGIN_S = 0.25  # after the opening listening word, before an answer is taught again
NO_GAP_OPENING_S = 0.6  # when no pause follows the opening word, how long to stay silent


@dataclass(frozen=True)
class Timing:
    id: str
    start: float | None
    end: float | None
    switch: float | None = None
    ignore_until: float | None = None
    why_left_out: str = ""


def speech_runs(runs: Runs) -> Runs:
    return [(a, b) for a, b in runs if b - a >= MIN_SPEECH_S]


def comma_share(text: str) -> float:
    """Where the first comma falls, as a share of the letters: roughly where the pause is."""
    before, after = text.split(",", 1)
    return len(before) / max(1, len(before) + len(after))


def opening_listening_word(text: str, listening: list[str]) -> str | None:
    """The listening phrase a stop clip opens with, if any ("Haan ji, mera..." -> "haan ji")."""
    lowered = text.lower()
    for phrase in sorted(listening, key=len, reverse=True):
        if re.match(rf"{re.escape(phrase.lower())}(?=$|[\s,.?!])", lowered):
            return phrase
    return None


def time_clip(clip_id: str, answer: str, text: str, runs: Runs, listening: list[str]) -> Timing:
    kept = speech_runs(runs)
    if not kept:
        return Timing(clip_id, None, None, why_left_out="no speech found")
    start, end = kept[0][0], kept[-1][1]
    gaps = [(a[1], b[0]) for a, b in zip(kept, kept[1:]) if b[0] - a[1] >= MIN_GAP_S]
    if answer == KEEP:
        if end - start > LONGEST_LISTENING_S:
            return Timing(clip_id, start, end, why_left_out=f"{end - start:.1f} s is too long for a listening sound")
        return Timing(clip_id, start, end)
    if answer == MIXED:
        expected = comma_share(text)
        near = [g for g in gaps if abs((g[0] - start) / (end - start) - expected) <= NEAR_THE_COMMA]
        if not near:
            return Timing(clip_id, start, end, why_left_out="no pause near the comma")
        pause = min(near, key=lambda g: abs((g[0] - start) / (end - start) - expected))
        return Timing(clip_id, start, end, switch=pause[1])
    if opening_listening_word(text, listening):
        early = [g for g in gaps if g[0] - start <= 1.2]
        until = early[0][1] + OPENING_MARGIN_S if early else start + NO_GAP_OPENING_S
        return Timing(clip_id, start, end, ignore_until=until)
    return Timing(clip_id, start, end)


def time_all(clips: list[dict], sounds: Path, detect: Callable[[np.ndarray], Runs], listening: list[str],
             log: Callable[[str], None] = print) -> list[Timing]:
    import soundfile

    from lab.eot_test.human import DETECTOR_RATE

    timings = []
    for n, clip in enumerate(clips, 1):
        samples, rate = soundfile.read(sounds / clip["file"], dtype="float32", always_2d=True)
        runs = detect(resample(samples.mean(axis=1), rate, DETECTOR_RATE))
        timings.append(time_clip(clip["id"], clip["answer"], clip["text"], runs, listening))
        if n % 200 == 0:
            log(f"{n} of {len(clips)} timed")
    (sounds / "timings.json").write_text(json.dumps([asdict(t) for t in timings], indent=1), encoding="utf-8")
    return timings


def load_timings(sounds: Path) -> dict[str, Timing]:
    return {t["id"]: Timing(**t) for t in json.loads((sounds / "timings.json").read_text(encoding="utf-8"))}
