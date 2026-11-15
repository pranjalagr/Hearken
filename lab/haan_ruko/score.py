"""Marking the haan vs ruko model the way a live call would use it (Part 4, steps 7 and 8).

A clip is not marked by single windows but as a call would play it: the model is asked 0.3 s
after the caller's first sound, then every 0.1 s until they finish, and the first "stop" ends
it. So a listening sound is a wrong stop if the model says stop at ANY of those checks, and
real speech is a missed stop if it NEVER says stop. Stopping during the "haan" of "haan,
lekin..." is early but harmless: the caller was about to speak anyway.
"""

from __future__ import annotations

import zlib
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from lab.haan_ruko.phrases import KEEP, MIXED
from lab.haan_ruko.sound import LEAD_S, Noises, Roughness, log_mel, on_a_rough_line, window
from lab.haan_ruko.speech import Timing

FIRST_CHECK_S = 0.3  # the reflex's budget: the first answer comes this long after the first sound
CHECK_EVERY_S = 0.1
LONGEST_LISTEN_S = 2.0  # after this much speech it can't be a listening sound
AFTER_END_S = 0.2  # checks go on this long after the last sound, until end of turn takes over
UNSURE_AFTER_SWITCH_S = 0.25  # the first moments of "lekin" sound like more "haan": not taught


def check_times(timing: Timing) -> list[float]:
    """When a live robot would ask the model, in the clean clip's seconds."""
    last = min(timing.end + AFTER_END_S, timing.start + LONGEST_LISTEN_S)
    times, t = [], timing.start + FIRST_CHECK_S
    while t <= last + 1e-9:
        times.append(round(t, 3))
        t += CHECK_EVERY_S
    return times or [round(timing.start + FIRST_CHECK_S, 3)]


def answer_at(answer: str, timing: Timing, t: float) -> int | None:
    """The right answer at clean-clip time t: 0 keep, 1 stop, None when no answer is fair."""
    if answer == KEEP:
        return 0
    if answer == MIXED:
        if t < timing.switch:
            return 0
        return None if t < timing.switch + UNSURE_AFTER_SWITCH_S else 1
    if timing.ignore_until is not None and t < timing.ignore_until:
        return None
    return 1


@dataclass(frozen=True)
class Played:
    """One clip played through the model on one kind of line."""

    clip: dict
    timing: Timing
    line: str
    times: list[float]  # seconds after the caller's first sound
    chances: list[float]


@dataclass(frozen=True)
class Heard:
    """One clip on one fixed line, as the model's sound picture. The test lines never change,
    so the picture is the same every time: training marks its check set every round from these
    rather than making the lines again (about 20 s a round, 11 Nov 2026)."""

    clip: dict
    timing: Timing
    line: str
    times: list[float]  # each check, in the clean clip's seconds
    features: np.ndarray
    speed: float


# Windows given to the model at once: big batches, without holding every window of a set at
# once (a set's 50,000 windows are 1.6 GB, and all 40,000 at once broke the ONNX run, 11 Nov 2026).
ASK_AT_ONCE = 8192


def hear(clips: list[tuple[dict, Timing, np.ndarray]], lines: dict[str, Roughness], noises: Noises | None) -> list[Heard]:
    heard = []
    for clip, timing, audio in clips:
        for name, rough in lines.items():
            # crc32, not hash(): Python changes hash() for text on every run, and a test must not.
            line, speed = on_a_rough_line(audio, rough, noises, dither_seed=zlib.crc32(clip["id"].encode()) % 10_000)
            heard.append(Heard(clip, timing, name, check_times(timing), log_mel(line), speed))
    return heard


def ask(heard: list[Heard], chances_of: Callable[[np.ndarray], np.ndarray]) -> list[Played]:
    """The model asked at every check of every heard clip, a batch at a time."""
    played: list[Played] = []
    group: list[Heard] = []
    batch: list[np.ndarray] = []

    def answer_group() -> None:
        out = chances_of(np.stack(batch))
        at = 0
        for h in group:
            played.append(Played(h.clip, h.timing, h.line, [round(t - h.timing.start, 3) for t in h.times],
                                 [float(c) for c in out[at:at + len(h.times)]]))
            at += len(h.times)
        group.clear()
        batch.clear()

    for h in heard:
        batch += [window(h.features, LEAD_S + t * h.speed) for t in h.times]
        group.append(h)
        if len(batch) >= ASK_AT_ONCE:
            answer_group()
    if group:
        answer_group()
    return played


def play(clips: list[tuple[dict, Timing, np.ndarray]], lines: dict[str, Roughness], noises: Noises | None,
         chances_of: Callable[[np.ndarray], np.ndarray]) -> list[Played]:
    """Every clip on every line, asked at every check."""
    return ask(hear(clips, lines, noises), chances_of)


def first_stop(p: Played, line: float) -> float | None:
    return next((t for t, c in zip(p.times, p.chances) if c >= line), None)


@dataclass(frozen=True)
class Marks:
    keep: int
    wrong_stops: int
    speech: int
    missed_stops: int
    stop_times: list[float]  # seconds from the caller's first sound (from the switch, for mixed)

    @property
    def wrong_share(self) -> float:
        return self.wrong_stops / max(1, self.keep)

    @property
    def missed_share(self) -> float:
        return self.missed_stops / max(1, self.speech)


def mark(played: list[Played], line: float) -> Marks:
    keep = wrong = speech = missed = 0
    times: list[float] = []
    for p in played:
        stop = first_stop(p, line)
        if p.clip["answer"] == KEEP:
            keep += 1
            wrong += stop is not None
            continue
        speech += 1
        if stop is None:
            missed += 1
        elif p.clip["answer"] == MIXED:
            times.append(max(0.0, stop - (p.timing.switch - p.timing.start)))
        else:
            times.append(stop)
    return Marks(keep, wrong, speech, missed, times)


def zero_miss_line(played: list[Played]) -> float:
    """The highest line that still stops on every piece of real speech: no caller is talked
    over, and as few listening sounds as possible cause a stop. Missing a real stop is the
    worse mistake (the user, 10 Nov 2026: when unsure, lean to stop)."""
    speech = [max(p.chances) for p in played if p.clip["answer"] != KEEP]
    return min(speech) if speech else 0.5


def separation(played: list[Played]) -> float:
    """How well the highest chance in a clip tells speech from listening (1 perfect, 0.5 guessing)."""
    speech = np.array([max(p.chances) for p in played if p.clip["answer"] != KEEP])
    keep = np.array([max(p.chances) for p in played if p.clip["answer"] == KEEP])
    if not len(speech) or not len(keep):
        return 0.5
    return float(np.mean((speech[:, None] > keep[None, :]) + 0.5 * (speech[:, None] == keep[None, :])))
