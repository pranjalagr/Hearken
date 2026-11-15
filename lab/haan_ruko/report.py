"""The haan vs ruko model's report, in plain words (Part 4, steps 7 and 8)."""

from __future__ import annotations

import statistics
import time
from collections import defaultdict

import numpy as np

from lab.haan_ruko.phrases import KEEP
from lab.haan_ruko.score import Marks, Played, first_stop, mark, separation
from lab.haan_ruko.sound import RATE, log_mel, window


def _share(n: int, total: int) -> str:
    return f"{n} of {total} ({100 * n / max(1, total):.0f}%)"


def _times(marks: Marks) -> str:
    if not marks.stop_times:
        return "-"
    times = sorted(marks.stop_times)
    return f"{statistics.median(times):.2f} s / {times[int(0.9 * (len(times) - 1))]:.2f} s"


def table(played: list[Played], line: float, title: str) -> list[str]:
    rows = [f"### {title}", "", "| Line | Listening sounds wrongly stopped | Real speech missed | Time to stop (usual / 9 in 10) |",
            "|---|---|---|---|"]
    by_line: dict[str, list[Played]] = defaultdict(list)
    for p in played:
        by_line[p.line].append(p)
    for name, mine in [*by_line.items(), ("all", played)]:
        got = mark(mine, line)
        rows.append(f"| {name} | {_share(got.wrong_stops, got.keep)} | {_share(got.missed_stops, got.speech)} | {_times(got)} |")
    return rows + [""]


def by_group(played: list[Played], line: float) -> list[str]:
    rows = ["| Group | Clips | Stopped |", "|---|---|---|"]
    groups: dict[str, list[Played]] = defaultdict(list)
    for p in played:
        groups[p.clip["group"]].append(p)
    for group, mine in sorted(groups.items()):
        stopped = sum(first_stop(p, line) is not None for p in mine)
        rows.append(f"| {group} ({'should keep' if group == KEEP else 'should stop'}) | {len(mine)} | {_share(stopped, len(mine))} |")
    return rows + [""]


def mistakes(played: list[Played], line: float, limit: int = 40) -> list[str]:
    rows = []
    for p in played:
        stop = first_stop(p, line)
        if p.clip["answer"] == KEEP and stop is not None:
            rows.append(f"- wrong stop at {stop:.1f} s: \"{p.clip['text']}\" ({p.clip['voice']}, {p.line} line)")
        elif p.clip["answer"] != KEEP and stop is None:
            rows.append(f"- missed: \"{p.clip['text'][:60]}\" ({p.clip['voice']}, {p.line} line)")
    return rows[:limit] + ([f"- ...and {len(rows) - limit} more"] if len(rows) > limit else [])


def check_speed(chances_of, repeats: int = 200) -> tuple[float, float]:
    """One live check: the sound picture of the last 2 s, then the model, in milliseconds."""
    audio = np.random.default_rng(0).normal(0, 0.05, int(2.6 * RATE)).astype(np.float32)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        chances_of(window(log_mel(audio), 2.5)[None])
        times.append((time.perf_counter() - start) * 1000)
    times.sort()
    return statistics.median(times), times[int(0.95 * (repeats - 1))]


def report(check: list[Played], test: list[Played], line: float, history: list[dict], speed: tuple[float, float],
           counts: dict[str, int], when: str, real_test: list[Played] | None = None) -> str:
    keep_n = sum(p.clip["answer"] == KEEP for p in test)
    lines = [f"# Haan vs ruko model ({when})", "",
             "Machine voices only: real voices are the real test (the user's recordings, then friends' in week 11).", "",
             f"Clips: {counts['train']} for training (4 voices), {counts['check']} for checking (the left-out voice, "
             f"inaaya), {counts['test']} held-back phrases for testing (never trained on, all voices).", "",
             f"**The line** (chosen on the left-out voice, never on the test): stop when the model's chance is at least "
             f"{line:.3f}. It is the highest line that still stops on every piece of real speech in the left-out voice.", "",
             f"Tells real speech from listening sounds: {separation(check):.3f} on the left-out voice, "
             f"{separation(test):.3f} on held-back phrases (1 is perfect, 0.5 is guessing).", "",
             "**Against \"stop on any sound\"** (Agent A's default): it never misses real speech, but it stops on every "
             f"listening sound: {keep_n} of {keep_n} on the held-back test.", ""]
    if real_test:
        lines += [f"Real people: {counts.get('real_train', 0)} IndicVoices pieces in training; the test below is "
                  f"{counts.get('real_test', 0)} pieces by speakers never heard in training (Hindi phone conversations, CC BY 4.0).", ""]
        lines += table(real_test, line, "Real people never heard in training (IndicVoices)")
        lines += [f"Separation on real people: {separation(real_test):.3f}.", ""]
    lines += table(test, line, "Held-back phrases (never trained on)")
    lines += by_group(test, line)
    lines += table(check, line, "Left-out voice (phrases it trained on, a voice it never heard)")
    lines += ["Time to stop is from the caller's first sound (from the start of \"lekin\" for mixed phrases). "
              "The first check is at 0.3 s, so 0.3 s is the fastest possible.", "",
              f"**Speed of one check** (sound picture + model, one CPU thread): usual {speed[0]:.1f} ms, "
              f"19 in 20 under {speed[1]:.1f} ms. Target: under 20 ms.", "",
              "## Mistakes on the held-back phrases", ""] + mistakes(test, line)
    lines += ["", "## Training rounds", "", "| Round | Loss | Left-out voice: wrong stops at its no-miss line | Separation |",
              "|---|---|---|---|"]
    lines += [f"| {h['round']} | {h['loss']} | {h['wrong_stops']} of {h['keep']} | {h['separation']} |" for h in history]
    return "\n".join(lines) + "\n"
