"""Runs the models on the approved clips and marks them (Part 2, decision 25).

Two mistakes are counted. A cut-off is an unfinished clip called finished: the robot talks
over the caller, which is the worse one. A wasted wait is a finished clip called unfinished:
the robot stays quiet too long.

Each model gets its own cut-off point on the 0-1 chance, because their scales differ. It's
picked on half the sentences and scored on the other half, then the halves swap, so a
model is never marked on a sentence its point was chosen on, and every clip still counts.
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile

from common.audio.resample import resample
from lab.eot_test.clips import clip_path
from lab.eot_test.models import MODEL_RATE, Model

# The most cut-offs a model may make; among points within this, the fewest wasted waits.
MAX_CUTOFFS = 0.05
# How long the caller has been quiet when a model is asked. 0.2 s is about when a voice
# detector reports silence; 0.5 s shows whether waiting a little longer helps.
WAITS_S = (0.2, 0.5)
UNFINISHED, FINISHED = "unfinished", "finished"


@dataclass(frozen=True)
class Row:
    model: str
    wait_s: float
    clip: str
    line: int
    kind: str
    voice: str
    noise: str  # kind and loudness, or "none"
    chance: float
    ms: float


def approved(manifest: list[dict], verdicts: dict[str, str]) -> list[dict]:
    """Only pairs whose unfinished clip a person approved. The pair goes together, so
    both kinds come from the same sentences, voices and noise."""
    return [clip for clip in manifest if verdicts.get(clip["plan"]["id"]) == "ok"]


def until_quiet_for(audio: np.ndarray, rate: int, speech_end_s: float, wait_s: float) -> np.ndarray:
    """The clip up to `wait_s` after speech stops, so every clip ends in the same quiet.
    Otherwise a small difference in the quiet at the end could tell a model the answer."""
    return audio[:int(round((speech_end_s + wait_s) * rate))]


def run_models(clips: list[dict], models: dict[str, Model], sounds: Path, waits: tuple[float, ...] = WAITS_S,
               log: Callable[[str], None] = print) -> list[Row]:
    rows: list[Row] = []
    for name, model in models.items():
        for clip in clips:
            plan = clip["plan"]
            noise = f"{plan['noise']} {plan['level']}" if plan["noise"] else "none"
            for kind, end_s in ((UNFINISHED, clip["unfinished_end_s"]), (FINISHED, clip["finished_end_s"])):
                audio, rate = soundfile.read(clip_path(sounds, plan["id"], kind), dtype="float32")
                for wait in waits:
                    asked = resample(until_quiet_for(audio, rate, end_s, wait), rate, MODEL_RATE)
                    start = time.perf_counter()
                    chance = model(asked)
                    rows.append(Row(name, wait, plan["id"], plan["line"], kind, plan["voice"], noise, chance,
                                    (time.perf_counter() - start) * 1000))
        log(f"{name}: done")
    return rows


def choose_point(unfinished: list[float], limit: float = MAX_CUTOFFS) -> float:
    """The lowest point (so the fewest wasted waits) that keeps cut-offs within the limit."""
    ordered = sorted(unfinished, reverse=True)
    allowed = int(limit * len(ordered))
    if allowed >= len(ordered):
        return 0.0
    return float(np.nextafter(ordered[allowed], 2.0))


@dataclass(frozen=True)
class Mark:
    cutoffs: int
    unfinished: int
    wasted: int
    finished: int

    @property
    def cutoff_share(self) -> float:
        return self.cutoffs / max(1, self.unfinished)

    @property
    def wasted_share(self) -> float:
        return self.wasted / max(1, self.finished)


def mark(rows: list[Row], point: float) -> Mark:
    unfinished = [r.chance for r in rows if r.kind == UNFINISHED]
    finished = [r.chance for r in rows if r.kind == FINISHED]
    return Mark(sum(c >= point for c in unfinished), len(unfinished), sum(c < point for c in finished), len(finished))


def cross_marked(rows: list[Row], limit: float = MAX_CUTOFFS) -> tuple[Mark, list[float], dict[str, Row | float]]:
    """Point chosen on one half of the sentences, scored on the other, both ways round.
    Returns the summed mark, the two points, and each clip-and-kind's point for breakdowns."""
    total = Mark(0, 0, 0, 0)
    points: list[float] = []
    point_of: dict[str, Row | float] = {}
    for half in (0, 1):
        choosing = [r for r in rows if r.line % 2 == half]
        scoring = [r for r in rows if r.line % 2 != half]
        point = choose_point([r.chance for r in choosing if r.kind == UNFINISHED], limit)
        points.append(point)
        got = mark(scoring, point)
        total = Mark(total.cutoffs + got.cutoffs, total.unfinished + got.unfinished, total.wasted + got.wasted, total.finished + got.finished)
        point_of.update({f"{r.clip}/{r.kind}": point for r in scoring})
    return total, points, point_of


def separation(rows: list[Row]) -> float:
    """The chance a random finished clip scores above a random unfinished one (0.5 = guessing)."""
    finished = np.array([r.chance for r in rows if r.kind == FINISHED])
    unfinished = np.array([r.chance for r in rows if r.kind == UNFINISHED])
    if not len(finished) or not len(unfinished):
        return 0.5
    return float(np.mean((finished[:, None] > unfinished[None, :]) + 0.5 * (finished[:, None] == unfinished[None, :])))


def _share(count: int, total: int) -> str:
    return f"{count} of {total} ({100 * count / max(1, total):.0f}%)"


MACHINE_NOTE = ("approved machine-voice sentences-in-a-voice, each finished and unfinished, at phone quality with background sound. "
                "Machine voices flatter these models: this picks a model for now, and real voices confirm it in week 11.")
HUMAN_NOTE = "takes by one real person, each finished and unfinished, at phone quality in the room's own sound. One voice is still only one voice."


def report(rows: list[Row], when: str, limit: float = MAX_CUTOFFS, note: str = MACHINE_NOTE) -> str:
    groups: dict[tuple[str, float], list[Row]] = defaultdict(list)
    for row in rows:
        groups[(row.model, row.wait_s)].append(row)
    pairs = len({r.clip for r in rows})
    lines = [f"# End-of-turn check ({when})", "",
             f"{pairs} {note}", "",
             f"## Each model at its own point (at most {limit:.0%} cut-offs when choosing)", "",
             "| Model | Quiet when asked | Cut-offs | Wasted waits | Separation | Points chosen |", "|---|---|---|---|---|---|"]
    for (name, wait), mine in sorted(groups.items(), key=lambda item: (item[0][1], item[0][0])):
        got, points, _ = cross_marked(mine, limit)
        lines.append(f"| {name} | {wait} s | {_share(got.cutoffs, got.unfinished)} | {_share(got.wasted, got.finished)} | "
                     f"{separation(mine):.2f} | {points[0]:.3f}, {points[1]:.3f} |")
    lines += ["", "## At the plain 0.5 point (for comparison with the models' own cards)", "",
              "| Model | Quiet when asked | Cut-offs | Wasted waits |", "|---|---|---|---|"]
    for (name, wait), mine in sorted(groups.items(), key=lambda item: (item[0][1], item[0][0])):
        got = mark(mine, 0.5)
        lines.append(f"| {name} | {wait} s | {_share(got.cutoffs, got.unfinished)} | {_share(got.wasted, got.finished)} |")
    lines += ["", "## Speed, one CPU thread", "", "| Model | Usual | Slowest in 20 |", "|---|---|---|"]
    for name in sorted({r.model for r in rows}):
        times = sorted(r.ms for r in rows if r.model == name)
        lines.append(f"| {name} | {times[len(times) // 2]:.0f} ms | {times[int(len(times) * 0.95)]:.0f} ms |")
    for title, field in (("voice", "voice"), ("background", "noise")):
        lines += ["", f"## By {title}, each model at its own point, asked after {WAITS_S[0]} s of quiet", "",
                  f"| Model | {title.capitalize()} | Cut-offs | Wasted waits |", "|---|---|---|---|"]
        for name in sorted({r.model for r in rows}):
            mine = [r for r in rows if r.model == name and r.wait_s == WAITS_S[0]]
            _, _, point_of = cross_marked(mine, limit)
            for value in sorted({getattr(r, field) for r in mine}):
                part = [r for r in mine if getattr(r, field) == value]
                cut = sum(r.kind == UNFINISHED and r.chance >= point_of[f"{r.clip}/{r.kind}"] for r in part)
                wasted = sum(r.kind == FINISHED and r.chance < point_of[f"{r.clip}/{r.kind}"] for r in part)
                kinds = Counter(r.kind for r in part)
                lines.append(f"| {name} | {value} | {_share(cut, kinds[UNFINISHED])} | {_share(wasted, kinds[FINISHED])} |")
    return "\n".join(lines) + "\n"


def save_rows(rows: list[Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(asdict(row)) + "\n" for row in rows), encoding="utf-8")
