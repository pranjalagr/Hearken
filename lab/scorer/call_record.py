"""One call's folder, read back with everything on the call clock (Part 5, L6).

Three programs wrote into the folder: the caller (events.jsonl, the audio,
clock.json), the robot (robot.jsonl) and the backend (backend.jsonl, plus
its bells inside the caller's events). Each stamped its lines with the one
shared steady clock; clock.json says which steady time each 20 ms piece of
the call went out at, so any of those times becomes call time.

The scorer reads a call only through this, after the call (L9: measured
afterwards, so a fixed scorer can re-mark old runs for free).
"""

from __future__ import annotations

import bisect
import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from common.audio.sound_files import read_sound
from common.wire.protocol import PIECE_MS, PIECE_SAMPLES

STOP_WORDS = ("ruko", "rukiye", "mat karo", "nahi nahi", "wait", "stop", "ek minute")
BACKCHANNEL_WORDS = ("haan haan", "hmm", "achha", "ji ji")


@dataclass(frozen=True)
class ClockMap:
    """Steady-clock nanoseconds <-> call milliseconds (L6)."""

    ticks: list[int]
    perf_ns: list[int]

    def to_ms(self, perf_ns: int) -> float:
        if not self.perf_ns:
            raise ValueError("the call has no clock map")
        i = bisect.bisect_right(self.perf_ns, perf_ns) - 1
        i = min(max(i, 0), len(self.perf_ns) - 1)
        # From the nearest piece before it, at 20 ms per piece; outside the
        # call it extends the same way (the bell before the first piece).
        return self.ticks[i] * PIECE_MS + (perf_ns - self.perf_ns[i]) / 1e6


@dataclass(frozen=True)
class Span:
    start_ms: float
    end_ms: float


@dataclass
class CallRecord:
    folder: Path
    about: dict[str, Any]
    outcome: dict[str, Any]
    clock: ClockMap
    events: list[dict[str, Any]]
    diary: list[dict[str, Any]]
    backend: list[dict[str, Any]]
    robot_sound: list[Span] = field(default_factory=list)  # at the caller's ear
    sound_by_tick: list[bool] = field(default_factory=list)

    # ---- what the caller said and heard -----------------------------------------------

    def caller_lines(self) -> list[dict[str, Any]]:
        """The caller's own turns (the brain's lines), with their true start and end."""
        return [e for e in self.events if e["event"] == "caller.line" and e.get("start_ms") is not None]

    def director_lines(self) -> list[dict[str, Any]]:
        return [e for e in self.events if e["event"] == "director.said"]

    def stops(self) -> list[float]:
        """When the caller started saying "stop" (ruko...), on the call clock."""
        found = [line["caller_ms"] for line in self.director_lines() if _has(line["line"], STOP_WORDS)]
        found += [line["start_ms"] for line in self.caller_lines() if _has(line["text"], STOP_WORDS)]
        return sorted(found)

    def backchannels(self) -> list[float]:
        return sorted(
            line["caller_ms"] for line in self.director_lines()
            if _has(line["line"], BACKCHANNEL_WORDS) and not _has(line["line"], STOP_WORDS)
        )

    def agreements(self) -> list[dict[str, Any]]:
        """The caller's real yeses: (action, order_no) and when the yes ended (L8)."""
        return [
            {"action": line["agrees"][0], "order_no": line["agrees"][1], "end_ms": line["end_ms"]}
            for line in self.caller_lines() if line.get("agrees")
        ]

    def clears(self) -> list[float]:
        return [e["caller_ms"] for e in self.events if e["event"] == "robot.cleared"]

    def end_ms(self) -> float:
        ends = [e.get("caller_ms", 0) for e in self.events if e["event"] == "call.end"]
        return float(ends[0]) if ends else float(len(self.sound_by_tick) * PIECE_MS)

    # ---- the backend --------------------------------------------------------------

    def bells(self, name: str, action: str | None = None) -> list[float]:
        return [
            self.clock.to_ms(e["perf_ns"]) for e in self.events
            if e["event"] == "bell" and e["bell"] == name and (action is None or e["action"] == action)
        ]

    def tool_spans(self) -> list[tuple[str, Span]]:
        """Every tool call, start to end, from the backend's own lines."""
        spans, open_at = [], {}
        for line in self.backend:
            if line["event"] == "tool.start":
                open_at.setdefault(line["tool"], []).append(line["at_ns"])
            elif line["event"] == "tool.end" and open_at.get(line["tool"]):
                start = open_at[line["tool"]].pop(0)
                spans.append((line["tool"], Span(self.clock.to_ms(start), self.clock.to_ms(line["at_ns"]))))
        return spans

    def final_db(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.folder / "final.db")
        db.row_factory = sqlite3.Row
        return db

    def robot_said(self) -> list[dict[str, Any]]:
        """What the robot meant to say (its own text, filled), from its diary (L10)."""
        return [line for line in self.diary if line["kind"] == "said"]


def load_call(folder: Path, sound_level: float) -> CallRecord:
    clock_raw = json.loads((folder / "clock.json").read_text(encoding="utf-8"))
    record = CallRecord(
        folder=folder,
        about=json.loads((folder / "caller.json").read_text(encoding="utf-8")),
        outcome=_json_or_empty(folder / "outcome.json"),
        clock=ClockMap(clock_raw["ticks"], clock_raw["perf_ns"]),
        events=_jsonl(folder / "events.jsonl"),
        diary=_jsonl(folder / "robot.jsonl"),
        backend=_jsonl(folder / "backend.jsonl"),
    )
    record.sound_by_tick = sound_ticks(folder / "robot.flac", sound_level)
    record.robot_sound = spans_of(record.sound_by_tick)
    return record


def sound_ticks(path: Path, level: float) -> list[bool]:
    """Per 20 ms piece: was there sound above `level` (RMS)? (L9: real sound only.)"""
    if not path.exists():
        return []
    samples, _ = read_sound(path)
    count = len(samples) // PIECE_SAMPLES
    if count == 0:
        return []
    pieces = np.asarray(samples[: count * PIECE_SAMPLES], dtype=np.float64).reshape(count, PIECE_SAMPLES)
    return list(np.sqrt(np.mean(pieces**2, axis=1)) > level)


def spans_of(flags: list[bool]) -> list[Span]:
    spans, start = [], None
    for tick, on in enumerate(flags):
        if on and start is None:
            start = tick
        elif not on and start is not None:
            spans.append(Span(start * PIECE_MS, tick * PIECE_MS))
            start = None
    if start is not None:
        spans.append(Span(start * PIECE_MS, len(flags) * PIECE_MS))
    return spans


def _has(text: str, words: tuple[str, ...]) -> bool:
    """Whole words only: "stop" must not be found inside "nonstop"."""
    lowered = " ".join(str(text).lower().split())
    return any(re.search(rf"(?<![a-z]){re.escape(word)}(?![a-z])", lowered) for word in words)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _json_or_empty(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
