"""The caller's side of one call's black box (Part 3, C16; Part 5, L6).

During the call it only collects, cheaply, inside the 20 ms beat: each piece
the caller sent, each robot piece the caller heard, timed events, and which
steady-clock time each piece went out at. At the end it writes them to the
call's folder:

    caller.json     who the caller was (task, caller facts, director plan, seed)
    events.jsonl    every caller-side event, with its time on the call clock
    caller.flac     the caller's sound as sent
    robot.flac      the robot's sound as the caller heard it
    clock.json      call-clock tick -> steady-clock time (L6)

L6: the steady clock (time.perf_counter_ns) is one clock for every program on
the machine. The clock map turns any program's steady time into call time.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from common.audio.pcm import pcm16_to_float
from common.audio.sound_files import write_sound
from common.wire.protocol import PIECE_MS, SAMPLE_RATE, SILENT_PIECE


class CallerRecorder:
    def __init__(self, call_id: str, clock_ns: Callable[[], int] = time.perf_counter_ns) -> None:
        self.call_id = call_id
        self._clock_ns = clock_ns
        self.sent: dict[int, bytes] = {}
        self.heard_pieces: dict[int, bytes | None] = {}
        self.ticks: list[int] = []
        self.perf_ns: list[int] = []
        self.events: list[dict[str, Any]] = []
        self.last_ms = 0

    # ---- during the call -----------------------------------------------------------

    def tap(self, tick: int, piece: bytes) -> None:
        """Every piece the caller sends (director lines included)."""
        self.sent[tick] = piece

    def heard(self, piece: bytes | None, caller_ms: int) -> None:
        """The caller's ear: every tick, after its piece was sent."""
        tick = caller_ms // PIECE_MS
        self.heard_pieces[tick] = piece
        self.ticks.append(tick)
        self.perf_ns.append(self._clock_ns())
        self.last_ms = caller_ms

    def event(self, name: str, fields: dict[str, Any] | None = None) -> None:
        record = {"event": name, "at_ns": self._clock_ns(), **(fields or {})}
        self.events.append(record)

    def director_event(self, happened: Any) -> None:
        """The director's report as it happens: a line it said, or a drop."""
        name = "director.dropped" if type(happened).__name__ == "Dropped" else "director.said"
        self.event(name, asdict(happened) if is_dataclass(happened) else {"what": repr(happened)})

    def cleared(self, caller_ms: int) -> None:
        """The robot cleared the caller's speaker: it stopped mid-sentence."""
        self.event("robot.cleared", {"caller_ms": caller_ms})

    def bell(self, heard: Any) -> None:
        """A backend bell, with the backend's own steady time (L6)."""
        bell = heard.bell
        self.event("bell", {"bell": bell.bell, "action": bell.action, "perf_ns": bell.perf_ns})

    # ---- after the call ------------------------------------------------------------

    def save(self, folder: Path, about: dict[str, Any]) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "caller.json").write_text(json.dumps(about, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
        with (folder / "events.jsonl").open("w", encoding="utf-8") as out:
            for record in self.events:
                out.write(json.dumps(record, ensure_ascii=False, sort_keys=True, default=str) + "\n")
        (folder / "clock.json").write_text(json.dumps({"ticks": self.ticks, "perf_ns": self.perf_ns}), encoding="utf-8")
        last = max([*self.sent, *self.heard_pieces], default=-1)
        if last < 0:
            return  # the call never got going: no sound, and an empty FLAC can't be read back
        write_sound(folder / "caller.flac", _track(self.sent, last), SAMPLE_RATE)
        write_sound(folder / "robot.flac", _track(self.heard_pieces, last), SAMPLE_RATE)


def _track(pieces: dict[int, bytes | None], last: int) -> np.ndarray:
    """One continuous track, a piece per tick; silence where nothing was sent or heard."""
    joined = b"".join(pieces.get(tick) or SILENT_PIECE for tick in range(last + 1))
    return pcm16_to_float(joined) if joined else np.zeros(0, dtype=np.float32)


class Ears:
    """Several ears on one line: every tick goes to each, in order."""

    def __init__(self, *ears: Any) -> None:
        self._ears = ears

    def heard(self, piece: bytes | None, caller_ms: int) -> None:
        for ear in self._ears:
            ear.heard(piece, caller_ms)
