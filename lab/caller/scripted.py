"""A scripted caller brain, for lab runs with the puppet robot (Part 5).

The real caller's brain (Part 3, C1-C3) is an AI. To check the lab itself
with known right answers, this stand-in says fixed lines: one after each
robot turn (after 0.8 s of robot quiet, C9), in order. It sits behind the
director exactly where the AI brain will, so race lines, tricks and drops
work the same.

It also says, as plain facts for the scorer, exactly when each of its lines
really started and ended on the caller's clock, and which lines are a real
yes and to what (L8). A real brain will report the same facts.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import Any

from common.wire.protocol import PIECE_MS, SILENT_PIECE
from lab.puppets.food import CallerTurn
from lab.synthetic_voice import SyntheticRecordings, is_speech, synthetic_line

ROBOT_DONE_MS = 800  # C9: the robot is done after this much quiet

Event = Callable[[str, dict[str, Any]], None]


class ScriptMic:
    """The scripted brain's mouth. Knows which line each piece belongs to, so
    it can tell when a line really started and ended, even when the director
    cuts it off (a race line or a planned silence stops the brain)."""

    def __init__(self, on_event: Event) -> None:
        self._pieces: deque[tuple[bytes, int]] = deque()  # (piece, line number)
        self._lines: dict[int, dict[str, Any]] = {}
        self._next_line = 0
        self._current: int | None = None
        self._on_event = on_event

    @property
    def speaking(self) -> bool:
        return bool(self._pieces)

    @property
    def queued_pieces(self) -> int:
        return len(self._pieces)

    def say_line(self, text: str, pieces: list[bytes], agrees: tuple[str, int] | None, language: str,
                 extra: dict[str, Any] | None = None) -> None:
        number = self._next_line
        self._next_line += 1
        self._lines[number] = {"text": text, "agrees": agrees, "pieces": len(pieces), "language": language,
                               "extra": extra or {}}
        self._pieces.extend((piece, number) for piece in pieces)

    def next_piece(self, tick: int) -> bytes:
        if not self._pieces:
            return SILENT_PIECE
        piece, number = self._pieces.popleft()
        line = self._lines[number]
        if self._current != number:
            self._current = number
            line["start_ms"] = tick * PIECE_MS
        if not self._pieces or self._pieces[0][1] != number:
            # Its last piece goes out during this tick: the line ends with it.
            self._finish(number, (tick + 1) * PIECE_MS, cut=False)
        return piece

    def stop(self) -> None:
        if self._pieces and self._current == self._pieces[0][1]:
            number = self._current
            # Started but cut off (by "ruko!" or a planned silence): it ended
            # where it stopped. Counted before the queue is cleared.
            last_ms = self._lines[number].get("start_ms", 0) + self._played(number) * PIECE_MS
            self._pieces.clear()
            self._finish(number, last_ms, cut=True)
        self._pieces.clear()

    def _played(self, number: int) -> int:
        return self._lines[number]["pieces"] - sum(1 for _, n in self._pieces if n == number)

    def _finish(self, number: int, end_ms: int, cut: bool) -> None:
        line = self._lines[number]
        self._current = None
        self._on_event("caller.line", {
            **line["extra"],
            "text": line["text"],
            "language": line["language"],
            "start_ms": line.get("start_ms"),
            "end_ms": end_ms,
            "cut": cut,
            # A real yes only if the whole line was said (L8).
            "agrees": list(line["agrees"]) if line["agrees"] and not cut else None,
        })


class ScriptedCaller:
    """The ear and brain of the scripted caller: hears the robot, and after
    each robot turn says its next line (None: says nothing that turn)."""

    def __init__(self, turns: list[CallerTurn], on_event: Event, voice=synthetic_line,
                 language: str = "hinglish") -> None:
        self._turns = list(turns)
        self._voice = voice
        self._language = language  # the caller's own, for lines that don't name one
        self.mic = ScriptMic(on_event)
        self._on_event = on_event
        self._robot_spoke = False
        self._quiet_ms = 0
        self.director = None  # set once the line is built: it hears what the caller says
        self.recordings = SyntheticRecordings()
        self.error = None

    # The runner's side: the chain caller has real work in each of these (lab/caller/chain.py).

    def wrap_mic(self, mic):
        return mic

    def attach(self, line, director) -> None:
        self.director = director

    async def prepare(self) -> None:
        pass

    def background(self) -> list:
        return []

    async def close(self) -> None:
        pass

    def about(self) -> dict[str, Any]:
        return {"voice": "synthetic stand-in", "brain": "scripted (puppet lab run)"}

    def director_event(self, happened: Any) -> None:
        pass

    def heard(self, piece: bytes | None, caller_ms: int) -> None:
        if piece is not None and is_speech(piece):
            self._robot_spoke, self._quiet_ms = True, 0
            return
        self._quiet_ms += PIECE_MS
        if not (self._robot_spoke and self._quiet_ms >= ROBOT_DONE_MS and self._turns):
            return
        self._robot_spoke = False
        turn = self._turns.pop(0)
        if turn.text is None:
            self._on_event("caller.skipped_turn", {"caller_ms": caller_ms})
            return
        self.mic.say_line(turn.text, self._voice(turn.text), turn.agrees, turn.language or self._language)
        if self.director is not None:
            self.director.notify_caller_said(turn.text)
