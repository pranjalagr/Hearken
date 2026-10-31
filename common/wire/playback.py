"""The caller's speaker: plays the robot's audio at real speed and notices marks.

The robot sends its audio as fast as its voice service makes it (Part 2
decision 22), so pieces pile up here and come out one per 20 ms tick, like a
real phone. A mark is "reached" once every piece queued before it has played;
that moment, on the caller's clock, is what the "played" receipt reports.

Pure bookkeeping, no sockets and no clock of its own: the line calls `tick`
once per 20 ms and says what time it is, so this is exact and easy to test.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from common.wire.protocol import PIECE_MS, check_piece


@dataclass(frozen=True)
class _MarkItem:
    name: str


class Playback:
    def __init__(self) -> None:
        self._queue: deque[bytes | _MarkItem] = deque()
        self.played_pieces = 0  # robot pieces really played since the call began

    @property
    def queued_pieces(self) -> int:
        return sum(1 for item in self._queue if isinstance(item, bytes))

    @property
    def is_playing(self) -> bool:
        """True while robot audio is waiting to be played."""
        return any(isinstance(item, bytes) for item in self._queue)

    def add_audio(self, piece: bytes) -> None:
        check_piece(piece)
        self._queue.append(piece)

    def add_mark(self, name: str) -> None:
        self._queue.append(_MarkItem(name))

    def tick(self, start_ms: int) -> tuple[bytes | None, list[tuple[str, int]]]:
        """One 20 ms step, covering caller time [start_ms, start_ms + 20).

        Returns the robot piece played in this step (None: silence) and the
        marks reached, each with its caller time:
        - a mark with nothing left before it is reached at start_ms;
        - a mark right after this step's piece is reached when that piece
          ends, at start_ms + 20.
        """
        reached = [(name, start_ms) for name in self._pop_marks()]
        piece = None
        if self._queue and isinstance(self._queue[0], bytes):
            piece = self._queue.popleft()
            self.played_pieces += 1
            reached += [(name, start_ms + PIECE_MS) for name in self._pop_marks()]
        return piece, reached

    def clear(self) -> list[str]:
        """Throw away everything not yet played. Returns the marks dropped
        with it: their audio never finished, so they will never be reached."""
        dropped = [item.name for item in self._queue if isinstance(item, _MarkItem)]
        self._queue.clear()
        return dropped

    def _pop_marks(self) -> list[str]:
        names = []
        while self._queue and isinstance(self._queue[0], _MarkItem):
            names.append(self._queue.popleft().name)
        return names
