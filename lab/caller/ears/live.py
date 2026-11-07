"""The pretend caller's ears (Part 3, C2): real hearing, not a text shortcut.

The caller gets the robot's sound the way a person on the phone does, so a robot that
speaks badly is heard badly. A shortcut would hide that, and Agent B has no text.

`feed` runs inside the 20 ms beat, so it only appends to a list. A worker streams the
sound out, using the same connections the ears test proved on real services. Final
words only: partial words change, and the caller answers finished sentences (C2).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable

from common.wire.protocol import PIECE_BYTES, SILENT_PIECE
from lab.ears_test.services import Heard, Service

# The size each service's stream was tested with, in the ears test.
PIECES_PER_SEND = 5  # 100 ms


class _Words(Heard):
    """The services' way of reporting words, turned into a call for each final."""

    def __init__(self, on_words: Callable[[str], None]) -> None:
        super().__init__(lambda: 0.0)
        self._on_words = on_words

    def final(self, text: str) -> None:
        if text.strip():
            self._on_words(text.strip())

    def final_piece(self, piece: str) -> None:
        self.final(piece)

    def partial(self, text: str) -> None:
        pass  # C2: final words only


class LiveEars:
    def __init__(self, service: Service, on_words: Callable[[str], None]) -> None:
        self._service = service
        self._on_words = on_words
        self._waiting: list[bytes] = []
        self._ready = asyncio.Event()
        self._ended = False
        self.seconds = 0.0

    def feed(self, piece: bytes | None) -> None:
        """Every tick of the call: the speaker's piece, or silence. A phone line
        doesn't stop between sentences, and services notice pauses from the sound."""
        self._waiting.append(piece if piece is not None else SILENT_PIECE)
        if len(self._waiting) >= PIECES_PER_SEND:
            self._ready.set()

    def end(self) -> None:
        self._ended = True
        self._ready.set()

    async def _chunks(self) -> AsyncIterator[bytes]:
        while True:
            await self._ready.wait()
            self._ready.clear()
            pieces, self._waiting = self._waiting, []
            if pieces:
                self.seconds += len(pieces) * PIECE_BYTES / 16000
                yield b"".join(pieces)
            if self._ended:
                return

    async def run(self) -> None:
        """Runs for the whole call. A service error comes out, and the call
        becomes a lab problem: a caller gone deaf can't fairly test a robot."""
        await self._service.stream(self._chunks(), _Words(self._on_words))
