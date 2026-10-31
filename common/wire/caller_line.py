"""The caller's end of the wire, for pretend callers and tests.

Like a phone, it does two things every 20 ms, for the whole call:
- sends one piece from the microphone (silence or noise when nobody speaks:
  a real line never goes quiet, and the robot's "no audio for 30 s" rule
  relies on that);
- plays one piece of the robot's audio, and reports any mark reached.

The microphone and the ear are passed in, so the same line serves the
pretend caller, the director and the tests.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Iterable
from typing import Protocol

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed

from common.wire.pacer import Pacer
from common.wire.playback import Playback
from common.wire.protocol import (
    CLOSE_PROTOCOL_ERROR,
    MAX_MESSAGE_BYTES,
    PIECE_MS,
    SILENT_PIECE,
    Clear,
    Cleared,
    End,
    Key,
    Mark,
    Played,
    ProtocolError,
    Start,
    check_piece,
    close_reason,
    decode,
    encode,
)


class Mic(Protocol):
    def next_piece(self, tick: int) -> bytes:
        """The piece to send during tick `tick`; exactly one 320-byte piece."""


class Ear(Protocol):
    def heard(self, piece: bytes | None, caller_ms: int) -> None:
        """The robot piece played during the tick starting at caller_ms (None: silence)."""


class SilentMic:
    def next_piece(self, tick: int) -> bytes:
        return SILENT_PIECE


class QueueMic:
    """Plays queued pieces in order, then silence: enough for tests and for
    the director's pre-made lines."""

    def __init__(self) -> None:
        self._pieces: deque[bytes] = deque()

    @property
    def speaking(self) -> bool:
        return bool(self._pieces)

    @property
    def queued_pieces(self) -> int:
        return len(self._pieces)

    def say(self, pieces: Iterable[bytes]) -> None:
        for piece in pieces:
            check_piece(piece)
            self._pieces.append(piece)

    def stop(self) -> None:
        self._pieces.clear()

    def next_piece(self, tick: int) -> bytes:
        return self._pieces.popleft() if self._pieces else SILENT_PIECE


class NullEar:
    def heard(self, piece: bytes | None, caller_ms: int) -> None:
        pass


class CallerLine:
    def __init__(
        self,
        url: str,
        call_id: str,
        mic: Mic,
        ear: Ear | None = None,
        pacer: Pacer | None = None,
        playback: Playback | None = None,
    ) -> None:
        self._url = url
        self._call_id = call_id
        self._mic = mic
        self._ear = ear or NullEar()
        self._pacer = pacer or Pacer()
        # Passed in when something (the director) must watch the speaker
        # before the line exists.
        self._playback = playback or Playback()
        self._ws: ClientConnection | None = None
        self._send_lock = asyncio.Lock()
        self._ended = asyncio.Event()
        self._end_reason = ""
        self._error: BaseException | None = None
        self._ticks_done = 0
        self._silent = False

    @property
    def caller_ms(self) -> int:
        """Now, on the caller's clock: 20 ms x pieces sent so far."""
        return self._ticks_done * PIECE_MS

    @property
    def playback(self) -> Playback:
        return self._playback

    @property
    def ended(self) -> bool:
        return self._ended.is_set()

    async def run(self) -> str:
        """Place the call and keep it going until either side ends it.

        Returns why it ended: "robot:<reason>", "caller:<reason>" or "closed".
        Raises ProtocolError if the robot broke the wire's rules.
        """
        async with connect(self._url, compression=None, max_size=MAX_MESSAGE_BYTES, proxy=None) as ws:
            self._ws = ws
            await self._send(encode(Start(self._call_id)))
            tasks = [asyncio.create_task(self._receive()), asyncio.create_task(self._beat())]
            try:
                await self._ended.wait()
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        if self._error is not None:
            raise self._error
        return self._end_reason

    async def press_key(self, digit: str) -> None:
        await self._send(encode(Key(digit, self.caller_ms)))

    async def hang_up(self, reason: str = "bye") -> None:
        """A clean hang-up: the red button."""
        if self._ended.is_set():
            return
        await self._send(encode(End(reason)))
        self._finish(f"caller:{reason}")

    def go_silent(self) -> None:
        """A silent drop: from now on nothing is sent, not even a goodbye,
        like a network dying in a lift. The robot must notice on its own."""
        self._silent = True

    # ---- the two loops ------------------------------------------------------------

    async def _beat(self) -> None:
        try:
            async for tick in self._pacer.ticks():
                start_ms = tick * PIECE_MS
                if not self._silent:
                    piece = self._mic.next_piece(tick)
                    check_piece(piece)
                    await self._send(piece)
                played, reached = self._playback.tick(start_ms)
                self._ear.heard(played, start_ms)
                self._ticks_done = tick + 1
                if not self._silent:
                    for name, at_ms in reached:
                        await self._send(encode(Played(name, at_ms)))
        except ConnectionClosed:
            self._finish("closed")
        except Exception as exc:  # any failure must end the call, never hang it
            self._fail(exc)

    async def _receive(self) -> None:
        try:
            async for message in self._ws:
                if isinstance(message, bytes):
                    self._playback.add_audio(message)
                else:
                    await self._handle(decode(message))
            self._finish("closed")
        except ConnectionClosed:
            self._finish("closed")
        except ProtocolError as exc:
            # Close first, so the robot gets code 1002 before the call is torn down.
            await self._ws.close(CLOSE_PROTOCOL_ERROR, close_reason(str(exc)))
            self._fail(exc)
        except Exception as exc:
            self._fail(exc)

    async def _handle(self, message: object) -> None:
        if isinstance(message, Mark):
            self._playback.add_mark(message.name)
        elif isinstance(message, Clear):
            dropped = self._playback.clear()
            if not self._silent:
                reply = Cleared(self._playback.played_pieces, self.caller_ms, tuple(dropped))
                await self._send(encode(reply))
        elif isinstance(message, End):
            self._finish(f"robot:{message.reason}")
        else:
            raise ProtocolError(f"a robot may not send {type(message).__name__.lower()}")

    # ---- helpers ----------------------------------------------------------------

    async def _send(self, data: str | bytes) -> None:
        # Several tasks send (the beat, the receiver's Cleared, keys, hang-up);
        # one lock keeps their messages whole and in order.
        async with self._send_lock:
            await self._ws.send(data)

    def _finish(self, reason: str) -> None:
        if not self._ended.is_set():
            self._end_reason = reason
            self._ended.set()

    def _fail(self, exc: BaseException) -> None:
        if not self._ended.is_set():
            self._error = exc
            self._ended.set()
