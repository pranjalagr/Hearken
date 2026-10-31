"""The robot's end of the wire: it answers calls.

The robot reads one stream of events (the caller's audio, keys, receipts,
hang-up) and sends its own audio as fast as its voice service makes it, with
no pacing (Part 2 decision 22); the caller's speaker keeps the real speed.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass

from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from common.wire.protocol import (
    CLOSE_NORMAL,
    CLOSE_PROTOCOL_ERROR,
    MAX_MESSAGE_BYTES,
    PIECE_MS,
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

# How long a new connection may take to say who it is.
START_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class CallerAudio:
    """The caller's k-th piece; it started at caller_ms = 20 x k."""

    piece: bytes
    caller_ms: int


@dataclass(frozen=True)
class LineClosed:
    """The connection closed without a goodbye message."""


CallerEvent = CallerAudio | Key | Played | Cleared | End | LineClosed


class RobotLine:
    def __init__(self, connection: ServerConnection, start: Start) -> None:
        self._connection = connection
        self.start = start
        self.pieces_received = 0

    @classmethod
    async def accept(cls, connection: ServerConnection, timeout_s: float = START_TIMEOUT_S) -> RobotLine:
        """The first message must be "start"; anything else is not a call."""
        try:
            first = await asyncio.wait_for(connection.recv(), timeout_s)
        except TimeoutError:
            raise ProtocolError("no start message") from None
        if isinstance(first, bytes):
            raise ProtocolError("audio before the start message")
        start = decode(first)
        if not isinstance(start, Start):
            raise ProtocolError(f"the first message must be start, not {type(start).__name__.lower()}")
        return cls(connection, start)

    async def events(self) -> AsyncIterator[CallerEvent]:
        """Everything the caller sends, in order. Ends after End or LineClosed."""
        try:
            async for message in self._connection:
                if isinstance(message, bytes):
                    check_piece(message)
                    yield CallerAudio(message, self.pieces_received * PIECE_MS)
                    self.pieces_received += 1
                    continue
                event = decode(message)
                if not isinstance(event, (Key, Played, Cleared, End)):
                    raise ProtocolError(f"a caller may not send {type(event).__name__.lower()} now")
                yield event
                if isinstance(event, End):
                    return
        except ConnectionClosed:
            pass
        yield LineClosed()

    async def send_audio(self, piece: bytes) -> None:
        check_piece(piece)
        await self._connection.send(piece)

    async def send_mark(self, name: str) -> None:
        await self._connection.send(encode(Mark(name)))

    async def clear(self) -> None:
        await self._connection.send(encode(Clear()))

    async def end(self, reason: str) -> None:
        """The robot hangs up: say why, then close."""
        try:
            await self._connection.send(encode(End(reason)))
        except ConnectionClosed:
            return
        await self._connection.close(CLOSE_NORMAL)


RobotApp = Callable[[RobotLine], Awaitable[None]]


async def serve_robot(app: RobotApp, host: str = "127.0.0.1", port: int = 0) -> Server:
    """Answer calls on ws://host:port. Each call runs `app` with its own line.

    A caller that breaks the wire's rules is cut off with close code 1002,
    and the robot app never sees a half-understood call.
    """

    async def handler(connection: ServerConnection) -> None:
        try:
            line = await RobotLine.accept(connection)
            await app(line)
        except ProtocolError as exc:
            await connection.close(CLOSE_PROTOCOL_ERROR, close_reason(str(exc)))

    return await serve(handler, host, port, compression=None, max_size=MAX_MESSAGE_BYTES)


def bound_port(server: Server) -> int:
    """The port a server started with port=0 really listens on."""
    return server.sockets[0].getsockname()[1]
