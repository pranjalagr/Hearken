"""The lab's end of the bell line (Part 3, C10).

Each call's backend connects here and rings; the director of that call reads
its bells from a queue. Kept strict: a malformed bell closes the connection
with 1002, because a misread bell would aim a race line at the wrong moment.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from common.wire.protocol import CLOSE_PROTOCOL_ERROR, close_reason
from common.wire.robot_line import bound_port
from engine.lab_controls.bells import Bell


@dataclass(frozen=True)
class HeardBell:
    bell: Bell
    received_ns: int  # time.perf_counter_ns when the lab got it


class BellBoard:
    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue[HeardBell]] = {}
        self._server: Server | None = None

    async def start(self, host: str = "127.0.0.1", port: int = 0) -> str:
        """Start listening; returns the URL to give each backend (serve_call --bells)."""
        self._server = await serve(self._handle, host, port, compression=None, max_size=4096)
        return f"ws://{host}:{bound_port(self._server)}/bells"

    def bells_for(self, call_id: str) -> asyncio.Queue[HeardBell]:
        """The queue of one call's bells. Made on first use, so a director can
        start listening before its backend connects."""
        return self._queues.setdefault(call_id, asyncio.Queue())

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def _handle(self, connection: ServerConnection) -> None:
        try:
            async for message in connection:
                if not isinstance(message, str):
                    raise ValueError("bells are text")
                bell = Bell.from_json(message)
                self.bells_for(bell.call_id).put_nowait(HeardBell(bell, time.perf_counter_ns()))
        except ConnectionClosed:
            pass
        except ValueError as exc:
            await connection.close(CLOSE_PROTOCOL_ERROR, close_reason(str(exc)))
