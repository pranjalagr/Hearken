"""Bells: the backend tells the lab's director the instant something happens
(Part 3, C10), so a race test can speak at an exact moment of an action.

    prepared  a risky action was prepared: the yes-question comes next
    arrived   a risky action's confirm (or, in plain mode, the action) arrived
    saving    its normal wait before the save starts: it is "running" now
    saved     its change is committed; the reply has not been delivered yet

This is a lab-only side line. It is not an AI tool and robots never see it.
A bell must never slow down or break a tool call: ringing only drops a line
into a queue, and a dead bell line just stops ringing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass
from typing import Protocol

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed

PREPARED = "prepared"
ARRIVED = "arrived"
SAVING = "saving"
SAVED = "saved"
BELLS = (PREPARED, ARRIVED, SAVING, SAVED)

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Bell:
    bell: str
    action: str
    call_id: str
    at: float  # pretend-clock seconds, the backend's own time for the event
    perf_ns: int  # time.perf_counter_ns when rung; one clock for all processes on a machine

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @staticmethod
    def from_json(text: str) -> Bell:
        """Strict, like the wire: a bell the lab misreads would aim a race test
        at the wrong moment."""
        body = json.loads(text)
        expected = {"bell": str, "action": str, "call_id": str, "at": (int, float), "perf_ns": int}
        if not isinstance(body, dict) or set(body) != set(expected):
            raise ValueError(f"a bell needs exactly the fields {sorted(expected)}")
        for name, kind in expected.items():
            if not isinstance(body[name], kind) or isinstance(body[name], bool):
                raise ValueError(f"bell.{name} has the wrong type")
        if body["bell"] not in BELLS:
            raise ValueError(f"unknown bell {body['bell']!r}")
        return Bell(**body)


class BellLine(Protocol):
    def ring(self, bell: str, action: str, at: float) -> None:
        """Must return at once and must never raise."""


class NoBells:
    """The default: real calls and most tests have no director listening."""

    def ring(self, bell: str, action: str, at: float) -> None:
        pass


NO_BELLS = NoBells()


class RecordingBells:
    """For tests: keeps every bell, in order."""

    def __init__(self, call_id: str = "test") -> None:
        self.call_id = call_id
        self.rung: list[Bell] = []

    def ring(self, bell: str, action: str, at: float) -> None:
        self.rung.append(Bell(bell, action, self.call_id, at, time.perf_counter_ns()))


class SocketBells:
    """Rings bells over a WebSocket to the lab's bell board.

    `ring` only queues the bell; one background task sends them in order.
    So a slow or dead lab can never hold up the backend's transaction.
    """

    def __init__(self, ws: ClientConnection, call_id: str) -> None:
        self._ws = ws
        self._call_id = call_id
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._dead = False
        self._sender = asyncio.create_task(self._send_all())

    @classmethod
    async def connect(cls, url: str, call_id: str) -> SocketBells:
        """Fails loudly: if the lab asked for bells and can't get them, every
        race test would silently miss its moment."""
        ws = await connect(url, compression=None, proxy=None, open_timeout=5)
        return cls(ws, call_id)

    def ring(self, bell: str, action: str, at: float) -> None:
        if not self._dead:
            self._queue.put_nowait(Bell(bell, action, self._call_id, at, time.perf_counter_ns()).to_json())

    async def aclose(self) -> None:
        """Send what is still queued (briefly), then hang up."""
        try:
            await asyncio.wait_for(self._queue.join(), 2)
        except TimeoutError:
            _log.warning("bell line: %d bells were never sent", self._queue.qsize())
        self._sender.cancel()
        await asyncio.gather(self._sender, return_exceptions=True)
        await self._ws.close()

    async def _send_all(self) -> None:
        while True:
            message = await self._queue.get()
            try:
                await self._ws.send(message)
            except ConnectionClosed:
                # The lab went away: stop ringing rather than pile up bells.
                self._dead = True
                _log.warning("bell line closed; later bells are dropped")
                self._queue.task_done()
                self._drain()
                return
            self._queue.task_done()

    def _drain(self) -> None:
        while not self._queue.empty():
            self._queue.get_nowait()
            self._queue.task_done()
