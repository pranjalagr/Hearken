"""Shared test helpers.

`open_call` gives a test a fresh call copy, wired up exactly like production
except for time: tests use a ManualClock, so time only moves when the test
moves it, and every run sees the same times.

The normal tool delays are off unless a test asks for them, so tests that
check exact pretend times measure only what they mean to.
"""

from __future__ import annotations

import asyncio
import gc
import itertools
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from engine.contract.pack import Pack
from engine.lab_controls.bells import NO_BELLS, BellLine
from engine.lab_controls.clock import ManualClock
from engine.lab_controls.normal_delays import NO_DELAYS, NormalDelays
from engine.lab_controls.switches import Switch, Switchboard
from engine.lab_controls.timeline import PlannedChange, Timeline
from engine.safety.guard_door import GuardDoor
from engine.safety.lifetimes import Lifetimes
from engine.service.backend import Backend
from engine.service.replies import ToolReply
from engine.storage.call_factory import CallFactory
from engine.storage.call_settings import CallSettings, read_call_settings
from engine.storage.database import SqliteDatabase
from engine.tasks.task_file import Task
from engine.tasks.task_setup import start_call


@dataclass
class CallKit:
    """Everything a test needs to drive one call."""

    path: Path
    clock: ManualClock
    settings: CallSettings
    backend: Backend
    guard: GuardDoor  # the Guard's private door, on its own connection as in Part 2
    db: SqliteDatabase  # the backend's connection, also used to read results

    async def tool(self, name: str, **inputs: Any) -> ToolReply:
        return await self.backend.call_tool(name, inputs)

    async def rows(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        async with self.db.transaction(write=False) as tx:
            return await tx.fetch_all(sql, params)

    async def one(self, sql: str, *params: Any) -> dict[str, Any] | None:
        rows = await self.rows(sql, *params)
        return rows[0] if rows else None


@pytest.fixture
async def open_call(tmp_path: Path):
    opened: list[SqliteDatabase] = []
    factories: dict[str, CallFactory] = {}
    numbers = itertools.count(1)

    async def _open(
        pack: Pack,
        mode: str = "safe",
        ticket_check: bool | None = None,
        switches: tuple[Switch, ...] = (),
        timeline: tuple[PlannedChange, ...] = (),
        task: Task | None = None,
        real_sleep: bool = False,
        lifetimes: Lifetimes = Lifetimes(),
        delays: NormalDelays = NO_DELAYS,
        bells: BellLine = NO_BELLS,
    ) -> CallKit:
        if ticket_check is None:
            ticket_check = mode == "safe"
        factory = factories.setdefault(pack.name, CallFactory(pack, tmp_path))
        call_id = f"call-{next(numbers)}"
        if task is not None:
            path = await start_call(factory, pack, task, call_id, mode, ticket_check, delays)
        else:
            path = await factory.new_call(CallSettings(call_id, pack.name, mode, ticket_check, delays=delays))
        db = await SqliteDatabase.open(path)
        guard_db = await SqliteDatabase.open(path)
        opened.extend([db, guard_db])
        async with db.transaction() as tx:
            for switch in switches:
                await Switchboard().add(tx, switch)
            for change in timeline:
                await Timeline(pack).add(tx, change)

        clock = ManualClock()

        async def pretend_sleep(seconds: float) -> None:
            # Waiting moves pretend time instead of real time, so tests stay fast.
            clock.advance(seconds)
            await asyncio.sleep(0)

        settings = await read_call_settings(db)
        backend = Backend(
            db, pack, clock, settings, lifetimes, sleep=asyncio.sleep if real_sleep else pretend_sleep, bells=bells
        )
        return CallKit(path, clock, settings, backend, GuardDoor(guard_db, clock, lifetimes), db)

    yield _open
    for db in opened:
        await db.close()


FREEZE_LIMIT_S = 0.1  # decision 14
_HEARTBEAT_S = 0.01


def _in_event_loop_plumbing(frame) -> bool:
    """True if the main thread is inside the event loop's own machinery:
    waiting for events, or in the OS socket calls asyncio makes for itself
    (measured here: once a single WSARecv on asyncio's wake-up socket took
    over 100 ms, in a test that opened no sockets at all).

    A late heartbeat while there means the operating system was slow, not our
    code. A late heartbeat while our code, or a library it called, is running
    is a freeze we caused. Socket and selector frames are looked through to
    see who called them, so a blocking socket call in our own code still counts.
    """
    while frame is not None:
        path = frame.f_code.co_filename.replace("\\", "/")
        if "/asyncio/" in path:
            return True
        if not path.endswith(("/socket.py", "/selectors.py")):
            return False
        frame = frame.f_back
    return False


class FreezeWatch:
    """Finds the moments our code blocks the event loop for over 100 ms.

    Part 2 decision 14: every wait must be async. One blocking call would
    freeze every call sharing the worker. How it is measured:
    - a heartbeat task ticks every 10 ms on the event loop;
    - a watcher thread checks it every 5 ms; when the heartbeat is over
      100 ms late, it looks at what the main thread is doing. Waiting for
      events: the OS woke the process late, not a freeze of ours. Running
      code: a freeze, recorded with exactly which code was running.

    Not asyncio's debug mode: it records a stack trace for every small step,
    and under bursts of audio messages that bookkeeping alone took over
    100 ms, so it raised false alarms and slowed the code it was measuring.

    Garbage collection: after hundreds of tests the process holds ~165,000
    objects, and one full collection was measured at 50 ms, enough to push
    a heartbeat past 100 ms. So, as a low-delay service does after start-up,
    the watch parks every existing object first (gc.freeze): collections
    during the test then only scan what the call itself made.
    """

    def __init__(self) -> None:
        self.freezes: list[str] = []
        self._main_thread = threading.get_ident()
        self._last_beat = time.perf_counter()
        self._stop = threading.Event()
        self._beat_task: asyncio.Task | None = None
        self._watcher = threading.Thread(target=self._watch, name="freeze-watcher", daemon=True)

    def start(self) -> None:
        gc.collect()  # before the heartbeat starts, so its own time is never counted
        gc.freeze()
        self._beat_task = asyncio.create_task(self._heartbeat())
        self._watcher.start()

    async def stop(self) -> list[str]:
        self._stop.set()
        self._watcher.join()
        self._beat_task.cancel()
        await asyncio.gather(self._beat_task, return_exceptions=True)
        gc.unfreeze()
        return self.freezes

    async def _heartbeat(self) -> None:
        while True:
            self._last_beat = time.perf_counter()
            await asyncio.sleep(_HEARTBEAT_S)

    def _watch(self) -> None:
        # Every 5 ms while the heartbeat is clearly late, note what the main
        # thread runs; once it is over 100 ms late, judge the whole late
        # stretch: any sample in our code means we caused it.
        #
        # Sampling starts at 50 ms late, not earlier: on Windows a 10 ms sleep
        # often wakes after 16 or 31 ms (the 15.6 ms system timer), which is
        # normal. And a sample only copies file/line/name: formatting a stack
        # reads source files from disk while holding Python's lock, which would
        # make the main thread wait for the watcher, adding the very stall it
        # measures. The readable stack is built after the test.
        current_beat, ours, reported_beat = None, [], None
        while not self._stop.wait(0.005):
            beat = self._last_beat
            if beat != current_beat:
                current_beat, ours = beat, []
            late = time.perf_counter() - beat - _HEARTBEAT_S
            if late <= 0.05:
                continue
            frame = sys._current_frames().get(self._main_thread)
            if frame is not None and not _in_event_loop_plumbing(frame):
                ours.append(_where(frame))
            if late > FREEZE_LIMIT_S and beat != reported_beat:
                reported_beat = beat  # one judgement per freeze
                if ours:
                    self.freezes.append(f"blocked over {FREEZE_LIMIT_S * 1000:.0f} ms in:\n" + "\n".join(ours[0]))


def _where(frame, depth: int = 6) -> list[str]:
    """The innermost frames as "file:line in function", without touching the
    disk (no source lines), so it is quick enough to run during a stall."""
    lines = []
    while frame is not None and len(lines) < depth:
        lines.append(f"  {frame.f_code.co_filename}:{frame.f_lineno} in {frame.f_code.co_name}")
        frame = frame.f_back
    return lines


@pytest.fixture
def make_freeze_watch():
    """For testing the freeze check itself."""
    return FreezeWatch


@pytest.fixture
async def no_freezes():
    """Fail the test if our code ever blocks the event loop for over 100 ms."""
    watch = FreezeWatch()
    watch.start()
    yield
    freezes = await watch.stop()
    assert not freezes, "the event loop froze:\n" + "\n".join(freezes)
