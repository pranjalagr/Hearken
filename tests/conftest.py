"""Shared test helpers.

`open_call` gives a test a fresh call copy, wired up exactly like production
except for time: tests use a ManualClock, so time only moves when the test
moves it, and every run sees the same times.
"""

from __future__ import annotations

import asyncio
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from engine.contract.pack import Pack
from engine.lab_controls.clock import ManualClock
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
    ) -> CallKit:
        if ticket_check is None:
            ticket_check = mode == "safe"
        factory = factories.setdefault(pack.name, CallFactory(pack, tmp_path))
        call_id = f"call-{next(numbers)}"
        if task is not None:
            path = await start_call(factory, pack, task, call_id, mode, ticket_check)
        else:
            path = await factory.new_call(CallSettings(call_id, pack.name, mode, ticket_check))
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
            db, pack, clock, settings, lifetimes, sleep=asyncio.sleep if real_sleep else pretend_sleep
        )
        return CallKit(path, clock, settings, backend, GuardDoor(guard_db, clock, lifetimes), db)

    yield _open
    for db in opened:
        await db.close()
