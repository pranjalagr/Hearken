"""The fast answer-key check (Part 5, L2): play a puppet's tool calls straight
against the backend, with no voice and pretend time, then mark the call with
the task checker. A perfect robot must pass; a careless one must fail.

Used by the test suite and by the lab's health check (L13), so "the tasks are
right" is checked the same way in both places.
"""

from __future__ import annotations

import asyncio
import itertools
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from engine.contract.pack import Pack
from engine.lab_controls.clock import ManualClock
from engine.lab_controls.normal_delays import NO_DELAYS
from engine.safety.guard_door import GuardDoor
from engine.service.backend import Backend
from engine.service.replies import ToolReply
from engine.storage.call_factory import CallFactory
from engine.storage.call_settings import SAFE, read_call_settings
from engine.storage.database import sqlite_session
from engine.tasks.records import business_tables, take_snapshot
from engine.tasks.task_checker import TaskVerdict, check_task
from engine.tasks.task_file import Task
from engine.tasks.task_setup import start_call
from lab.dummy_robot.robot import RetryLast, Step, UseTool

# The pause before a confirm: the robot reads the question and the caller
# answers. Pretend time, so it costs nothing; it keeps tokens from looking
# instant, as in a real call.
ASK_AND_ANSWER_S = 6.0

_numbers = itertools.count(1)


async def check_answer_key(pack: Pack, task: Task, steps: list[Step], mode: str, workdir: Path) -> TaskVerdict:
    call_id = f"key-{task.id}-{next(_numbers)}"
    path = await start_call(CallFactory(pack, workdir), pack, task, call_id, mode, mode == SAFE, NO_DELAYS)
    async with sqlite_session(path) as db, sqlite_session(path) as guard_db:
        before = await take_snapshot(db, business_tables(pack))
        clock = ManualClock()

        async def pretend_sleep(seconds: float) -> None:
            clock.advance(seconds)
            await asyncio.sleep(0)

        settings = await read_call_settings(db)
        backend = Backend(db, pack, clock, settings, sleep=pretend_sleep)
        guard = GuardDoor(guard_db, clock)
        await play_tools(steps, backend.call_tool, guard.issue_ticket, clock.advance)
        return await check_task(db, task, before)


async def play_tools(
    steps: list[Step],
    call_tool: Callable[[str, dict[str, Any]], Awaitable[ToolReply]],
    issue_ticket: Callable[[str], Awaitable[str]],
    advance: Callable[[float], None],
) -> list[ToolReply]:
    """Only the tool calls of a puppet script, in order: lines and listening
    have no effect on the records. Returns every reply."""
    replies: list[ToolReply] = []
    token: str | None = None
    last: tuple[str, dict[str, Any]] | None = None
    for step in steps:
        if isinstance(step, RetryLast):
            replies.append(await call_tool(*last))
            continue
        if not isinstance(step, UseTool):
            continue
        inputs = dict(step.inputs)
        if step.approve:
            advance(ASK_AND_ANSWER_S)
            inputs = {"token": token, "ticket": await issue_ticket(token)}
        last = (step.tool, inputs)
        reply = await call_tool(step.tool, inputs)
        replies.append(reply)
        if step.tool.startswith("prepare_") and "token" in reply.data:
            token = reply.data["token"]
    return replies
