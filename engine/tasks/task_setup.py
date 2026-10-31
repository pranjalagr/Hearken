"""Turn a task into a ready call copy: a fresh database with the task's
breakdown switches and timeline already written in."""

from __future__ import annotations

from pathlib import Path

from engine.contract.pack import Pack
from engine.lab_controls.normal_delays import NormalDelays
from engine.lab_controls.switches import Switchboard
from engine.lab_controls.timeline import Timeline
from engine.storage.call_factory import CallFactory
from engine.storage.call_settings import CallSettings
from engine.storage.database import Tx, sqlite_session
from engine.tasks.task_file import Task


async def start_call(
    factory: CallFactory,
    pack: Pack,
    task: Task,
    call_id: str,
    mode: str,
    ticket_check: bool,
    delays: NormalDelays = NormalDelays(),
) -> Path:
    if task.pack != pack.name:
        raise ValueError(f"task {task.id} is for pack {task.pack!r}, not {pack.name!r}")
    if mode not in task.modes:
        raise ValueError(f"task {task.id} is not meant for {mode} mode")
    if task.race is not None:
        risky = {action.name for action in pack.actions() if action.risky}
        if task.race.action not in risky:
            raise ValueError(f"task {task.id}: race.action {task.race.action!r} is not a risky action of {pack.name}")
    settings = CallSettings(
        call_id=call_id, pack=pack.name, mode=mode, ticket_check=ticket_check, task_id=task.id, delays=delays
    )
    path = await factory.new_call(settings)
    switches, timeline = Switchboard(), Timeline(pack)
    async with sqlite_session(path) as db:
        async with db.transaction() as tx:
            # Checked on the fresh copy, before the timeline can move anything:
            # the facts must hold at the start of the call (Part 3, C6).
            problems = await _caller_problems(tx, pack, task)
            if not problems:
                for switch in task.switches:
                    await switches.add(tx, switch)
                for change in task.timeline:
                    await timeline.add(tx, change)
    if problems:
        # Don't leave a half-made call copy behind for someone to run by mistake.
        for leftover in (path, path.with_name(path.name + "-wal"), path.with_name(path.name + "-shm")):
            leftover.unlink(missing_ok=True)
        raise ValueError(f"task bug in {task.id}: " + "; ".join(problems))
    return path


async def _caller_problems(tx: Tx, pack: Pack, task: Task) -> list[str]:
    if task.caller is None:
        return []  # a bare task built in code, for checker tests
    return await pack.check_caller(tx, task)
