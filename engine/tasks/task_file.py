"""Task files (decision 13): one YAML file per task.

A task says what the caller wants (goal, for people), which seed data it
uses, how the lab breaks things (switches, timeline), and what the database
must look like at the end (expect), plus what must never happen (forbid).

It also says who the pretend caller is (`caller:`, Part 3 C6) and, for race
and drop tests, what the director does (`race:` C10, `drop:` C12).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from engine.lab_controls.switches import AFTER_SAVE, REPLY_LOST, SLOW, Switch
from engine.lab_controls.timeline import PlannedChange
from engine.storage.call_settings import PLAIN, SAFE
from engine.tasks.caller_spec import CallerSpec, DropSpec, RaceSpec, parse_caller, parse_drop, parse_race

# Table and column names from a task file end up inside SQL, so only plain
# names are allowed. Tasks are our own files, but a typo should fail loudly,
# not run strange SQL.
_NAME = re.compile(r"^[a-z_][a-z0-9_]*$")


@dataclass(frozen=True)
class Expectation:
    """One check on the final database.

    With `fields`: exactly one row matches `where`, and it has these values.
    With `count`: exactly this many rows match `where`.
    """

    table: str
    where: dict[str, Any]
    fields: dict[str, Any] | None = None
    count: int | None = None


@dataclass(frozen=True)
class Forbidden:
    """An action that must never really happen in this task (a refused or
    prepared-only attempt is fine; an actual change is not)."""

    action: str


@dataclass(frozen=True)
class Task:
    id: str
    pack: str
    goal: str  # what the pretend caller wants, in plain words
    uses: list[Any]  # which seed records the task is about
    modes: tuple[str, ...]  # the modes this task makes sense in
    switches: list[Switch] = field(default_factory=list)
    timeline: list[PlannedChange] = field(default_factory=list)
    expect: list[Expectation] = field(default_factory=list)
    forbid: list[Forbidden] = field(default_factory=list)
    # Required in task files. Optional here only so checker tests can build
    # a bare task in code.
    caller: CallerSpec | None = None
    race: RaceSpec | None = None
    drop: DropSpec | None = None


def load_task(path: Path) -> Task:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        if "caller" not in raw:
            # Without it, no pretend caller could ever run this task.
            raise ValueError("every task needs a caller: section")
        task = Task(
            id=raw["id"],
            pack=raw["pack"],
            goal=raw["goal"].strip(),
            uses=list(raw.get("uses", [])),
            # Some races only exist in one mode: "right after prepare" never
            # happens in plain mode, which has no prepare step.
            modes=tuple(raw.get("modes", [PLAIN, SAFE])),
            switches=[Switch(**item) for item in raw.get("switches", [])],
            timeline=[PlannedChange(**item) for item in raw.get("timeline", [])],
            expect=[Expectation(**item) for item in raw.get("expect", [])],
            forbid=[Forbidden(**item) for item in raw.get("forbid", [])],
            caller=parse_caller(raw["caller"]),
            race=parse_race(raw["race"]) if "race" in raw else None,
            drop=parse_drop(raw["drop"]) if "drop" in raw else None,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"bad task file {path.name}: {exc}") from exc
    _check_names(task, path)
    _check_director_plan(task, path)
    return task


def load_tasks(folder: Path) -> list[Task]:
    return [load_task(path) for path in sorted(folder.glob("*.yaml"))]


def _check_names(task: Task, path: Path) -> None:
    for mode in task.modes:
        if mode not in (PLAIN, SAFE):
            raise ValueError(f"bad task file {path.name}: unknown mode {mode!r}")
    for item in task.expect:
        names = [item.table, *item.where, *(item.fields or {})]
        bad = [name for name in names if not _NAME.match(name)]
        if bad:
            raise ValueError(f"bad task file {path.name}: bad names {bad}")
        if (item.fields is None) == (item.count is None):
            raise ValueError(f"bad task file {path.name}: give either fields or count for {item.table}")


def _check_director_plan(task: Task, path: Path) -> None:
    """Race and drop plans that could never happen as written are task bugs."""
    bad = f"bad task file {path.name}:"
    has_drop_trick = task.caller is not None and "drops_call" in task.caller.tricks
    if has_drop_trick != (task.drop is not None):
        raise ValueError(f"{bad} the drops_call trick and a drop: section go together")
    race = task.race
    if race is None:
        return
    if race.moment == "while_asking" and task.modes != (SAFE,):
        # Plain mode never asks "cancel kar doon?": there is no prepare step.
        raise ValueError(f"{bad} a while_asking race needs modes: [safe]")
    if race.moment == "saved_not_heard":
        # Without a delay after the save, the reply reaches the robot at once
        # and there is no moment to hit (C10).
        gap = any(
            s.action == race.action and (s.kind == REPLY_LOST or (s.kind == SLOW and s.delay_at == AFTER_SAVE))
            for s in task.switches
        )
        if not gap:
            raise ValueError(f"{bad} a saved_not_heard race needs a reply_lost or slow after_save switch on {race.action}")
