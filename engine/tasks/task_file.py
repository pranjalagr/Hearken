"""Task files (decision 13): one YAML file per task.

A task says what the caller wants (goal), which seed data it uses, how the
lab breaks things (switches, timeline), and what the database must look like
at the end (expect), plus what must never happen (forbid).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from engine.lab_controls.switches import Switch
from engine.lab_controls.timeline import PlannedChange
from engine.storage.call_settings import PLAIN, SAFE

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


def load_task(path: Path) -> Task:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
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
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"bad task file {path.name}: {exc}") from exc
    _check_names(task, path)
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
