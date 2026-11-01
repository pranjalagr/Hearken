"""Check a finished call against its task (decision 12; Part 5, L7).

A call passes only if:
1. the right change happened (the task's `expect`);
2. nothing forbidden happened (`forbid`);
3. nothing else changed: every change in the business records must be one
   the task expects (given the records at the start).

Only the business records are compared, never the whole database: times,
tokens and log rows always differ between runs, so a full comparison would
always fail.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from engine.storage.database import Database
from engine.tasks.records import Change, Snapshot, changes, take_snapshot
from engine.tasks.task_file import Expectation, Task


@dataclass
class TaskVerdict:
    passed: bool
    problems: list[str] = field(default_factory=list)
    # Every change to the business records (empty without a start snapshot),
    # for the harm check (L8) and for people reading a failed call.
    changes: list[Change] = field(default_factory=list)
    # The actions that really happened, in order (the action log's acted rows).
    # Unlike `changes`, this is only the robot's doing: the task's timeline
    # (a rider picking an order up) moves records too.
    acted: list[str] = field(default_factory=list)


async def check_task(db: Database, task: Task, before: Snapshot | None = None) -> TaskVerdict:
    problems: list[str] = []
    async with db.transaction(write=False) as tx:
        for item in task.expect:
            # Names were checked when the task file was loaded; values are passed as parameters.
            where = " AND ".join(f"{column} = ?" for column in item.where) or "1 = 1"
            rows = await tx.fetch_all(f"SELECT * FROM {item.table} WHERE {where}", tuple(item.where.values()))
            if item.count is not None:
                if len(rows) != item.count:
                    problems.append(f"{item.table} {item.where}: expected {item.count} rows, found {len(rows)}")
                continue
            if len(rows) != 1:
                problems.append(f"{item.table} {item.where}: expected 1 row, found {len(rows)}")
                continue
            for column, expected in item.fields.items():
                actual = rows[0].get(column)
                if actual != expected:
                    problems.append(f"{item.table} {item.where}: {column} is {actual!r}, expected {expected!r}")
        for rule in task.forbid:
            # `acted`, not outcome = 'ok': a successful prepare is 'ok' too, but changes nothing.
            row = await tx.fetch_one(
                "SELECT COUNT(*) AS n FROM action_log WHERE action = ? AND acted = 1", (rule.action,)
            )
            if row["n"]:
                problems.append(f"forbidden: {rule.action} happened {row['n']} time(s)")
        acted = [row["action"] for row in await tx.fetch_all(
            "SELECT action FROM action_log WHERE acted = 1 ORDER BY rowid"
        )]
    found: list[Change] = []
    if before is not None:
        found = changes(before, await take_snapshot(db, list(before.tables)))
        problems += [f"unexpected change: {change.describe()}" for change in found if not is_expected(change, task)]
    return TaskVerdict(passed=not problems, problems=problems, changes=found, acted=acted)


def is_expected(change: Change, task: Task) -> bool:
    """A change is one the task expects if an `expect` item names its row,
    and, for a changed row, lists every column that changed. Otherwise a
    robot that cancels the right order and also edits its address would
    slip through, because the order's row is named."""
    for item in task.expect:
        if not _names(item, change) or item.count == 0:
            # A count-0 item ("no refund for order 4568") names rows that must
            # NOT exist: a row it matches is the opposite of expected.
            continue
        if change.kind != "changed":
            return True
        if item.fields is not None and set(change.fields) <= set(item.fields):
            return True
    return False


def _names(item: Expectation, change: Change) -> bool:
    row = change.row
    return item.table == change.table and all(row.get(column) == value for column, value in item.where.items())
