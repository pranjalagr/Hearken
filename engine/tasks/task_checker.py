"""Check a finished call against its task (decision 12).

Only the fields that matter are compared, never the whole database: times,
tokens and log rows always differ between runs, so a full comparison would
always fail.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from engine.storage.database import Database
from engine.tasks.task_file import Task


@dataclass
class TaskVerdict:
    passed: bool
    problems: list[str] = field(default_factory=list)


async def check_task(db: Database, task: Task) -> TaskVerdict:
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
    return TaskVerdict(passed=not problems, problems=problems)
