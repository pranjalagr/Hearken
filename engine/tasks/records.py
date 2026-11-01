"""The business records at one moment, and what changed between two moments
(Part 5, L7 and L8).

A pack's own tables are its business records (orders, refunds, hand-overs...);
the engine's tables (log, tokens, switches) are not, because their rows always
differ between runs. Comparing the records before and after a call answers
"did anything change that the task didn't expect?", which the task's own
expect list can't: it only looks at the rows it names.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from engine.contract.pack import Pack
from engine.storage.database import Database

_CREATE_TABLE = re.compile(r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([a-z_][a-z0-9_]*)", re.IGNORECASE)

Row = dict[str, Any]
Key = tuple[Any, ...]


def business_tables(pack: Pack) -> list[str]:
    """The pack's own tables, read from its schema, so a new table is
    covered without anyone remembering to list it."""
    return _CREATE_TABLE.findall(pack.schema_sql())


@dataclass(frozen=True)
class Snapshot:
    tables: dict[str, dict[Key, Row]]


@dataclass(frozen=True)
class Change:
    table: str
    kind: str  # added, removed, changed
    before: Row | None
    after: Row | None
    fields: tuple[str, ...] = field(default_factory=tuple)  # changed: which columns

    @property
    def row(self) -> Row:
        """The row as it is now, or as it was if it was removed."""
        return self.after if self.after is not None else self.before

    def describe(self) -> str:
        if self.kind == "changed":
            what = ", ".join(f"{name} {self.before[name]!r} -> {self.after[name]!r}" for name in self.fields)
            return f"{self.table} changed: {what}"
        return f"{self.table} {self.kind}: {self.row}"


async def take_snapshot(db: Database, tables: list[str]) -> Snapshot:
    taken: dict[str, dict[Key, Row]] = {}
    async with db.transaction(write=False) as tx:
        for table in tables:
            # Names come from the pack's own schema, never from outside input.
            info = await tx.fetch_all(f"PRAGMA table_info({table})")
            key_columns = [col["name"] for col in sorted(info, key=lambda c: c["pk"]) if col["pk"]]
            rows = await tx.fetch_all(f"SELECT rowid AS _rowid, * FROM {table}")
            by_key: dict[Key, Row] = {}
            for row in rows:
                rowid = row.pop("_rowid")
                # A table without a primary key is still compared, by its row id.
                by_key[tuple(row[c] for c in key_columns) if key_columns else (rowid,)] = row
            taken[table] = by_key
    return Snapshot(taken)


def changes(before: Snapshot, after: Snapshot) -> list[Change]:
    found: list[Change] = []
    for table, old_rows in before.tables.items():
        new_rows = after.tables.get(table, {})
        for key, old in old_rows.items():
            new = new_rows.get(key)
            if new is None:
                found.append(Change(table, "removed", old, None))
            elif new != old:
                changed = tuple(name for name in old if old[name] != new.get(name))
                found.append(Change(table, "changed", old, new, changed))
        for key, new in new_rows.items():
            if key not in old_rows:
                found.append(Change(table, "added", None, new))
    return found
