"""The settings of one call, stored in its single `call_info` row."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from engine.lab_controls.normal_delays import NormalDelays
from engine.storage.database import Database, Tx

PLAIN = "plain"
SAFE = "safe"


@dataclass(frozen=True)
class CallSettings:
    call_id: str
    pack: str
    mode: str = SAFE
    ticket_check: bool = True
    task_id: str | None = None
    # Real-world start time, only used by the live clock.
    started_at_wall: float = field(default_factory=time.time)
    # Kept with the call, not only in code, so the black box shows which waits
    # a run used (Part 2 decision 33: every number traces back to its settings).
    delays: NormalDelays = NormalDelays()

    def __post_init__(self) -> None:
        if self.mode not in (PLAIN, SAFE):
            raise ValueError(f"mode must be '{PLAIN}' or '{SAFE}', not {self.mode!r}")
        # Decision 8: plain mode has no confirm step, so there is nothing to check a ticket at.
        if self.mode == PLAIN and self.ticket_check:
            raise ValueError("plain mode cannot have ticket_check on")


async def write_call_settings(tx: Tx, settings: CallSettings) -> None:
    await tx.execute(
        "INSERT INTO call_info (id, call_id, task_id, pack, mode, ticket_check, started_at_wall,"
        " lookup_delay_s, change_delay_s)"
        " VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            settings.call_id,
            settings.task_id,
            settings.pack,
            settings.mode,
            int(settings.ticket_check),
            settings.started_at_wall,
            settings.delays.lookup_s,
            settings.delays.change_s,
        ),
    )


async def read_call_settings(db: Database) -> CallSettings:
    async with db.transaction(write=False) as tx:
        row = await tx.fetch_one("SELECT * FROM call_info WHERE id = 1")
    if row is None:
        raise LookupError("this database has no call_info row: it is not a call copy")
    return CallSettings(
        call_id=row["call_id"],
        pack=row["pack"],
        mode=row["mode"],
        ticket_check=bool(row["ticket_check"]),
        task_id=row["task_id"],
        started_at_wall=row["started_at_wall"],
        delays=NormalDelays(lookup_s=row["lookup_delay_s"], change_s=row["change_delay_s"]),
    )
