"""The timeline: planned changes that happen on their own during a call.

It plays the pretend kitchen and rider. Without it nothing would ever move
during a call, so a race like "the order left while the caller said haan"
could never happen, and the re-check at confirm would never be tested.

Every change has one trigger: "at second X", or "right after event Y". Race
tests use events, so the change lands at the same point in every run however
fast or slow the agent is.
"""

from __future__ import annotations

from dataclasses import dataclass

from engine.contract.pack import Pack
from engine.storage.database import Row, Tx

PLANNED = "planned"
APPLIED = "applied"
SKIPPED = "skipped"


def prepared_event(action: str) -> str:
    """Fires right after a successful prepare, e.g. "prepared:cancel_order"."""
    return f"prepared:{action}"


def done_event(action: str) -> str:
    """Fires right after an action is really done, e.g. "done:cancel_order"."""
    return f"done:{action}"


@dataclass(frozen=True)
class PlannedChange:
    target: str  # what changes, in the pack's own words, e.g. "4521"
    change: str  # the change, in the pack's own words, e.g. "out_for_delivery"
    at_second: float | None = None
    after_event: str | None = None

    def __post_init__(self) -> None:
        if (self.at_second is None) == (self.after_event is None):
            raise ValueError("a planned change needs exactly one trigger: at_second or after_event")


class Timeline:
    def __init__(self, pack: Pack) -> None:
        self._pack = pack

    async def add(self, tx: Tx, change: PlannedChange) -> None:
        await tx.execute(
            "INSERT INTO timeline (target, change, at_second, after_event) VALUES (?, ?, ?, ?)",
            (change.target, change.change, change.at_second, change.after_event),
        )

    async def apply_due(self, tx: Tx, now: float) -> None:
        """Apply every time-triggered change whose second has come.

        Called at the start of every tool call, inside the same transaction as
        the rule checks, so nothing can slip in between "apply" and "check".
        """
        rows = await tx.fetch_all(
            "SELECT * FROM timeline WHERE state = ? AND at_second <= ? ORDER BY at_second, id",
            (PLANNED, now),
        )
        for row in rows:
            await self._apply(tx, row, now)

    async def fire_event(self, tx: Tx, event: str, now: float) -> None:
        """Apply every change waiting for this event, in the same transaction."""
        rows = await tx.fetch_all(
            "SELECT * FROM timeline WHERE state = ? AND after_event = ? ORDER BY id",
            (PLANNED, event),
        )
        for row in rows:
            await self._apply(tx, row, now)

    async def _apply(self, tx: Tx, row: Row, now: float) -> None:
        # The pack decides whether the change is allowed: timeline changes
        # follow the business rules too.
        verdict = await self._pack.apply_change(tx, row["target"], row["change"])
        await tx.execute(
            "UPDATE timeline SET state = ?, done_at = ?, note = ? WHERE id = ?",
            (APPLIED if verdict.allowed else SKIPPED, now, verdict.reason, row["id"]),
        )
