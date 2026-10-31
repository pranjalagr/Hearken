"""Breakdown switches: make an action fail, run slow, or lose its reply, on purpose.

A switch belongs to a business action, not to a tool name. That way the same
task breaks the same way in plain mode (`cancel_order`) and in safe mode
(`confirm`). It fires at the save step only, so a call that gets refused by a
rule does not use it up.
"""

from __future__ import annotations

from dataclasses import dataclass

from engine.storage.database import Tx

FAIL = "fail"
SLOW = "slow"
REPLY_LOST = "reply_lost"
BEFORE_SAVE = "before_save"
AFTER_SAVE = "after_save"


@dataclass(frozen=True)
class Switch:
    action: str
    kind: str  # FAIL, SLOW or REPLY_LOST
    # SLOW: the extra wait. REPLY_LOST: how long until the timeout error arrives.
    seconds: float = 0.0
    # SLOW only. Before the save, the action can still be stopped (race moment 3).
    # After the save, it is already done and it is too late (race moment 4).
    delay_at: str | None = None
    times: int = 1  # how many saves it breaks before it goes quiet

    def __post_init__(self) -> None:
        if self.kind not in (FAIL, SLOW, REPLY_LOST):
            raise ValueError(f"unknown switch kind {self.kind!r}")
        if (self.kind == SLOW) != (self.delay_at in (BEFORE_SAVE, AFTER_SAVE)):
            raise ValueError("a slow switch needs delay_at; other kinds must not have one")
        if self.seconds < 0 or self.times < 1:
            raise ValueError("seconds must be >= 0 and times >= 1")

    @property
    def waits_before_save(self) -> bool:
        return self.kind == SLOW and self.delay_at == BEFORE_SAVE

    def note(self) -> str:
        """Short text for the action log."""
        if self.kind == SLOW:
            return f"slow {self.seconds:g}s {self.delay_at.replace('_', ' ')}"
        if self.kind == REPLY_LOST:
            return f"reply lost: agent got a timeout after {self.seconds:g}s"
        return "fail: nothing was saved"


class Switchboard:
    async def add(self, tx: Tx, switch: Switch) -> None:
        await tx.execute(
            "INSERT INTO breakdown_switches (action, kind, seconds, delay_at, times)"
            " VALUES (?, ?, ?, ?, ?)",
            (switch.action, switch.kind, switch.seconds, switch.delay_at, switch.times),
        )

    async def peek(self, tx: Tx, action: str) -> Switch | None:
        """The switch for this action, if it still has uses left. Changes nothing."""
        row = await tx.fetch_one(
            "SELECT * FROM breakdown_switches WHERE action = ? AND used < times", (action,)
        )
        if row is None:
            return None
        return Switch(row["action"], row["kind"], row["seconds"], row["delay_at"], row["times"])

    async def claim(self, tx: Tx, switch: Switch | None) -> Switch | None:
        """Use the switch once, at the save step. None if it was already used up."""
        if switch is None:
            return None
        touched = await tx.execute(
            "UPDATE breakdown_switches SET used = used + 1 WHERE action = ? AND used < times",
            (switch.action,),
        )
        return switch if touched == 1 else None
