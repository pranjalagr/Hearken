"""Yes-tickets: proof that the caller really said yes to one exact action.

Only the Guard creates tickets, through `GuardDoor`, which is not an AI tool,
so the AI cannot make one. A ticket belongs to one token and keeps a copy of
its exact action, so a ticket for "cancel 4521" never works for "refund 4512".

Rows are never deleted. They move open -> used, or open -> cancelled, so the
black box keeps the whole story.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from engine.safety.tokens import PreparedAction
from engine.storage.database import Row, Tx

OPEN = "open"
USED = "used"
CANCELLED = "cancelled"

TICKET_NEEDED = "A yes-ticket is needed. Read the summary to the caller and wait for a clear yes."


@dataclass(frozen=True)
class Ticket:
    ticket: str
    token: str
    action: str
    inputs: dict[str, Any]
    state: str
    issued_at: float
    expires_at: float


class TicketBook:
    async def create(self, tx: Tx, prepared: PreparedAction, now: float, expires_at: float) -> Ticket:
        count = await tx.fetch_one("SELECT COUNT(*) AS n FROM tickets")
        ticket = f"K-{count['n'] + 1}"
        await tx.execute(
            "INSERT INTO tickets (ticket, token, action, inputs_json, state, issued_at, expires_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                ticket,
                prepared.token,
                prepared.action,
                json.dumps(prepared.inputs, sort_keys=True, ensure_ascii=False),
                OPEN,
                now,
                expires_at,
            ),
        )
        return Ticket(ticket, prepared.token, prepared.action, prepared.inputs, OPEN, now, expires_at)

    async def get(self, tx: Tx, ticket: str) -> Ticket | None:
        row = await tx.fetch_one("SELECT * FROM tickets WHERE ticket = ?", (ticket,))
        return _from_row(row) if row else None

    async def problem_with(
        self, tx: Tx, ticket_id: str | None, prepared: PreparedAction, now: float
    ) -> str | None:
        """Why this ticket cannot approve this prepared action, or None if it can.

        The messages are for the agent: each one says what to do next.
        """
        if not ticket_id:
            return TICKET_NEEDED
        ticket = await self.get(tx, ticket_id)
        if ticket is None:
            return f"Unknown ticket {ticket_id}."
        same_action = (
            ticket.token == prepared.token
            and ticket.action == prepared.action
            and ticket.inputs == prepared.inputs
        )
        if not same_action:
            return f"Ticket {ticket_id} is for a different action."
        if ticket.state == USED:
            return f"Ticket {ticket_id} was already used."
        if ticket.state == CANCELLED:
            return "The caller stopped this action. Do not do it. Ask what they want now."
        if now >= ticket.expires_at:
            return "That yes is too old. Read the summary again and ask for a fresh yes."
        return None

    async def mark_used(self, tx: Tx, ticket: str, now: float) -> None:
        await tx.execute(
            "UPDATE tickets SET state = ?, closed_at = ? WHERE ticket = ?", (USED, now, ticket)
        )

    async def cancel(self, tx: Tx, ticket: str, reason: str, now: float) -> bool:
        """Cancel one open ticket. False if it was not open (for example already used)."""
        touched = await tx.execute(
            "UPDATE tickets SET state = ?, closed_at = ?, note = ? WHERE ticket = ? AND state = ?",
            (CANCELLED, now, reason, ticket, OPEN),
        )
        return touched == 1

    async def cancel_all_open(self, tx: Tx, reason: str, now: float) -> int:
        return await tx.execute(
            "UPDATE tickets SET state = ?, closed_at = ?, note = ? WHERE state = ?",
            (CANCELLED, now, reason, OPEN),
        )


def _from_row(row: Row) -> Ticket:
    return Ticket(
        ticket=row["ticket"],
        token=row["token"],
        action=row["action"],
        inputs=json.loads(row["inputs_json"]),
        state=row["state"],
        issued_at=row["issued_at"],
        expires_at=row["expires_at"],
    )
