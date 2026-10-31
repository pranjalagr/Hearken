"""The Guard's private door into one call's database (decision 7).

This is NOT an AI tool. It never appears in the tool catalog or the MCP
server. Only the Guard's own code (plain code, built in Part 2) holds a
GuardDoor, so the AI has no way to approve its own actions.
"""

from __future__ import annotations

from engine.lab_controls.clock import Clock
from engine.safety.lifetimes import Lifetimes
from engine.safety.tickets import TicketBook
from engine.safety.tokens import WAITING, TokenBook
from engine.storage.database import Database


class TicketRefused(Exception):
    """The Guard asked for a ticket that must not exist."""


class GuardDoor:
    def __init__(self, db: Database, clock: Clock, lifetimes: Lifetimes = Lifetimes()) -> None:
        self._db = db
        self._clock = clock
        self._lifetimes = lifetimes
        self._tokens = TokenBook()
        self._tickets = TicketBook()

    async def issue_ticket(self, token: str) -> str:
        """The caller said a real yes to this token's summary: write a yes-ticket."""
        async with self._db.transaction() as tx:
            now = self._clock.now()
            prepared = await self._tokens.get(tx, token)
            if prepared is None:
                raise TicketRefused(f"unknown token {token}")
            if prepared.state != WAITING:
                raise TicketRefused(f"token {token} is already {prepared.state}")
            if prepared.expired(now):
                raise TicketRefused(f"token {token} has expired")
            ticket = await self._tickets.create(tx, prepared, now, now + self._lifetimes.ticket_s)
        return ticket.ticket

    async def cancel_ticket(self, ticket: str, reason: str) -> bool:
        """The caller said "ruko": stop this ticket at once.

        Returns False when it is too late (the ticket was already used). The
        agent must then tell the caller the truth, not promise to stop it.
        """
        async with self._db.transaction() as tx:
            return await self._tickets.cancel(tx, ticket, reason, self._clock.now())

    async def cancel_all_open(self, reason: str) -> int:
        """Cancel every open ticket in this call; returns how many."""
        async with self._db.transaction() as tx:
            return await self._tickets.cancel_all_open(tx, reason, self._clock.now())
