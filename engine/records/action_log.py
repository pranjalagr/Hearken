"""The action log: one row per tool call.

Each row is written inside the same transaction as the action itself, so an
action and its log row are saved together or not at all. A "refund happened
but no log row" state can never exist to fool the scorer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from engine.storage.database import Tx


@dataclass(frozen=True)
class LogEntry:
    at: float
    tool: str  # what the AI called, e.g. "confirm"
    action: str | None  # the business action behind it, e.g. "cancel_order"
    inputs: dict[str, Any] = field(default_factory=dict)
    outcome: str = ""  # ok / refused / already_done / expired / error
    acted: bool = False  # True only when the action's change really happened
    reason: str = ""
    token: str | None = None
    ticket: str | None = None
    switch_note: str = ""


class ActionLog:
    async def write(self, tx: Tx, entry: LogEntry) -> None:
        await tx.execute(
            "INSERT INTO action_log"
            " (at, tool, action, inputs_json, outcome, acted, reason, token, ticket, switch_note)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                entry.at,
                entry.tool,
                entry.action,
                json.dumps(entry.inputs, sort_keys=True, ensure_ascii=False),
                entry.outcome,
                int(entry.acted),
                entry.reason,
                entry.token,
                entry.ticket,
                entry.switch_note,
            ),
        )
