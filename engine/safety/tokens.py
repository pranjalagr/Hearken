"""Tokens: one request turns into at most one action.

`prepare` saves the exact action as "waiting" and hands back a token.
`confirm` with that token does the action once and marks it "done".
A second confirm with the same token finds "done" and does nothing again.

Tokens are made here, never by the AI. An AI might invent a fresh token on a
retry, and the backend would then think it was a new request.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from engine.contract.action import Action
from engine.storage.database import Row, Tx

WAITING = "waiting"
DONE = "done"


@dataclass(frozen=True)
class PreparedAction:
    token: str
    action: str
    inputs: dict[str, Any]
    request_key: str
    summary: str
    state: str
    created_at: float
    expires_at: float
    reply: dict[str, Any] | None  # saved when done, sent again on "already done"

    def expired(self, now: float) -> bool:
        return now >= self.expires_at


def request_key(action: Action, inputs: dict[str, Any]) -> str:
    """Decision 14: the same action on the same thing, in the same call, is the
    same request, so prepare hands back the earlier token instead of a new one."""
    names = action.same_request_inputs or tuple(sorted(inputs))
    picked = {name: _comparable(inputs.get(name)) for name in names}
    return action.name + ":" + json.dumps(picked, sort_keys=True, ensure_ascii=False)


def _comparable(value: Any) -> Any:
    # 120 and 120.0 are the same amount, and "Flat 5" and "flat  5" the same
    # address. Without this they would look like two different requests.
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        return " ".join(value.lower().split())
    return value


class TokenBook:
    async def create(
        self,
        tx: Tx,
        action: str,
        inputs: dict[str, Any],
        key: str,
        summary: str,
        now: float,
        expires_at: float,
    ) -> PreparedAction:
        # Rows are never deleted, so counting them gives a new unique number.
        # Short, ordered tokens (T-1, T-2) are easy for an AI to copy back.
        count = await tx.fetch_one("SELECT COUNT(*) AS n FROM prepared_actions")
        token = f"T-{count['n'] + 1}"
        await tx.execute(
            "INSERT INTO prepared_actions"
            " (token, action, inputs_json, request_key, summary, state, created_at, expires_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (token, action, _to_json(inputs), key, summary, WAITING, now, expires_at),
        )
        return PreparedAction(token, action, inputs, key, summary, WAITING, now, expires_at, None)

    async def get(self, tx: Tx, token: str) -> PreparedAction | None:
        row = await tx.fetch_one("SELECT * FROM prepared_actions WHERE token = ?", (token,))
        return _from_row(row) if row else None

    async def done_for_request(self, tx: Tx, key: str) -> PreparedAction | None:
        row = await tx.fetch_one(
            "SELECT * FROM prepared_actions WHERE request_key = ? AND state = ? LIMIT 1",
            (key, DONE),
        )
        return _from_row(row) if row else None

    async def waiting_for_request(self, tx: Tx, key: str, now: float) -> PreparedAction | None:
        row = await tx.fetch_one(
            # rowid = insertion order, so the newest token wins even within one second.
            "SELECT * FROM prepared_actions WHERE request_key = ? AND state = ? AND expires_at > ?"
            " ORDER BY rowid DESC LIMIT 1",
            (key, WAITING, now),
        )
        return _from_row(row) if row else None

    async def mark_done(self, tx: Tx, token: str, reply: dict[str, Any], now: float) -> None:
        await tx.execute(
            "UPDATE prepared_actions SET state = ?, done_at = ?, reply_json = ? WHERE token = ?",
            (DONE, now, _to_json(reply), token),
        )


def _to_json(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _from_row(row: Row) -> PreparedAction:
    return PreparedAction(
        token=row["token"],
        action=row["action"],
        inputs=json.loads(row["inputs_json"]),
        request_key=row["request_key"],
        summary=row["summary"],
        state=row["state"],
        created_at=row["created_at"],
        expires_at=row["expires_at"],
        reply=json.loads(row["reply_json"]) if row["reply_json"] else None,
    )
