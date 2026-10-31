"""Toy pack: moving money between two boxes. Kept in one small file on purpose."""

from __future__ import annotations

from typing import Any

from engine.contract.action import INTEGER, STRING, Action, ActionContext, Check, Input
from engine.contract.pack import Pack
from engine.storage.database import Tx

SCHEMA = """
CREATE TABLE boxes (
    name          TEXT PRIMARY KEY,
    balance_paise INTEGER NOT NULL CHECK (balance_paise >= 0),
    locked        INTEGER NOT NULL DEFAULT 0 CHECK (locked IN (0, 1))
) STRICT;

CREATE TABLE moves (
    move_id      INTEGER PRIMARY KEY,
    from_box     TEXT NOT NULL REFERENCES boxes (name),
    to_box       TEXT NOT NULL REFERENCES boxes (name),
    amount_paise INTEGER NOT NULL CHECK (amount_paise > 0),
    token        TEXT,
    at           REAL NOT NULL
) STRICT;
"""

START_BALANCES = {"a": 10_000, "b": 0}


class ReadBoxes(Action):
    name = "read_boxes"
    description = "Show how much money is in each box."
    inputs = {}

    async def check(self, tx: Tx, inputs: dict[str, Any]) -> Check:
        return Check.ok()

    async def do(self, tx: Tx, inputs: dict[str, Any], ctx: ActionContext) -> dict[str, Any]:
        rows = await tx.fetch_all("SELECT name, balance_paise FROM boxes ORDER BY name")
        return {row["name"]: row["balance_paise"] for row in rows}


class MoveMoney(Action):
    name = "move_money"
    description = "Move money from one box to another."
    risky = True
    inputs = {
        "from_box": Input(STRING, "Box to take money from."),
        "to_box": Input(STRING, "Box to put money in."),
        "amount_paise": Input(INTEGER, "How much to move, in paise."),
    }

    async def check(self, tx: Tx, inputs: dict[str, Any]) -> Check:
        source = await tx.fetch_one("SELECT * FROM boxes WHERE name = ?", (inputs["from_box"],))
        target = await tx.fetch_one("SELECT * FROM boxes WHERE name = ?", (inputs["to_box"],))
        if source is None or target is None:
            return Check.refuse("No such box.")
        if source["name"] == target["name"]:
            return Check.refuse("Pick two different boxes.")
        if source["locked"] or target["locked"]:
            return Check.refuse("A box is locked.")
        if inputs["amount_paise"] <= 0:
            return Check.refuse("The amount must be more than zero.")
        if source["balance_paise"] < inputs["amount_paise"]:
            return Check.refuse("Not enough money in the box.")
        return Check.ok()

    async def summarize(self, tx: Tx, inputs: dict[str, Any]) -> str:
        return f"move {inputs['amount_paise']} paise from box {inputs['from_box']} to box {inputs['to_box']}"

    async def do(self, tx: Tx, inputs: dict[str, Any], ctx: ActionContext) -> dict[str, Any]:
        amount = inputs["amount_paise"]
        await tx.execute("UPDATE boxes SET balance_paise = balance_paise - ? WHERE name = ?", (amount, inputs["from_box"]))
        await tx.execute("UPDATE boxes SET balance_paise = balance_paise + ? WHERE name = ?", (amount, inputs["to_box"]))
        await tx.execute(
            "INSERT INTO moves (from_box, to_box, amount_paise, token, at) VALUES (?, ?, ?, ?, ?)",
            (inputs["from_box"], inputs["to_box"], amount, ctx.token, ctx.now),
        )
        return {"moved": amount}


class ToyPack(Pack):
    name = "toy"

    def actions(self) -> list[Action]:
        return [ReadBoxes(), MoveMoney()]

    def schema_sql(self) -> str:
        return SCHEMA

    async def load_seed(self, tx: Tx) -> None:
        for name, balance in START_BALANCES.items():
            await tx.execute("INSERT INTO boxes (name, balance_paise) VALUES (?, ?)", (name, balance))

    async def apply_change(self, tx: Tx, target: str, change: str) -> Check:
        """Timeline change: lock or unlock a box."""
        if change not in ("lock", "unlock"):
            return Check.refuse(f"unknown change {change!r}")
        touched = await tx.execute(
            "UPDATE boxes SET locked = ? WHERE name = ?", (int(change == "lock"), target)
        )
        return Check.ok() if touched == 1 else Check.refuse(f"no box {target!r}")
