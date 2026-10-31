"""All SQL of the food pack lives here, so rules and actions read like plain logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from engine.storage.database import Tx


@dataclass(frozen=True)
class Order:
    order_no: int
    customer_name: str
    amount_paise: int
    payment_method: str
    address: str
    status: str


async def load_order(tx: Tx, order_no: int) -> Order | None:
    row = await tx.fetch_one(
        "SELECT o.order_no, c.name AS customer_name, o.amount_paise, o.payment_method,"
        " o.address, o.status"
        " FROM orders o JOIN customers c ON c.customer_id = o.customer_id"
        " WHERE o.order_no = ?",
        (order_no,),
    )
    return Order(**row) if row else None


async def load_items(tx: Tx, order_no: int) -> list[dict[str, Any]]:
    return await tx.fetch_all(
        "SELECT item, quantity, price_paise FROM order_items WHERE order_no = ? ORDER BY item",
        (order_no,),
    )


async def refunded_so_far(tx: Tx, order_no: int) -> int:
    row = await tx.fetch_one(
        "SELECT COALESCE(SUM(amount_paise), 0) AS total FROM refunds WHERE order_no = ?", (order_no,)
    )
    return row["total"]


async def set_status(tx: Tx, order_no: int, status: str) -> None:
    await tx.execute("UPDATE orders SET status = ? WHERE order_no = ?", (status, order_no))


async def set_address(tx: Tx, order_no: int, address: str) -> None:
    await tx.execute("UPDATE orders SET address = ? WHERE order_no = ?", (address, order_no))


async def add_refund(
    tx: Tx, order_no: int, amount_paise: int, method: str, reason: str, token: str | None, at: float
) -> None:
    await tx.execute(
        "INSERT INTO refunds (order_no, amount_paise, method, reason, token, at) VALUES (?, ?, ?, ?, ?, ?)",
        (order_no, amount_paise, method, reason, token, at),
    )


async def add_handoff(tx: Tx, order_no: int | None, reason: str, at: float) -> None:
    await tx.execute(
        "INSERT INTO handoffs (order_no, reason, at) VALUES (?, ?, ?)", (order_no, reason, at)
    )
