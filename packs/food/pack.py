"""The food-delivery pack: ties its tables, seed data, actions and timeline rule together."""

from __future__ import annotations

import json
from pathlib import Path

from engine.contract.action import Action, Check
from engine.contract.pack import Pack
from engine.storage.database import Tx
from packs.food import order_store as store
from packs.food import rules
from packs.food.actions import CancelOrder, ChangeAddress, FindOrder, HandOverToHuman, RefundOrder

HERE = Path(__file__).parent


class FoodPack(Pack):
    name = "food"

    def actions(self) -> list[Action]:
        return [FindOrder(), ChangeAddress(), CancelOrder(), RefundOrder(), HandOverToHuman()]

    def schema_sql(self) -> str:
        return (HERE / "schema.sql").read_text(encoding="utf-8")

    async def load_seed(self, tx: Tx) -> None:
        seed = json.loads((HERE / "seed.json").read_text(encoding="utf-8"))
        for customer in seed["customers"]:
            await tx.execute(
                "INSERT INTO customers (customer_id, name) VALUES (?, ?)",
                (customer["customer_id"], customer["name"]),
            )
        for order in seed["orders"]:
            # The amount is worked out from the items, so the two can never disagree.
            amount = sum(item["quantity"] * item["price_paise"] for item in order["items"])
            await tx.execute(
                "INSERT INTO orders (order_no, customer_id, amount_paise, payment_method, address, status)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (order["order_no"], order["customer_id"], amount, order["payment_method"],
                 order["address"], order["status"]),
            )
            for item in order["items"]:
                await tx.execute(
                    "INSERT INTO order_items (order_no, item, quantity, price_paise) VALUES (?, ?, ?, ?)",
                    (order["order_no"], item["item"], item["quantity"], item["price_paise"]),
                )
        for refund in seed["refunds"]:
            await store.add_refund(
                tx, refund["order_no"], refund["amount_paise"], refund["method"],
                refund["reason"], None, refund["at"],
            )

    async def apply_change(self, tx: Tx, target: str, change: str) -> Check:
        """Timeline change: `target` is an order number, `change` its new status."""
        if not target.isdigit():
            return Check.refuse(f"not an order number: {target!r}")
        order = await store.load_order(tx, int(target))
        if order is None:
            return Check.refuse(f"no order {target}")
        verdict = rules.can_move(order.status, change)
        if verdict.allowed:
            await store.set_status(tx, order.order_no, change)
        return verdict
