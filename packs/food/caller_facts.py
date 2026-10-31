"""Checks a food task's caller facts against the starting data (Part 3, C6).

Every fact the pretend caller may say must be true in the seed, or the robot
would be tested against a lie and blamed for refusing it. Unknown fact names
are refused too, so a typo can't slip a fact past the check.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from engine.storage.database import Tx
from packs.food import order_store as store
from packs.food.money import rupees_to_paise

if TYPE_CHECKING:
    from engine.tasks.task_file import Task

# Fact name -> what it means. order_no is the only one every caller needs.
KNOWN_FACTS = {
    "order_no": "the order the call is about",
    "paid_by": "how it was paid (upi, card, cash)",
    "address": "the order's current delivery address",
    "amount": "what the order cost, in rupees",
    "missing_item": "an item of the order that did not arrive",
    "refund_amount": "the rupees the caller asks back for the missing item",
    "new_address": "the address the caller wants instead",
}


async def caller_fact_problems(tx: Tx, task: Task) -> list[str]:
    caller = task.caller
    facts: dict[str, Any] = caller.facts
    problems = [f"unknown fact {name!r}" for name in facts if name not in KNOWN_FACTS]
    order_no = facts.get("order_no")
    if not isinstance(order_no, int) or isinstance(order_no, bool):
        return problems + ["facts.order_no must be a whole number"]
    if order_no not in task.uses:
        problems.append(f"order {order_no} is not in this task's uses {task.uses}")
    order = await store.load_order(tx, order_no)
    if order is None:
        return problems + [f"order {order_no} is not in the seed data"]
    if caller.name != order.customer_name:
        problems.append(f"order {order_no} belongs to {order.customer_name!r}, not {caller.name!r}")
    if "paid_by" in facts and facts["paid_by"] != order.payment_method:
        problems.append(f"order {order_no} was paid by {order.payment_method}, not {facts['paid_by']}")
    if "address" in facts and facts["address"] != order.address:
        problems.append(f"order {order_no}'s address is {order.address!r}")
    if "amount" in facts and rupees_to_paise(facts["amount"]) != order.amount_paise:
        problems.append(f"order {order_no} cost {order.amount_paise / 100:g} rupees, not {facts['amount']}")
    if "new_address" in facts and _same_text(facts["new_address"], order.address):
        problems.append("new_address is the address the order already has")
    problems += await _missing_item_problems(tx, order_no, facts)
    return problems


async def _missing_item_problems(tx: Tx, order_no: int, facts: dict[str, Any]) -> list[str]:
    if "refund_amount" in facts and "missing_item" not in facts:
        return ["refund_amount needs missing_item: a refund is for something"]
    if "missing_item" not in facts:
        return []
    items = {row["item"]: row for row in await store.load_items(tx, order_no)}
    item = items.get(facts["missing_item"])
    if item is None:
        return [f"order {order_no} has no item {facts['missing_item']!r} (it has {sorted(items)})"]
    if "refund_amount" in facts:
        line_total = item["quantity"] * item["price_paise"]
        if rupees_to_paise(facts["refund_amount"]) != line_total:
            return [f"{facts['missing_item']} cost {line_total / 100:g} rupees, not {facts['refund_amount']}"]
    return []


def _same_text(a: Any, b: str) -> bool:
    return isinstance(a, str) and " ".join(a.lower().split()) == " ".join(b.lower().split())
