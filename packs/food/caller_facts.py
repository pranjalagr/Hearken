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
    # Part 5, L1: tasks 10, 11 and 13 need a caller who isn't the owner, or
    # who has two orders. Named facts, checked like the rest, instead of
    # switching the check off for those tasks.
    "order_owner": "the name on the order, when the caller is someone else (a test of the name check)",
    "other_order_no": "a second order of the same customer, also part of this call",
    "new_address_order": "the order new_address is for, when it isn't order_no",
}


async def caller_fact_problems(tx: Tx, task: Task) -> list[str]:
    caller = task.caller
    facts: dict[str, Any] = caller.facts
    problems = [f"unknown fact {name!r}" for name in facts if name not in KNOWN_FACTS]
    order_no = facts.get("order_no")
    if not _whole(order_no):
        return problems + ["facts.order_no must be a whole number"]
    if order_no not in task.uses:
        problems.append(f"order {order_no} is not in this task's uses {task.uses}")
    order = await store.load_order(tx, order_no)
    if order is None:
        return problems + [f"order {order_no} is not in the seed data"]
    problems += _owner_problems(order_no, order.customer_name, caller.name, facts)
    if "paid_by" in facts and facts["paid_by"] != order.payment_method:
        problems.append(f"order {order_no} was paid by {order.payment_method}, not {facts['paid_by']}")
    if "address" in facts and facts["address"] != order.address:
        problems.append(f"order {order_no}'s address is {order.address!r}")
    if "amount" in facts and rupees_to_paise(facts["amount"]) != order.amount_paise:
        problems.append(f"order {order_no} cost {order.amount_paise / 100:g} rupees, not {facts['amount']}")
    other = None
    if "other_order_no" in facts:
        other, other_problems = await _other_order(tx, task, order_no, order.customer_name, facts["other_order_no"])
        problems += other_problems
    problems += _new_address_problems(facts, order_no, order.address, other)
    problems += await _missing_item_problems(tx, order_no, facts)
    return problems


def _owner_problems(order_no: int, owner: str, caller_name: str, facts: dict[str, Any]) -> list[str]:
    """The caller is the order's owner, unless the task says on purpose that
    someone else is calling (order_owner), to test the robot's name check."""
    if "order_owner" not in facts:
        if caller_name != owner:
            return [f"order {order_no} belongs to {owner!r}, not {caller_name!r}"]
        return []
    if facts["order_owner"] != owner:
        return [f"order {order_no} belongs to {owner!r}, not {facts['order_owner']!r}"]
    if caller_name == owner:
        return ["order_owner is only for a caller who isn't the owner"]
    return []


async def _other_order(tx: Tx, task: Task, order_no: int, owner: str, other_no: Any):
    if not _whole(other_no):
        return None, ["facts.other_order_no must be a whole number"]
    if other_no == order_no:
        return None, ["other_order_no must be a different order"]
    problems = [] if other_no in task.uses else [f"order {other_no} is not in this task's uses {task.uses}"]
    other = await store.load_order(tx, other_no)
    if other is None:
        return None, problems + [f"order {other_no} is not in the seed data"]
    if other.customer_name != owner:
        problems.append(f"order {other_no} belongs to {other.customer_name!r}, not {owner!r}")
    return other, problems


def _new_address_problems(facts: dict[str, Any], order_no: int, address: str, other) -> list[str]:
    target = facts.get("new_address_order", order_no)
    if "new_address_order" in facts:
        if "new_address" not in facts:
            return ["new_address_order needs new_address"]
        if target not in (order_no, facts.get("other_order_no")):
            return ["new_address_order must be order_no or other_order_no"]
    if "new_address" not in facts:
        return []
    current = address if target == order_no else other.address if other is not None else None
    if current is not None and _same_text(facts["new_address"], current):
        return [f"new_address is the address order {target} already has"]
    return []


def _whole(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


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
        # Some or all of the units can be missing (task 8: one of two thalis),
        # so the refund is a whole number of units, never a made-up amount.
        price, quantity = item["price_paise"], item["quantity"]
        asked = rupees_to_paise(facts["refund_amount"])
        if asked is None or asked % price or not 1 <= asked // price <= quantity:
            units = ", ".join(f"{n * price / 100:g}" for n in range(1, quantity + 1))
            return [f"{facts['missing_item']} refunds can be {units} rupees, not {facts['refund_amount']}"]
    return []


def _same_text(a: Any, b: str) -> bool:
    return isinstance(a, str) and " ".join(a.lower().split()) == " ".join(b.lower().split())
