"""Food-delivery business rules, as small pure functions.

The rules live in code, not in the AI's instructions: money rules must never
depend on whether a model decides to follow them. Each rule returns a Check,
whose reason is written so the agent can read it to the caller.
"""

from __future__ import annotations

from engine.contract.action import Check
from packs.food.money import format_rupees
from packs.food.order_store import Order

PLACED = "placed"
PREPARING = "preparing"
OUT_FOR_DELIVERY = "out_for_delivery"
DELIVERED = "delivered"
CANCELLED = "cancelled"

# The only path an order walks, always forward.
FORWARD_PATH = (PLACED, PREPARING, OUT_FOR_DELIVERY, DELIVERED)
BEFORE_DISPATCH = (PLACED, PREPARING)
PREPAID = ("upi", "card")


def name_matches(name_on_order: str, said: str) -> bool:
    """Decision 2: the caller gives the order number + the name on the order.

    A first name alone is accepted, because callers often say only that.
    Case and extra spaces are ignored, because speech-to-text varies them.
    """
    on_order = " ".join(name_on_order.lower().split())
    said = " ".join(said.lower().split())
    return said == on_order or said == on_order.split()[0]


def can_cancel(order: Order) -> Check:
    n = order.order_no
    if order.status == CANCELLED:
        return Check.refuse(f"Order {n} is already cancelled.")
    if order.status == OUT_FOR_DELIVERY:
        return Check.refuse(f"Order {n} is already out for delivery, so it can't be cancelled.")
    if order.status == DELIVERED:
        return Check.refuse(f"Order {n} was already delivered, so it can't be cancelled.")
    return Check.ok()


def can_change_address(order: Order, new_address: str) -> Check:
    if order.status not in BEFORE_DISPATCH:
        return Check.refuse(
            f"Order {order.order_no} is {order.status.replace('_', ' ')}, so the address can't be changed now."
        )
    if " ".join(new_address.lower().split()) == " ".join(order.address.lower().split()):
        return Check.refuse(f"That is already the address of order {order.order_no}.")
    return Check.ok()


def refundable_left(order: Order, refunded: int) -> int:
    """How much can still go back. A cash order is paid at the door, so the
    agent can never send money back for it."""
    if order.payment_method not in PREPAID:
        return 0
    return order.amount_paise - refunded


def can_refund(order: Order, refunded: int, amount: int, to_method: str | None) -> Check:
    """Refund for a problem with a delivered order (decision 3: cancelling
    refunds by itself; this is only for things like a missing item)."""
    n = order.order_no
    if order.status != DELIVERED:
        return Check.refuse(f"Item refunds are only for delivered orders. Order {n} is {order.status.replace('_', ' ')}.")
    if order.payment_method not in PREPAID:
        return Check.refuse(f"Order {n} was paid in cash, so a refund can't be sent by phone. Hand the call to a human.")
    if to_method and to_method != order.payment_method:
        return Check.refuse(f"Refunds only go back to the original payment method ({order.payment_method}).")
    if amount <= 0:
        return Check.refuse("The refund amount must be more than zero.")
    left = refundable_left(order, refunded)
    if amount > left:
        return Check.refuse(f"That is more than what is left to refund on order {n} ({format_rupees(left)}).")
    return Check.ok()


def can_move(status_now: str, new_status: str) -> Check:
    """Timeline rule: orders only move forward, and a cancelled order never moves.

    This is what stops "out for delivery" from happening to an order the
    caller already cancelled.
    """
    if status_now == CANCELLED:
        return Check.refuse("cancelled orders never move")
    if new_status not in FORWARD_PATH:
        return Check.refuse(f"unknown status {new_status!r}")
    if FORWARD_PATH.index(new_status) <= FORWARD_PATH.index(status_now):
        return Check.refuse(f"orders only move forward ({status_now} -> {new_status})")
    return Check.ok()
