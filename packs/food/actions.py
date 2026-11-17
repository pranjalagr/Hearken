"""The five food-delivery actions, each filling in the engine's Action shape.

Each action only says WHAT is allowed (check), WHAT to tell the caller
(summarize for logs, spoken_question for the caller's ear) and WHAT changes
(do). WHEN and HOW SAFELY it runs (tokens, tickets, the transaction, the log)
is the engine's job.
"""

from __future__ import annotations

from typing import Any

from engine.contract.action import (
    INTEGER,
    STRING,
    Action,
    ActionContext,
    Check,
    Input,
    Phrase,
    SpokenQuestion,
)
from engine.storage.database import Tx
from packs.food import order_store as store
from packs.food import rules
from packs.food.money import format_rupees, spoken_amount

ORDER_NO = Input(INTEGER, "The 4-digit order number.")


async def identify(tx: Tx, inputs: dict[str, Any]) -> tuple[store.Order | None, Check]:
    """Find the order. The order number alone identifies it (decision 2, changed 17 Nov 2026:
    no name is asked or checked). A miss names the wrong input, so an agent can ask again
    without reading the words (decision 27)."""
    order = await store.load_order(tx, inputs["order_no"])
    if order is None:
        return None, Check.refuse(f"No order {inputs['order_no']} found. Ask the caller to repeat the number.",
                                 wrong_input="order_no")
    return order, Check.ok()


async def _item_refund(tx: Tx, inputs: dict[str, Any]) -> tuple[store.Order, dict[str, Any] | None, int, Check]:
    """The order, its line for the item, and the refund in paise: price times how many.

    Decided 17 Nov 2026: the agent names the item from the order; code fills its price, so
    an AI never writes an amount (on 16 Nov 2026 a Thinker made up Rs 50 in food_03). One
    place for this sum, so the log, the yes-question and the refund always agree."""
    order, verdict = await identify(tx, inputs)
    if order is None:
        return order, None, 0, verdict
    wanted = " ".join(str(inputs["item"]).lower().split())
    line = next((it for it in await store.load_items(tx, order.order_no)
                 if " ".join(it["item"].lower().split()) == wanted), None)
    if line is None:
        return order, None, 0, Check.refuse(f"Order {order.order_no} has no item {inputs['item']!r}.",
                                            wrong_input="item")
    quantity = inputs.get("quantity") or line["quantity"]  # left out: all of that item
    if not 0 < quantity <= line["quantity"]:
        return order, line, 0, Check.refuse(
            f"Order {order.order_no} has {line['quantity']} x {line['item']}.", wrong_input="quantity")
    return order, line, line["price_paise"] * quantity, Check.ok()


async def _cancel_refund(tx: Tx, inputs: dict[str, Any]) -> tuple[store.Order, int]:
    """The order and how much cancelling it sends back, in paise.

    One place for this sum, because three things must agree on it: the log
    summary, the yes-question the caller hears, and the refund actually made.
    """
    order = await store.load_order(tx, inputs["order_no"])
    return order, rules.refundable_left(order, await store.refunded_so_far(tx, order.order_no))


class FindOrder(Action):
    name = "find_order"
    description = "Look up an order: status, items, amount, payment method and address."
    inputs = {"order_no": ORDER_NO}

    async def check(self, tx: Tx, inputs: dict[str, Any]) -> Check:
        _, verdict = await identify(tx, inputs)
        return verdict

    async def do(self, tx: Tx, inputs: dict[str, Any], ctx: ActionContext) -> dict[str, Any]:
        order = await store.load_order(tx, inputs["order_no"])
        items = await store.load_items(tx, order.order_no)
        refunded = await store.refunded_so_far(tx, order.order_no)
        # Only what the agent needs to answer: a short reply is faster and cheaper to read.
        return {
            "order_no": order.order_no,
            "status": order.status,
            "items": [f"{item['quantity']} x {item['item']}" for item in items],
            "amount": format_rupees(order.amount_paise),
            "payment": order.payment_method,
            "address": order.address,
            "refunded": format_rupees(refunded),
        }


class ChangeAddress(Action):
    name = "change_address"
    description = "Change the delivery address of an order that has not left yet."
    risky = True
    inputs = {
        "order_no": ORDER_NO,
        "new_address": Input(STRING, "The full new delivery address."),
    }
    same_request_inputs = ("order_no", "new_address")

    async def check(self, tx: Tx, inputs: dict[str, Any]) -> Check:
        order, verdict = await identify(tx, inputs)
        if order is None:
            return verdict
        return rules.can_change_address(order, inputs["new_address"])

    async def summarize(self, tx: Tx, inputs: dict[str, Any]) -> str:
        return f"change the delivery address of order {inputs['order_no']} to: {inputs['new_address']}"

    async def spoken_question(self, tx: Tx, inputs: dict[str, Any]) -> SpokenQuestion:
        # The new address is read back exactly as the caller gave it, so a
        # misheard street name gets caught before anything changes.
        return SpokenQuestion(
            "change_address",
            values={"order_no": str(inputs["order_no"]), "new_address": inputs["new_address"]},
        )

    async def do(self, tx: Tx, inputs: dict[str, Any], ctx: ActionContext) -> dict[str, Any]:
        await store.set_address(tx, inputs["order_no"], inputs["new_address"])
        return {"order_no": inputs["order_no"], "address": inputs["new_address"]}


class CancelOrder(Action):
    name = "cancel_order"
    description = "Cancel an order that has not left yet. A prepaid order is refunded automatically."
    risky = True
    inputs = {"order_no": ORDER_NO}
    same_request_inputs = ("order_no",)

    async def check(self, tx: Tx, inputs: dict[str, Any]) -> Check:
        order, verdict = await identify(tx, inputs)
        if order is None:
            return verdict
        return rules.can_cancel(order)

    async def summarize(self, tx: Tx, inputs: dict[str, Any]) -> str:
        order, refund = await _cancel_refund(tx, inputs)
        if refund > 0:
            return f"cancel order {order.order_no} and refund {format_rupees(refund)} to {order.payment_method}"
        return f"cancel order {order.order_no} (cash on delivery, so nothing to refund)"

    async def spoken_question(self, tx: Tx, inputs: dict[str, Any]) -> SpokenQuestion:
        # The same numbers as the summary, from the same helper, so the line
        # the caller hears and the line in the log can never disagree.
        order, refund = await _cancel_refund(tx, inputs)
        if refund > 0:
            return SpokenQuestion(
                "cancel_with_refund",
                values={"order_no": str(order.order_no)},
                phrases={"amount": spoken_amount(refund), "payment": Phrase("payment_method", order.payment_method)},
            )
        return SpokenQuestion("cancel_no_refund", values={"order_no": str(order.order_no)})

    async def do(self, tx: Tx, inputs: dict[str, Any], ctx: ActionContext) -> dict[str, Any]:
        order, refund = await _cancel_refund(tx, inputs)
        await store.set_status(tx, order.order_no, rules.CANCELLED)
        # Decision 3: one yes covers both the cancel and its refund.
        if refund > 0:
            await store.add_refund(
                tx, order.order_no, refund, order.payment_method, "order cancelled", ctx.token, ctx.now
            )
        return {
            "order_no": order.order_no,
            "status": rules.CANCELLED,
            "refund": format_rupees(refund) if refund else "none",
        }


class RefundOrder(Action):
    name = "refund_order"
    description = "Refund part of a delivered order for a problem such as a missing item."
    risky = True
    inputs = {
        "order_no": ORDER_NO,
        "item": Input(STRING, "The item that went wrong, exactly as named in the order."),
        "quantity": Input(INTEGER, "How many of that item, if the caller says; leave out for all of them.",
                          required=False),
        "reason": Input(STRING, "What went wrong.", choices=("missing_item", "wrong_item", "damaged_item")),
        "to_method": Input(
            STRING,
            "Only if the caller asks for a specific method. Refunds go to the original method.",
            required=False,
            choices=("upi", "card", "cash"),
        ),
    }
    # Not the reason: if a retry changed the reason, the same money must still
    # not go out twice. Two refunds for one item on one order go to a human instead.
    same_request_inputs = ("order_no", "item")

    async def check(self, tx: Tx, inputs: dict[str, Any]) -> Check:
        order, _, amount, verdict = await _item_refund(tx, inputs)
        if not verdict.allowed:
            return verdict
        refunded = await store.refunded_so_far(tx, order.order_no)
        return rules.can_refund(order, refunded, amount, inputs.get("to_method"))

    async def summarize(self, tx: Tx, inputs: dict[str, Any]) -> str:
        order, _, amount, _ = await _item_refund(tx, inputs)
        problem = inputs["reason"].replace("_", " ")
        return f"refund {format_rupees(amount)} on order {order.order_no} to {order.payment_method} for a {problem}"

    async def spoken_question(self, tx: Tx, inputs: dict[str, Any]) -> SpokenQuestion:
        order, _, amount, _ = await _item_refund(tx, inputs)
        return SpokenQuestion(
            "refund",
            values={"order_no": str(order.order_no)},
            # The reason is said back too, so a refund asked for the wrong
            # problem ("wrong item" vs "missing item") gets caught by the caller.
            phrases={
                "amount": spoken_amount(amount),
                "payment": Phrase("payment_method", order.payment_method),
                "reason": Phrase("refund_reason", inputs["reason"]),
            },
        )

    async def do(self, tx: Tx, inputs: dict[str, Any], ctx: ActionContext) -> dict[str, Any]:
        order, _, amount, _ = await _item_refund(tx, inputs)
        await store.add_refund(
            tx, order.order_no, amount, order.payment_method, inputs["reason"], ctx.token, ctx.now
        )
        return {"order_no": order.order_no, "refunded": format_rupees(amount), "to": order.payment_method}


class HandOverToHuman(Action):
    name = "hand_over_to_human"
    description = "Hand the call to a human agent when you cannot help."
    inputs = {
        "reason": Input(STRING, "Why the call needs a human, in one short sentence."),
        "order_no": Input(INTEGER, "The order it is about, if any.", required=False),
    }

    async def check(self, tx: Tx, inputs: dict[str, Any]) -> Check:
        # The escape hatch is never refused: a caller must always be able to reach a person.
        return Check.ok()

    async def do(self, tx: Tx, inputs: dict[str, Any], ctx: ActionContext) -> dict[str, Any]:
        order_no = inputs.get("order_no")
        if order_no is not None and await store.load_order(tx, order_no) is None:
            order_no = None  # keep the handoff even if the number was misheard
        await store.add_handoff(tx, order_no, inputs["reason"], ctx.now)
        return {"handed_over": True}
