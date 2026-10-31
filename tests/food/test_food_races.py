"""Food races: the order moves while the caller is being asked, and replies get lost."""

from engine.lab_controls.switches import Switch
from engine.lab_controls.timeline import PlannedChange
from packs.food.pack import FoodPack

AMAN = {"order_no": 4530, "customer_name": "Aman"}
PRIYA_REFUND = {"order_no": 4512, "customer_name": "Priya", "amount_rupees": 40, "reason": "missing_item"}


async def test_order_leaves_while_the_caller_is_asked_so_cancel_is_refused(open_call):
    rider = PlannedChange("4530", "out_for_delivery", after_event="prepared:cancel_order")
    kit = await open_call(FoodPack(), timeline=(rider,))
    prepared = await kit.tool("prepare_cancel_order", **AMAN)
    assert prepared.status == "ok"  # it was still allowed when prepared
    kit.clock.advance(8)  # the agent asks "cancel kar doon?", the caller says "haan"
    ticket = await kit.guard.issue_ticket(prepared.data["token"])
    reply = await kit.tool("confirm", token=prepared.data["token"], ticket=ticket)
    # The truthful answer the agent must pass on:
    assert reply.message == "Order 4530 is already out for delivery, so it can't be cancelled."
    assert (await kit.one("SELECT status FROM orders WHERE order_no = 4530"))["status"] == "out_for_delivery"
    assert await kit.rows("SELECT * FROM refunds WHERE order_no = 4530") == []


async def test_a_cancelled_order_is_never_sent_out_for_delivery(open_call):
    rider = PlannedChange("4530", "out_for_delivery", after_event="done:cancel_order")
    kit = await open_call(FoodPack(), timeline=(rider,))
    prepared = await kit.tool("prepare_cancel_order", **AMAN)
    ticket = await kit.guard.issue_ticket(prepared.data["token"])
    await kit.tool("confirm", token=prepared.data["token"], ticket=ticket)
    assert (await kit.one("SELECT status FROM orders WHERE order_no = 4530"))["status"] == "cancelled"
    assert await kit.one("SELECT state, note FROM timeline") == {
        "state": "skipped",
        "note": "cancelled orders never move",
    }


async def test_plain_agent_that_retries_a_lost_refund_pays_twice(open_call):
    lost = Switch("refund_order", "reply_lost", seconds=2)
    kit = await open_call(FoodPack(), mode="plain", switches=(lost,))
    assert (await kit.tool("refund_order", **PRIYA_REFUND)).status == "error"
    assert (await kit.tool("refund_order", **PRIYA_REFUND)).status == "ok"
    rows = await kit.rows("SELECT amount_paise FROM refunds WHERE order_no = 4512")
    assert rows == [{"amount_paise": 4000}, {"amount_paise": 4000}]  # the double refund


async def test_safe_agent_that_retries_a_lost_refund_pays_once(open_call):
    lost = Switch("refund_order", "reply_lost", seconds=2)
    kit = await open_call(FoodPack(), switches=(lost,))
    prepared = await kit.tool("prepare_refund_order", **PRIYA_REFUND)
    ticket = await kit.guard.issue_ticket(prepared.data["token"])
    assert (await kit.tool("confirm", token="T-1", ticket=ticket)).status == "error"
    retry = await kit.tool("confirm", token="T-1", ticket=ticket)
    assert retry.status == "already_done"
    assert retry.message == "Already done earlier: refund ₹40 on order 4512 to upi for a missing item. Do not do it again."
    assert len(await kit.rows("SELECT * FROM refunds WHERE order_no = 4512")) == 1


async def test_changing_only_the_reason_is_still_the_same_request(open_call):
    kit = await open_call(FoodPack())
    first = await kit.tool("prepare_refund_order", **PRIYA_REFUND)
    second = await kit.tool("prepare_refund_order", **{**PRIYA_REFUND, "reason": "damaged_item"})
    assert first.data["token"] == second.data["token"]
