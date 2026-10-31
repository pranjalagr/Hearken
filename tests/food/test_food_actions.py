"""The five food actions, driven through the real backend."""

from packs.food.pack import FoodPack

RAHUL = {"order_no": 4521, "customer_name": "Rahul Verma"}


async def approve_and_confirm(kit, prepare_tool, **inputs):
    """Prepare, let the Guard approve as if the caller said a clear yes, then confirm."""
    prepared = await kit.tool(prepare_tool, **inputs)
    if prepared.status != "ok":
        return prepared
    ticket = await kit.guard.issue_ticket(prepared.data["token"])
    return await kit.tool("confirm", token=prepared.data["token"], ticket=ticket)


async def test_find_order_gives_a_short_answer(open_call):
    kit = await open_call(FoodPack())
    reply = await kit.tool("find_order", order_no=4561, customer_name="priya")
    assert reply.status == "ok"
    assert reply.data == {
        "order_no": 4561,
        "status": "placed",
        "items": ["1 x Club Sandwich", "1 x Cold Coffee"],
        "amount": "₹249.50",
        "payment": "card",
        "address": "House 7, Lane 3, Model Town, Ludhiana",
        "refunded": "₹0",
    }


async def test_find_order_checks_who_is_calling(open_call):
    kit = await open_call(FoodPack())
    wrong_name = await kit.tool("find_order", order_no=4521, customer_name="Priya")
    assert wrong_name.status == "refused" and "name does not match" in wrong_name.message
    unknown = await kit.tool("find_order", order_no=4599, customer_name="Priya")
    assert unknown.message == "No order 4599 found. Ask the caller to repeat the number."


async def test_cancel_prepaid_order_refunds_to_the_original_method(open_call):
    kit = await open_call(FoodPack())
    prepared = await kit.tool("prepare_cancel_order", **RAHUL)
    assert prepared.data["summary"] == "cancel order 4521 and refund ₹450 to card"
    ticket = await kit.guard.issue_ticket(prepared.data["token"])
    done = await kit.tool("confirm", token=prepared.data["token"], ticket=ticket)
    assert done.data == {"order_no": 4521, "status": "cancelled", "refund": "₹450"}
    refund = await kit.one("SELECT amount_paise, method, token FROM refunds WHERE order_no = 4521")
    assert refund == {"amount_paise": 45000, "method": "card", "token": "T-1"}


async def test_cancel_cash_order_has_no_refund(open_call):
    kit = await open_call(FoodPack())
    reply = await approve_and_confirm(kit, "prepare_cancel_order", order_no=4533, customer_name="Neha")
    assert reply.message == "Done: cancel order 4533 (cash on delivery, so nothing to refund)."
    assert await kit.rows("SELECT * FROM refunds WHERE order_no = 4533") == []


async def test_cancel_is_refused_once_the_order_left_or_was_cancelled(open_call):
    kit = await open_call(FoodPack())
    gone = await kit.tool("prepare_cancel_order", order_no=4540, customer_name="Vikram")
    assert gone.message == "Order 4540 is already out for delivery, so it can't be cancelled."
    twice = await kit.tool("prepare_cancel_order", order_no=4572, customer_name="Neha")
    assert twice.message == "Order 4572 is already cancelled."


async def test_change_address_only_before_dispatch(open_call):
    kit = await open_call(FoodPack())
    new = "Flat 2C, Rose Apartments, Model Town, Ludhiana"
    done = await approve_and_confirm(kit, "prepare_change_address", order_no=4561, customer_name="Priya", new_address=new)
    assert done.status == "ok"
    assert (await kit.one("SELECT address FROM orders WHERE order_no = 4561"))["address"] == new
    late = await kit.tool("prepare_change_address", order_no=4540, customer_name="Vikram", new_address=new)
    assert late.message == "Order 4540 is out for delivery, so the address can't be changed now."


async def test_partial_refund_for_a_missing_item(open_call):
    kit = await open_call(FoodPack())
    reply = await approve_and_confirm(
        kit, "prepare_refund_order", order_no=4512, customer_name="Priya", amount_rupees=40, reason="missing_item"
    )
    assert reply.message == "Done: refund ₹40 on order 4512 to upi for a missing item."
    assert (await kit.one("SELECT amount_paise FROM refunds WHERE order_no = 4512"))["amount_paise"] == 4000


async def test_refund_is_refused_when_it_breaks_a_rule(open_call):
    kit = await open_call(FoodPack())
    sana = {"order_no": 4547, "customer_name": "Sana", "reason": "missing_item"}
    # ₹500 paid, ₹100 already refunded before the call, so ₹400 is left.
    too_much = await kit.tool("prepare_refund_order", **sana, amount_rupees=401)
    assert too_much.message == "That is more than what is left to refund on order 4547 (₹400)."
    other_method = await kit.tool("prepare_refund_order", **sana, amount_rupees=50, to_method="card")
    assert other_method.message == "Refunds only go back to the original payment method (upi)."
    odd_amount = await kit.tool("prepare_refund_order", **sana, amount_rupees=10.555)
    assert odd_amount.message == "The amount must be in rupees, like 40 or 120.50."
    not_delivered = await kit.tool("prepare_refund_order", **RAHUL, amount_rupees=50, reason="missing_item")
    assert "only for delivered orders" in not_delivered.message
    assert await kit.rows("SELECT * FROM refunds WHERE at >= 0") == []  # no refund was made


async def test_hand_over_always_works(open_call):
    kit = await open_call(FoodPack())
    assert (await kit.tool("hand_over_to_human", reason="caller is upset", order_no=4540)).status == "ok"
    assert (await kit.tool("hand_over_to_human", reason="number misheard", order_no=9999)).status == "ok"
    rows = await kit.rows("SELECT order_no, reason FROM handoffs ORDER BY handoff_id")
    assert rows == [
        {"order_no": 4540, "reason": "caller is upset"},
        {"order_no": None, "reason": "number misheard"},  # kept, even with a bad number
    ]


async def test_plain_mode_cancels_at_once_without_a_yes(open_call):
    kit = await open_call(FoodPack(), mode="plain")
    reply = await kit.tool("cancel_order", **RAHUL)
    assert reply.message == "Done: cancel order 4521 and refund ₹450 to card."
    refund = await kit.one("SELECT token FROM refunds WHERE order_no = 4521")
    assert refund == {"token": None}  # plain mode has no token
