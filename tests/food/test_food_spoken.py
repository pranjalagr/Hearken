"""The food pack's spoken words (Part 2 decisions 8, 9 and 10)."""

import pytest

from engine.contract.action import Action
from engine.contract.spoken import LANGUAGES
from packs.food import rules
from packs.food.actions import RefundOrder
from packs.food.pack import FoodPack

# Every status an order can have. Built from rules.py, so a new status there
# fails this test until it gets a phrase too.
ALL_STATUSES = set(rules.FORWARD_PATH) | {rules.CANCELLED}


def test_every_order_status_has_a_phrase_and_no_extra_ones():
    assert FoodPack().spoken_words().values("status") == ALL_STATUSES


def test_every_status_phrase_exists_in_every_language():
    words = FoodPack().spoken_words()
    for status in ALL_STATUSES:
        for language in LANGUAGES:
            assert words.phrase("status", status, language)


def test_a_status_phrase_reads_as_expected():
    words = FoodPack().spoken_words()
    assert words.phrase("status", "preparing", "hinglish") == "Kitchen mein aapka khana ban raha hai."


# ---- yes-questions (decision 9) ---------------------------------------------------


def test_every_refund_reason_and_prepaid_method_has_a_phrase():
    words = FoodPack().spoken_words()
    assert words.values("refund_reason") == set(RefundOrder.inputs["reason"].choices)
    assert words.values("payment_method") == set(rules.PREPAID)


def test_every_risky_action_says_its_own_yes_question():
    for action in FoodPack().actions():
        if action.risky:
            assert type(action).spoken_question is not Action.spoken_question, action.name


async def ask(kit, tool, **inputs):
    """Prepare, then read the yes-question the caller would hear, in both languages."""
    prepared = await kit.tool(tool, **inputs)
    assert prepared.status == "ok", prepared.message
    token = prepared.data["token"]
    return {language: await kit.guard.question(token, language) for language in LANGUAGES}


async def test_cancel_with_a_refund_names_the_amount_and_where_it_goes(open_call):
    kit = await open_call(FoodPack())
    asked = await ask(kit, "prepare_cancel_order", order_no=4521, customer_name="Rahul")
    assert asked["hinglish"] == "Order 4521 cancel karke ₹450 aapke card par wapas kar doon? Haan ya nahi?"
    assert asked["english"] == "Shall I cancel order 4521 and send ₹450 back to your card? Yes or no?"


async def test_cancel_of_a_cash_order_says_nothing_comes_back(open_call):
    kit = await open_call(FoodPack())
    asked = await ask(kit, "prepare_cancel_order", order_no=4533, customer_name="Neha")
    assert asked["hinglish"] == (
        "Order 4533 cancel kar doon? Cash on delivery tha, toh koi refund nahi banega. Haan ya nahi?"
    )


async def test_refund_names_the_amount_the_method_and_the_reason(open_call):
    kit = await open_call(FoodPack())
    asked = await ask(
        kit, "prepare_refund_order", order_no=4512, customer_name="Priya", amount_rupees=40, reason="missing_item"
    )
    assert asked["hinglish"] == "Order 4512 par ₹40 aapke UPI par wapas bhej doon, kyunki item missing tha? Haan ya nahi?"
    assert asked["english"] == "Shall I send ₹40 back to your UPI for order 4512, because an item was missing? Yes or no?"


# ---- money with paise (decision 10) ---------------------------------------------
# The voice test (1 Nov 2026) found "₹249.50" read as a decimal ("dashamlav
# paanch", "point five zero"), so the caller hears the units as words instead.


def test_the_money_words_cover_every_kind_of_amount():
    assert FoodPack().spoken_words().values("money") == {"whole", "with_paise", "only_paise"}


async def test_cancel_says_a_paise_refund_with_its_units_as_words(open_call):
    kit = await open_call(FoodPack())
    async with kit.db.transaction() as tx:  # an order that cost ₹249.50
        assert await tx.execute("UPDATE orders SET amount_paise = 24950 WHERE order_no = 4521") == 1
    prepared = await kit.tool("prepare_cancel_order", order_no=4521, customer_name="Rahul")
    asked = {language: await kit.guard.question(prepared.data["token"], language) for language in LANGUAGES}
    assert asked["hinglish"] == "Order 4521 cancel karke 249 rupaye 50 paise aapke card par wapas kar doon? Haan ya nahi?"
    assert asked["english"] == "Shall I cancel order 4521 and send 249 rupees 50 paise back to your card? Yes or no?"
    # The log and the AI's tool reply keep the short written form: they aren't spoken.
    assert prepared.data["summary"] == "cancel order 4521 and refund ₹249.50 to card"


@pytest.mark.parametrize(("rupees", "hinglish", "english"), [
    (40.5, "40 rupaye 50 paise", "40 rupees 50 paise"),
    (40.05, "40 rupaye 5 paise", "40 rupees 5 paise"),
    (0.5, "50 paise", "50 paise"),
    (40, "₹40", "₹40"),
])
async def test_refund_says_the_amount_the_way_people_say_money(open_call, rupees, hinglish, english):
    kit = await open_call(FoodPack())
    asked = await ask(
        kit, "prepare_refund_order", order_no=4512, customer_name="Priya", amount_rupees=rupees, reason="missing_item"
    )
    assert asked["hinglish"] == f"Order 4512 par {hinglish} aapke UPI par wapas bhej doon, kyunki item missing tha? Haan ya nahi?"
    assert asked["english"] == f"Shall I send {english} back to your UPI for order 4512, because an item was missing? Yes or no?"
    assert "." not in asked["hinglish"].replace("?", "")  # no decimal point left for a voice to misread


async def test_change_address_reads_the_new_address_back_exactly(open_call):
    kit = await open_call(FoodPack())
    asked = await ask(
        kit, "prepare_change_address", order_no=4561, customer_name="Priya", new_address="Flat 12, MG Road"
    )
    assert asked["hinglish"] == 'Order 4561 ka address badal kar "Flat 12, MG Road" kar doon? Haan ya nahi?'


async def test_the_question_is_kept_with_the_token_but_not_sent_to_the_ai(open_call):
    """The black box keeps what was asked; the AI's tool reply stays as it was,
    so Agents A and B read exactly what they read before."""
    kit = await open_call(FoodPack())
    prepared = await kit.tool("prepare_cancel_order", order_no=4521, customer_name="Rahul")
    row = await kit.one("SELECT questions_json FROM prepared_actions WHERE token = ?", prepared.data["token"])
    assert "Haan ya nahi?" in row["questions_json"]
    assert "Haan ya nahi" not in prepared.to_json()
