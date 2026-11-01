"""Food rules and money helpers, as plain functions (no database needed)."""

import pytest

from engine.contract.action import Phrase
from packs.food import rules
from packs.food.money import format_rupees, rupees_to_paise, spoken_amount
from packs.food.order_store import Order


def order(status="preparing", method="card", amount=45000, no=4521):
    return Order(no, "Rahul Verma", amount, method, "Flat 12B, Noida", status)


def test_name_check_accepts_full_or_first_name_in_any_case():
    assert rules.name_matches("Rahul Verma", "rahul  VERMA")
    assert rules.name_matches("Rahul Verma", "Rahul")
    assert not rules.name_matches("Rahul Verma", "Verma")
    assert not rules.name_matches("Rahul Verma", "Priya")


def test_cancel_only_before_out_for_delivery():
    assert rules.can_cancel(order("placed")).allowed
    assert rules.can_cancel(order("preparing")).allowed
    for status in ("out_for_delivery", "delivered", "cancelled"):
        assert not rules.can_cancel(order(status)).allowed


def test_cash_orders_have_nothing_to_refund():
    assert rules.refundable_left(order(method="cash"), 0) == 0
    assert rules.refundable_left(order(method="upi"), 10000) == 35000


def test_refund_rules():
    delivered = order("delivered", "upi", 50000)
    assert rules.can_refund(delivered, 10000, 40000, None).allowed  # exactly what is left
    assert "more than what is left" in rules.can_refund(delivered, 10000, 40001, None).reason
    assert "original payment method" in rules.can_refund(delivered, 0, 100, "card").reason
    assert "only for delivered" in rules.can_refund(order("preparing", "upi"), 0, 100, None).reason
    assert "paid in cash" in rules.can_refund(order("delivered", "cash"), 0, 100, None).reason


def test_timeline_moves_only_forward_and_never_moves_a_cancelled_order():
    assert rules.can_move("preparing", "out_for_delivery").allowed
    assert rules.can_move("placed", "out_for_delivery").allowed  # skipping a step is fine
    assert not rules.can_move("out_for_delivery", "preparing").allowed
    assert not rules.can_move("cancelled", "out_for_delivery").allowed


def test_money_helpers_never_lose_a_paisa():
    assert rupees_to_paise(40) == 4000
    assert rupees_to_paise(120.5) == 12050
    assert rupees_to_paise(0.1 + 0.2) is None  # 0.30000000000000004 is not a real amount
    assert rupees_to_paise(1.234) is None
    assert format_rupees(45000) == "₹450"
    assert format_rupees(24950) == "₹249.50"


@pytest.mark.parametrize(("paise", "said"), [
    (45000, Phrase("money", "whole", {"rupees": "450"})),
    (100, Phrase("money", "whole", {"rupees": "1"})),
    (24950, Phrase("money", "with_paise", {"rupees": "249", "paise": "50"})),
    # "05" would be read "zero five"; it is five paise.
    (24905, Phrase("money", "with_paise", {"rupees": "249", "paise": "5"})),
    (12001, Phrase("money", "with_paise", {"rupees": "120", "paise": "1"})),
    # Nobody says "0 rupaye 50 paise".
    (50, Phrase("money", "only_paise", {"paise": "50"})),
    (123456789, Phrase("money", "with_paise", {"rupees": "1234567", "paise": "89"})),
])
def test_an_amount_is_said_with_its_units_as_words_when_it_has_paise(paise, said):
    """Part 2 decision 10: "₹249.50" was read as a decimal by 4 of 5 voices."""
    assert spoken_amount(paise) == said
