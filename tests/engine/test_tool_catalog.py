"""Which tools the AI sees in each mode (decision 5 and 8)."""

import pytest

from engine.contract.action import Action, Check
from engine.service.backend import Backend
from engine.service.tool_catalog import build_catalog
from engine.storage.call_settings import CallSettings
from packs.food.pack import FoodPack
from packs.toy.pack import ToyPack


def names(mode, ticket_check):
    return [tool.name for tool in build_catalog(FoodPack().actions(), CallSettings("c", "food", mode, ticket_check))]


def test_plain_mode_shows_every_action_as_one_tool():
    assert names("plain", False) == [
        "find_order", "change_address", "cancel_order", "refund_order", "hand_over_to_human",
    ]


def test_safe_mode_splits_risky_actions_and_adds_one_confirm():
    assert names("safe", True) == [
        "find_order", "prepare_change_address", "prepare_cancel_order", "prepare_refund_order",
        "hand_over_to_human", "confirm",
    ]


def test_confirm_asks_for_a_ticket_only_when_the_ticket_check_is_on():
    with_ticket = build_catalog(FoodPack().actions(), CallSettings("c", "food", "safe", True))[-1]
    without = build_catalog(FoodPack().actions(), CallSettings("c", "food", "safe", False))[-1]
    assert with_ticket.input_schema["required"] == ["token", "ticket"]
    assert list(without.input_schema["properties"]) == ["token"]


def test_schema_marks_required_inputs_and_allowed_values():
    refund = next(t for t in build_catalog(FoodPack().actions(), CallSettings("c", "food")) if t.name == "prepare_refund_order")
    schema = refund.input_schema
    assert schema["required"] == ["order_no", "customer_name", "amount_rupees", "reason"]
    assert schema["properties"]["reason"]["enum"] == ["missing_item", "wrong_item", "damaged_item"]
    assert schema["additionalProperties"] is False


def test_a_pack_cannot_use_an_engine_tool_name():
    class Hijack(Action):
        name = "confirm"
        description = "x"

        async def check(self, tx, inputs):
            return Check.ok()

        async def do(self, tx, inputs, ctx):
            return {}

    class BadPack(ToyPack):
        def actions(self):
            return [Hijack()]

    with pytest.raises(ValueError, match="reserved"):
        Backend(None, BadPack(), None, CallSettings("c", "toy"))
