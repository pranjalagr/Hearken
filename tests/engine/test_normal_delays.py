"""Normal tool delays (Part 2 decision 6) on the toy pack.

The pretend sleep moves the ManualClock, so "how long did it wait" is simply
the clock reading after the call. read_boxes is a lookup; move_money is risky.
"""

import asyncio

import pytest

from engine.lab_controls.normal_delays import NormalDelays
from engine.lab_controls.switches import Switch
from packs.toy.pack import ToyPack

MOVE = {"from_box": "a", "to_box": "b", "amount_paise": 500}
REAL = NormalDelays()  # the production values: 0.3 s lookup, 0.8 s change


async def moved_total(kit):
    return (await kit.one("SELECT balance_paise FROM boxes WHERE name = 'b'"))["balance_paise"]


async def test_a_lookup_waits_the_lookup_delay(open_call):
    kit = await open_call(ToyPack(), mode="plain", delays=REAL)
    assert (await kit.tool("read_boxes")).status == "ok"
    assert kit.clock.now() == pytest.approx(0.3)


async def test_a_risky_action_in_plain_mode_waits_the_change_delay(open_call):
    kit = await open_call(ToyPack(), mode="plain", delays=REAL)
    assert (await kit.tool("move_money", **MOVE)).status == "ok"
    assert kit.clock.now() == pytest.approx(0.8)


async def test_prepare_is_a_lookup_and_confirm_is_a_change(open_call):
    kit = await open_call(ToyPack(), delays=REAL)
    prepared = await kit.tool("prepare_move_money", **MOVE)
    assert kit.clock.now() == pytest.approx(0.3)
    ticket = await kit.guard.issue_ticket(prepared.data["token"])
    assert (await kit.tool("confirm", token="T-1", ticket=ticket)).status == "ok"
    assert kit.clock.now() == pytest.approx(1.1)


async def test_the_change_delay_comes_before_the_save(open_call):
    """The save's own time already includes the wait, so the wait happened first."""
    kit = await open_call(ToyPack(), mode="plain", delays=REAL)
    await kit.tool("move_money", **MOVE)
    move = await kit.one("SELECT at FROM moves")
    assert move["at"] == pytest.approx(0.8)


async def test_a_slow_switch_adds_on_top_of_the_normal_delay(open_call):
    slow = Switch("move_money", "slow", seconds=2, delay_at="before_save")
    kit = await open_call(ToyPack(), mode="plain", switches=(slow,), delays=REAL)
    await kit.tool("move_money", **MOVE)
    assert kit.clock.now() == pytest.approx(2.8)


async def test_a_call_stopped_at_the_door_does_not_wait(open_call):
    """Missing inputs: no work was done, so there is nothing to wait for."""
    kit = await open_call(ToyPack(), mode="plain", delays=REAL)
    assert (await kit.tool("move_money", from_box="a")).status == "refused"
    assert kit.clock.now() == 0


async def test_ruko_during_the_normal_change_delay_still_stops_it(open_call):
    """Race moment 3 without any breakdown switch: the normal wait alone is
    long enough for the Guard to cancel the ticket before the save."""
    kit = await open_call(ToyPack(), real_sleep=True, delays=NormalDelays(lookup_s=0, change_s=0.3))
    prepared = await kit.tool("prepare_move_money", **MOVE)
    ticket = await kit.guard.issue_ticket(prepared.data["token"])
    confirm = asyncio.create_task(kit.tool("confirm", token="T-1", ticket=ticket))
    await asyncio.sleep(0.05)  # the confirm is now waiting
    assert await kit.guard.cancel_ticket(ticket, "caller said ruko") is True
    assert (await confirm).status == "refused"
    assert await moved_total(kit) == 0


async def test_the_delays_are_saved_with_the_call(open_call):
    kit = await open_call(ToyPack(), delays=REAL)
    row = await kit.one("SELECT lookup_delay_s, change_delay_s FROM call_info")
    assert row == {"lookup_delay_s": 0.3, "change_delay_s": 0.8}
    assert kit.settings.delays == REAL


def test_a_negative_delay_is_refused():
    with pytest.raises(ValueError):
        NormalDelays(lookup_s=-1)
