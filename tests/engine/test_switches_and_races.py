"""Breakdown switches and timed races on the toy pack.

The plain-mode tests show the bug really happens (money moved twice), so the
safe-mode tests that pass are measuring a real protection, not an impossible state.
"""

import asyncio

from engine.lab_controls.switches import Switch
from engine.lab_controls.timeline import PlannedChange
from engine.service.backend import Backend
from engine.storage.database import SqliteDatabase
from packs.toy.pack import ToyPack

MOVE = {"from_box": "a", "to_box": "b", "amount_paise": 500}
REPLY_LOST = Switch("move_money", "reply_lost", seconds=3)


async def moved_total(kit):
    return (await kit.one("SELECT balance_paise FROM boxes WHERE name = 'b'"))["balance_paise"]


async def test_plain_mode_acts_at_once(open_call):
    kit = await open_call(ToyPack(), mode="plain")
    reply = await kit.tool("move_money", **MOVE)
    assert reply.status == "ok"
    assert await moved_total(kit) == 500


async def test_plain_mode_retry_after_lost_reply_moves_money_twice(open_call):
    kit = await open_call(ToyPack(), mode="plain", switches=(REPLY_LOST,))
    first = await kit.tool("move_money", **MOVE)
    assert first.status == "error" and "Timed out" in first.message
    assert await moved_total(kit) == 500  # saved, even though the agent heard a timeout
    await kit.tool("move_money", **MOVE)  # the careless retry
    assert await moved_total(kit) == 1_000  # the bug: moved twice


async def test_safe_mode_retry_after_lost_reply_moves_money_once(open_call):
    kit = await open_call(ToyPack(), switches=(REPLY_LOST,))
    prepared = await kit.tool("prepare_move_money", **MOVE)
    ticket = await kit.guard.issue_ticket(prepared.data["token"])
    first = await kit.tool("confirm", token="T-1", ticket=ticket)
    assert first.status == "error"
    assert (await kit.tool("confirm", token="T-1", ticket=ticket)).status == "already_done"
    assert (await kit.tool("prepare_move_money", **MOVE)).status == "already_done"
    assert await moved_total(kit) == 500
    log = await kit.one("SELECT outcome, switch_note FROM action_log WHERE tool = 'confirm' ORDER BY id LIMIT 1")
    assert log == {"outcome": "ok", "switch_note": "reply lost: agent got a timeout after 3s"}


async def test_lost_reply_arrives_after_the_switch_delay(open_call):
    kit = await open_call(ToyPack(), mode="plain", switches=(REPLY_LOST,))
    await kit.tool("move_money", **MOVE)
    assert kit.clock.now() == 3  # the pretend wait moved the clock


async def test_fail_switch_saves_nothing_and_breaks_only_once(open_call):
    kit = await open_call(ToyPack(), mode="plain", switches=(Switch("move_money", "fail"),))
    assert (await kit.tool("move_money", **MOVE)).status == "error"
    assert await moved_total(kit) == 0
    assert (await kit.tool("move_money", **MOVE)).status == "ok"  # times = 1
    assert await moved_total(kit) == 500


async def test_a_refused_call_does_not_use_up_a_switch(open_call):
    kit = await open_call(ToyPack(), mode="plain", switches=(Switch("move_money", "fail"),))
    too_much = await kit.tool("move_money", from_box="a", to_box="b", amount_paise=99_999)
    assert too_much.status == "refused"
    assert (await kit.tool("move_money", **MOVE)).status == "error"  # the switch is still armed


async def test_ruko_during_a_slow_wait_before_the_save_still_stops_it(open_call):
    """Race moment 3: the action is running but not saved yet."""
    slow = Switch("move_money", "slow", seconds=0.3, delay_at="before_save")
    kit = await open_call(ToyPack(), switches=(slow,), real_sleep=True)
    prepared = await kit.tool("prepare_move_money", **MOVE)
    ticket = await kit.guard.issue_ticket(prepared.data["token"])
    confirm = asyncio.create_task(kit.tool("confirm", token="T-1", ticket=ticket))
    await asyncio.sleep(0.05)  # the confirm is now waiting
    assert await kit.guard.cancel_ticket(ticket, "caller said ruko") is True
    reply = await confirm
    assert reply.status == "refused"
    assert await moved_total(kit) == 0


async def test_ruko_during_a_slow_wait_after_the_save_is_too_late(open_call):
    """Race moment 4: already saved, but the agent has not heard back."""
    slow = Switch("move_money", "slow", seconds=0.3, delay_at="after_save")
    kit = await open_call(ToyPack(), switches=(slow,), real_sleep=True)
    prepared = await kit.tool("prepare_move_money", **MOVE)
    ticket = await kit.guard.issue_ticket(prepared.data["token"])
    confirm = asyncio.create_task(kit.tool("confirm", token="T-1", ticket=ticket))
    await asyncio.sleep(0.05)
    assert await kit.guard.cancel_ticket(ticket, "caller said ruko") is False  # too late
    assert (await confirm).status == "ok"
    assert await moved_total(kit) == 500


async def test_change_right_after_prepare_beats_the_confirm(open_call):
    lock = PlannedChange("a", "lock", after_event="prepared:move_money")
    kit = await open_call(ToyPack(), timeline=(lock,))
    prepared = await kit.tool("prepare_move_money", **MOVE)
    ticket = await kit.guard.issue_ticket(prepared.data["token"])
    reply = await kit.tool("confirm", token="T-1", ticket=ticket)
    assert reply.status == "refused"
    assert await moved_total(kit) == 0


async def test_time_triggered_change_applies_at_the_start_of_the_next_call(open_call):
    kit = await open_call(ToyPack(), mode="plain", timeline=(PlannedChange("a", "lock", at_second=30),))
    kit.clock.advance(10)
    assert (await kit.tool("move_money", **MOVE)).status == "ok"
    kit.clock.advance(25)  # now second 35
    assert (await kit.tool("move_money", **MOVE)).message == "A box is locked."
    row = await kit.one("SELECT state, done_at FROM timeline")
    assert row == {"state": "applied", "done_at": 35}


async def test_a_change_the_pack_forbids_is_skipped_with_a_reason(open_call):
    kit = await open_call(ToyPack(), mode="plain", timeline=(PlannedChange("zzz", "lock", at_second=0),))
    await kit.tool("read_boxes")
    row = await kit.one("SELECT state, note FROM timeline")
    assert row == {"state": "skipped", "note": "no box 'zzz'"}


async def test_two_confirms_at_the_same_moment_act_once(open_call):
    """Two connections to the same call, as two processes would have, both
    confirm the same token at once. BEGIN IMMEDIATE makes one of them wait."""
    kit = await open_call(ToyPack(), ticket_check=False)
    second_db = await SqliteDatabase.open(kit.path)
    try:
        second = Backend(second_db, ToyPack(), kit.clock, kit.settings)
        await kit.tool("prepare_move_money", **MOVE)
        replies = await asyncio.gather(
            kit.backend.call_tool("confirm", {"token": "T-1"}),
            second.call_tool("confirm", {"token": "T-1"}),
        )
    finally:
        await second_db.close()
    assert sorted(reply.status for reply in replies) == ["already_done", "ok"]
    assert await moved_total(kit) == 500


async def test_two_confirms_on_one_connection_act_once(open_call):
    kit = await open_call(ToyPack(), ticket_check=False)
    await kit.tool("prepare_move_money", **MOVE)
    replies = await asyncio.gather(kit.tool("confirm", token="T-1"), kit.tool("confirm", token="T-1"))
    assert sorted(reply.status for reply in replies) == ["already_done", "ok"]
    assert await moved_total(kit) == 500
