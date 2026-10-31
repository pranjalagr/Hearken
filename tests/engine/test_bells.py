"""Bells (Part 3, C10): which ones ring, in what order, and at what time.

The pretend clock makes the times exact: with the real normal delays, a
prepare takes 0.3 s and a confirm's wait before the save takes 0.8 s.
"""

import pytest

from engine.lab_controls.bells import ARRIVED, PREPARED, SAVED, SAVING, RecordingBells
from engine.lab_controls.normal_delays import NormalDelays
from engine.lab_controls.switches import Switch
from packs.toy.pack import ToyPack

MOVE = {"from_box": "a", "to_box": "b", "amount_paise": 500}


def heard(bells: RecordingBells) -> list[tuple[str, str, float]]:
    return [(bell.bell, bell.action, round(bell.at, 6)) for bell in bells.rung]


async def prepare_and_approve(kit):
    prepared = await kit.tool("prepare_move_money", **MOVE)
    return prepared.data["token"], await kit.guard.issue_ticket(prepared.data["token"])


async def test_a_safe_action_rings_all_four_bells_in_order_at_the_right_times(open_call):
    bells = RecordingBells()
    kit = await open_call(ToyPack(), delays=NormalDelays(), bells=bells)
    token, ticket = await prepare_and_approve(kit)
    assert (await kit.tool("confirm", token=token, ticket=ticket)).status == "ok"
    assert heard(bells) == [
        (PREPARED, "move_money", 0.3),  # after the prepare's own 0.3 s
        (ARRIVED, "move_money", 0.3),  # the confirm arrives...
        (SAVING, "move_money", 0.3),  # ...and starts its 0.8 s wait at once
        (SAVED, "move_money", 1.1),  # committed after the wait
    ]


async def test_plain_mode_rings_for_the_risky_action_itself(open_call):
    bells = RecordingBells()
    kit = await open_call(ToyPack(), mode="plain", delays=NormalDelays(), bells=bells)
    await kit.tool("read_boxes")  # a lookup: no race moment, no bell
    assert (await kit.tool("move_money", **MOVE)).status == "ok"
    assert heard(bells) == [
        (ARRIVED, "move_money", 0.3),
        (SAVING, "move_money", 0.3),
        (SAVED, "move_money", 1.1),
    ]


async def test_saved_rings_before_a_lost_or_slow_reply_is_delivered(open_call):
    """Race moment 4: the change is in, the robot hasn't heard back yet."""
    bells = RecordingBells()
    lost = Switch("move_money", "reply_lost", seconds=3)
    kit = await open_call(ToyPack(), mode="plain", switches=(lost,), delays=NormalDelays(), bells=bells)
    assert (await kit.tool("move_money", **MOVE)).status == "error"
    assert heard(bells)[-1] == (SAVED, "move_money", 0.8)
    assert kit.clock.now() == pytest.approx(3.8)  # the reply came 3 s after the bell


async def test_a_refused_confirm_rings_no_saved(open_call):
    bells = RecordingBells()
    kit = await open_call(ToyPack(), bells=bells)
    token, ticket = await prepare_and_approve(kit)
    await kit.guard.cancel_ticket(ticket, "caller said ruko")
    assert (await kit.tool("confirm", token=token, ticket=ticket)).status == "refused"
    assert [b.bell for b in bells.rung] == [PREPARED, ARRIVED, SAVING]


async def test_an_already_done_confirm_rings_no_second_saved(open_call):
    bells = RecordingBells()
    kit = await open_call(ToyPack(), bells=bells)
    token, ticket = await prepare_and_approve(kit)
    await kit.tool("confirm", token=token, ticket=ticket)
    assert (await kit.tool("confirm", token=token, ticket=ticket)).status == "already_done"
    assert [b.bell for b in bells.rung].count(SAVED) == 1


async def test_a_refused_prepare_rings_nothing_and_a_repeated_one_rings_again(open_call):
    bells = RecordingBells()
    kit = await open_call(ToyPack(), bells=bells)
    too_much = await kit.tool("prepare_move_money", from_box="a", to_box="b", amount_paise=10**9)
    assert too_much.status == "refused" and bells.rung == []
    first = await kit.tool("prepare_move_money", **MOVE)
    second = await kit.tool("prepare_move_money", **MOVE)
    assert first.data["token"] == second.data["token"]  # the same request, the same token
    assert [b.bell for b in bells.rung] == [PREPARED, PREPARED]  # the question is asked again


async def test_a_confirm_for_an_unknown_token_rings_nothing(open_call):
    bells = RecordingBells()
    kit = await open_call(ToyPack(), ticket_check=False, bells=bells)
    assert (await kit.tool("confirm", token="T-99")).status == "refused"
    assert bells.rung == []


async def test_a_broken_bell_line_never_breaks_a_money_action(open_call, caplog):
    class Broken:
        def ring(self, bell, action, at):
            raise RuntimeError("the lab is on fire")

    kit = await open_call(ToyPack(), mode="plain", bells=Broken())
    assert (await kit.tool("move_money", **MOVE)).status == "ok"
    assert (await kit.one("SELECT balance_paise FROM boxes WHERE name = 'b'"))["balance_paise"] == 500
    assert "could not ring" in caplog.text
