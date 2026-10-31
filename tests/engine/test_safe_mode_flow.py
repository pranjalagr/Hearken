"""Safe mode on the toy pack: tokens, yes-tickets, expiry and the order of checks.

Running these on the toy pack proves the engine works with no food code at all.
"""

import pytest

from engine.lab_controls.timeline import PlannedChange
from engine.safety.guard_door import TicketRefused
from packs.toy.pack import MoveMoney, ReadBoxes, ToyPack

MOVE = {"from_box": "a", "to_box": "b", "amount_paise": 500}


async def balances(kit):
    rows = await kit.rows("SELECT name, balance_paise FROM boxes ORDER BY name")
    return {row["name"]: row["balance_paise"] for row in rows}


async def prepare_and_approve(kit, **move):
    """Prepare a move, then the Guard approves it as if the caller said a clear yes."""
    prepared = await kit.tool("prepare_move_money", **(move or MOVE))
    assert prepared.status == "ok", prepared.message
    ticket = await kit.guard.issue_ticket(prepared.data["token"])
    return prepared.data["token"], ticket


async def test_prepare_changes_nothing_and_confirm_acts_once(open_call):
    kit = await open_call(ToyPack())
    prepared = await kit.tool("prepare_move_money", **MOVE)
    assert prepared.data == {"token": "T-1", "summary": "move 500 paise from box a to box b"}
    assert await balances(kit) == {"a": 10_000, "b": 0}  # prepare changes nothing

    ticket = await kit.guard.issue_ticket("T-1")
    done = await kit.tool("confirm", token="T-1", ticket=ticket)
    assert done.status == "ok"
    assert done.message == "Done: move 500 paise from box a to box b."
    assert await balances(kit) == {"a": 9_500, "b": 500}
    assert (await kit.one("SELECT state FROM prepared_actions"))["state"] == "done"
    assert (await kit.one("SELECT state FROM tickets"))["state"] == "used"


async def test_a_second_confirm_with_the_same_token_does_nothing(open_call):
    kit = await open_call(ToyPack())
    token, ticket = await prepare_and_approve(kit)
    await kit.tool("confirm", token=token, ticket=ticket)
    again = await kit.tool("confirm", token=token, ticket=ticket)
    assert again.status == "already_done"
    assert await balances(kit) == {"a": 9_500, "b": 500}  # moved once, not twice


async def test_risky_action_cannot_skip_prepare_in_safe_mode(open_call):
    kit = await open_call(ToyPack())
    reply = await kit.tool("move_money", **MOVE)
    assert reply.status == "refused"
    assert await balances(kit) == {"a": 10_000, "b": 0}


async def test_confirm_without_a_ticket_is_refused(open_call):
    kit = await open_call(ToyPack())
    await kit.tool("prepare_move_money", **MOVE)
    reply = await kit.tool("confirm", token="T-1")
    assert reply.status == "refused"
    assert "yes-ticket is needed" in reply.message
    assert await balances(kit) == {"a": 10_000, "b": 0}


async def test_a_ticket_only_works_for_its_own_action(open_call):
    kit = await open_call(ToyPack())
    _, ticket_for_first = await prepare_and_approve(kit)
    await kit.tool("prepare_move_money", from_box="a", to_box="b", amount_paise=9_000)  # T-2
    reply = await kit.tool("confirm", token="T-2", ticket=ticket_for_first)
    assert reply.status == "refused"
    assert "different action" in reply.message
    assert await balances(kit) == {"a": 10_000, "b": 0}


async def test_a_made_up_ticket_is_refused(open_call):
    kit = await open_call(ToyPack())
    await kit.tool("prepare_move_money", **MOVE)
    reply = await kit.tool("confirm", token="T-1", ticket="K-99")
    assert reply.status == "refused"
    assert await balances(kit) == {"a": 10_000, "b": 0}


async def test_ruko_cancels_the_ticket_and_confirm_is_refused(open_call):
    kit = await open_call(ToyPack())
    token, ticket = await prepare_and_approve(kit)
    assert await kit.guard.cancel_ticket(ticket, "caller said ruko") is True
    reply = await kit.tool("confirm", token=token, ticket=ticket)
    assert reply.status == "refused"
    assert "stopped" in reply.message
    assert await balances(kit) == {"a": 10_000, "b": 0}


async def test_cancelling_a_used_ticket_reports_too_late(open_call):
    kit = await open_call(ToyPack())
    token, ticket = await prepare_and_approve(kit)
    await kit.tool("confirm", token=token, ticket=ticket)
    # The agent must now tell the truth ("it already happened"), not promise to stop it.
    assert await kit.guard.cancel_ticket(ticket, "caller said ruko") is False


async def test_prepared_action_expires(open_call):
    kit = await open_call(ToyPack())
    token, ticket = await prepare_and_approve(kit)
    kit.clock.advance(121)  # prepared actions live 120 s
    reply = await kit.tool("confirm", token=token, ticket=ticket)
    assert reply.status == "expired"
    assert await balances(kit) == {"a": 10_000, "b": 0}


async def test_old_ticket_is_refused(open_call):
    kit = await open_call(ToyPack())
    token, ticket = await prepare_and_approve(kit)
    kit.clock.advance(16)  # tickets live 15 s
    reply = await kit.tool("confirm", token=token, ticket=ticket)
    assert reply.status == "refused"
    assert "too old" in reply.message


async def test_done_is_checked_before_expired(open_call):
    """A late retry after a lost reply must hear "already done", not "expired",
    or the agent would prepare again and move the money twice."""
    kit = await open_call(ToyPack())
    token, ticket = await prepare_and_approve(kit)
    kit.clock.advance(10)
    assert (await kit.tool("confirm", token=token, ticket=ticket)).status == "ok"
    kit.clock.advance(500)  # long past the 120 s lifetime
    assert (await kit.tool("confirm", token=token, ticket=ticket)).status == "already_done"


async def test_guard_refuses_tickets_for_bad_tokens(open_call):
    kit = await open_call(ToyPack())
    with pytest.raises(TicketRefused):
        await kit.guard.issue_ticket("T-9")  # unknown
    token, ticket = await prepare_and_approve(kit)
    await kit.tool("confirm", token=token, ticket=ticket)
    with pytest.raises(TicketRefused):
        await kit.guard.issue_ticket(token)  # already done
    await kit.tool("prepare_move_money", from_box="a", to_box="b", amount_paise=100)
    kit.clock.advance(200)
    with pytest.raises(TicketRefused):
        await kit.guard.issue_ticket("T-2")  # expired


async def test_preparing_the_same_thing_again_gives_the_same_token(open_call):
    kit = await open_call(ToyPack())
    first = await kit.tool("prepare_move_money", **MOVE)
    second = await kit.tool("prepare_move_money", **MOVE)
    assert first.data["token"] == second.data["token"] == "T-1"


async def test_preparing_again_after_done_says_already_done(open_call):
    """Decision 14: an agent that starts over with prepare after a lost reply
    must not get a fresh token for the same thing."""
    kit = await open_call(ToyPack())
    token, ticket = await prepare_and_approve(kit)
    await kit.tool("confirm", token=token, ticket=ticket)
    again = await kit.tool("prepare_move_money", **MOVE)
    assert again.status == "already_done"
    assert await balances(kit) == {"a": 9_500, "b": 500}


async def test_without_ticket_check_the_token_alone_is_enough(open_call):
    kit = await open_call(ToyPack(), mode="safe", ticket_check=False)
    await kit.tool("prepare_move_money", **MOVE)
    assert (await kit.tool("confirm", token="T-1")).status == "ok"
    assert (await kit.tool("confirm", token="T-1")).status == "already_done"
    assert await balances(kit) == {"a": 9_500, "b": 500}


async def test_rules_are_checked_again_at_confirm(open_call):
    # The box gets locked at second 20, after prepare but before confirm.
    kit = await open_call(ToyPack(), timeline=(PlannedChange("a", "lock", at_second=20),))
    token, ticket = await prepare_and_approve(kit)
    kit.clock.advance(21)
    ticket = await kit.guard.issue_ticket(token)  # a fresh yes, so only the rule can stop it
    reply = await kit.tool("confirm", token=token, ticket=ticket)
    assert reply.status == "refused"
    assert reply.message == "A box is locked."
    assert await balances(kit) == {"a": 10_000, "b": 0}


async def test_bad_inputs_are_refused_before_any_rule(open_call):
    kit = await open_call(ToyPack())
    assert (await kit.tool("prepare_move_money", from_box="a", to_box="b")).message == "Missing input: amount_paise."
    assert (await kit.tool("prepare_move_money", **MOVE, extra=1)).message == "Unknown input: extra."
    reply = await kit.tool("prepare_move_money", from_box="a", to_box="b", amount_paise="lots")
    assert reply.message == "amount_paise must be a whole number."


class BrokenMove(MoveMoney):
    """Moves the money, then crashes before the transaction ends."""

    async def do(self, tx, inputs, ctx):
        await super().do(tx, inputs, ctx)
        raise RuntimeError("crash halfway")


class BrokenToyPack(ToyPack):
    def actions(self):
        return [ReadBoxes(), BrokenMove()]


async def test_a_crash_halfway_rolls_everything_back(open_call):
    kit = await open_call(BrokenToyPack())
    token, ticket = await prepare_and_approve(kit)
    reply = await kit.tool("confirm", token=token, ticket=ticket)
    assert reply.status == "error"
    assert await balances(kit) == {"a": 10_000, "b": 0}  # the half-done move is gone
    assert (await kit.one("SELECT state FROM prepared_actions"))["state"] == "waiting"
    assert (await kit.one("SELECT state FROM tickets"))["state"] == "open"
    last = await kit.one("SELECT outcome, reason FROM action_log ORDER BY id DESC LIMIT 1")
    assert last["outcome"] == "error" and "crash halfway" in last["reason"]


async def test_every_tool_call_writes_exactly_one_log_row(open_call):
    kit = await open_call(ToyPack())
    token, ticket = await prepare_and_approve(kit)
    await kit.tool("confirm", token=token, ticket=ticket)
    await kit.tool("confirm", token=token, ticket=ticket)
    await kit.tool("read_boxes")
    await kit.tool("no_such_tool")
    log = await kit.rows("SELECT tool, outcome FROM action_log ORDER BY id")
    assert [(row["tool"], row["outcome"]) for row in log] == [
        ("prepare_move_money", "ok"),
        ("confirm", "ok"),
        ("confirm", "already_done"),
        ("read_boxes", "ok"),
        ("no_such_tool", "refused"),
    ]
