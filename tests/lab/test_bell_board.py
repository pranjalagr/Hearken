"""The bell line between a backend and the lab's bell board, over a real socket."""

import asyncio

import pytest
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from engine.lab_controls.bells import ARRIVED, PREPARED, SAVED, SAVING, Bell, SocketBells
from lab.bell_board import BellBoard

pytestmark = pytest.mark.usefixtures("no_freezes")


async def test_bells_arrive_in_order_for_their_own_call():
    board = BellBoard()
    url = await board.start()
    try:
        bells = await SocketBells.connect(url, "call-7")
        for n, name in enumerate((PREPARED, ARRIVED, SAVING, SAVED)):
            bells.ring(name, "cancel_order", n * 0.1)
        queue = board.bells_for("call-7")
        got = [await asyncio.wait_for(queue.get(), 5) for _ in range(4)]
        await bells.aclose()
    finally:
        await board.close()
    assert [h.bell.bell for h in got] == [PREPARED, ARRIVED, SAVING, SAVED]
    assert {h.bell.call_id for h in got} == {"call-7"} and got[3].bell.at == pytest.approx(0.3)
    assert all(h.received_ns >= h.bell.perf_ns for h in got)  # one clock on one machine
    assert board.bells_for("another-call").empty()


async def test_ringing_after_the_lab_went_away_never_raises():
    board = BellBoard()
    url = await board.start()
    bells = await SocketBells.connect(url, "c")
    await board.close()
    for _ in range(50):
        bells.ring(SAVED, "cancel_order", 1.0)
        await asyncio.sleep(0)
    await bells.aclose()


async def test_a_backend_that_cannot_reach_the_lab_fails_loudly():
    with pytest.raises(OSError):
        await SocketBells.connect("ws://127.0.0.1:9/bells", "c")


async def test_the_board_cuts_off_a_malformed_bell_with_1002():
    board = BellBoard()
    url = await board.start()
    try:
        async with connect(url, proxy=None) as ws:
            await ws.send('{"bell": "rang", "action": "x"}')
            with pytest.raises(ConnectionClosed) as closed:
                await asyncio.wait_for(ws.recv(), 5)
        assert closed.value.rcvd.code == 1002
    finally:
        await board.close()


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        '{"bell": "saved", "action": "a", "call_id": "c", "at": 1.0}',  # missing perf_ns
        '{"bell": "rung", "action": "a", "call_id": "c", "at": 1.0, "perf_ns": 5}',  # unknown bell
        '{"bell": "saved", "action": "a", "call_id": "c", "at": true, "perf_ns": 5}',
        '{"bell": "saved", "action": "a", "call_id": "c", "at": 1.0, "perf_ns": 5, "extra": 1}',
    ],
)
def test_a_bell_must_match_exactly(text):
    with pytest.raises(ValueError):
        Bell.from_json(text)


def test_a_bell_survives_a_round_trip():
    bell = Bell(SAVING, "refund_order", "c1", 2.5, 123)
    assert Bell.from_json(bell.to_json()) == bell
