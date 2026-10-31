"""Real calls over real sockets on this machine: both ends of the wire."""

import asyncio
import contextlib

import pytest
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from common.wire.caller_line import CallerLine, QueueMic, SilentMic
from common.wire.protocol import PIECE_BYTES, Cleared, End, Key, Played, ProtocolError, Start, encode
from common.wire.robot_line import CallerAudio, LineClosed, RobotLine, bound_port, serve_robot

pytestmark = pytest.mark.usefixtures("no_freezes")


def piece(n: int) -> bytes:
    return n.to_bytes(2, "little") + bytes(PIECE_BYTES - 2)


class Ear:
    """Records every tick: (robot piece or None, caller_ms)."""

    def __init__(self) -> None:
        self.ticks: list[tuple[bytes | None, int]] = []

    def heard(self, piece, caller_ms) -> None:
        self.ticks.append((piece, caller_ms))

    def pieces(self) -> list[tuple[bytes, int]]:
        return [(p, ms) for p, ms in self.ticks if p is not None]


@contextlib.asynccontextmanager
async def robot(app):
    server = await serve_robot(app)
    try:
        yield f"ws://127.0.0.1:{bound_port(server)}/call"
    finally:
        server.close()
        await server.wait_closed()


async def next_event(events, kind):
    async for event in events:
        if isinstance(event, kind):
            return event
    raise AssertionError(f"the line ended before a {kind.__name__}")


async def test_the_caller_sends_one_piece_every_20_ms_on_its_own_clock():
    got: list[CallerAudio] = []

    async def app(line: RobotLine):
        assert line.start == Start("call-1")
        async for event in line.events():
            if isinstance(event, CallerAudio):
                got.append(event)
                if len(got) == 25:  # half a second of the caller's audio
                    await line.end("done")
                    return

    async with robot(app) as url:
        reason = await asyncio.wait_for(CallerLine(url, "call-1", SilentMic()).run(), 10)
    assert reason == "robot:done"
    assert [event.caller_ms for event in got] == [20 * k for k in range(25)]


async def test_a_mark_comes_back_exactly_when_its_audio_has_played():
    played: list[Played] = []

    async def app(line: RobotLine):
        for n in range(25):  # half a second of robot audio, sent all at once
            await line.send_audio(piece(n))
        await line.send_mark("q1")
        events = line.events()
        played.append(await next_event(events, Played))
        await line.end("done")

    ear = Ear()
    async with robot(app) as url:
        await asyncio.wait_for(CallerLine(url, "c", SilentMic(), ear).run(), 10)
    heard = ear.pieces()
    assert [p for p, _ in heard] == [piece(n) for n in range(25)]  # all of it, in order
    first_ms = heard[0][1]
    assert [ms for _, ms in heard] == [first_ms + 20 * n for n in range(25)]  # one per tick, no gaps
    assert played == [Played("q1", first_ms + 25 * 20)]  # exactly when the last piece ended


async def test_clear_stops_the_robot_at_once_and_reports_what_was_heard():
    cleared: list[Cleared] = []

    async def app(line: RobotLine):
        for n in range(100):  # 2 seconds of robot audio
            await line.send_audio(piece(n))
        await line.send_mark("question-end")
        events = line.events()
        for _ in range(10):  # let 0.2 s of the call pass
            await next_event(events, CallerAudio)
        await line.clear()  # the caller said "ruko"
        cleared.append(await next_event(events, Cleared))
        for _ in range(10):  # 0.2 s more: nothing must play now
            await next_event(events, CallerAudio)
        await line.end("done")

    ear = Ear()
    async with robot(app) as url:
        await asyncio.wait_for(CallerLine(url, "c", SilentMic(), ear).run(), 10)
    reply = cleared[0]
    assert 1 <= reply.played_pieces < 100
    assert reply.dropped_marks == ("question-end",)  # the question never finished
    heard = ear.pieces()
    assert len(heard) == reply.played_pieces  # the reply matches what really played
    assert [p for p, _ in heard] == [piece(n) for n in range(reply.played_pieces)]
    assert all(ms < reply.caller_ms for _, ms in heard)


async def test_a_key_reaches_the_robot_on_the_callers_clock():
    keys: list[Key] = []

    async def app(line: RobotLine):
        keys.append(await next_event(line.events(), Key))
        await line.end("done")

    async def caller(url):
        line = CallerLine(url, "c", SilentMic())
        call = asyncio.create_task(line.run())
        await asyncio.sleep(0.1)
        await line.press_key("1")
        return await asyncio.wait_for(call, 10)

    async with robot(app) as url:
        await caller(url)
    assert keys[0].digit == "1" and keys[0].caller_ms > 0 and keys[0].caller_ms % 20 == 0


async def test_a_clean_hang_up_reaches_the_robot():
    seen: list[object] = []

    async def app(line: RobotLine):
        async for event in line.events():
            if not isinstance(event, CallerAudio):
                seen.append(event)

    async def caller(url):
        line = CallerLine(url, "c", SilentMic())
        call = asyncio.create_task(line.run())
        await asyncio.sleep(0.1)
        await line.hang_up("bye")
        return await asyncio.wait_for(call, 10)

    async with robot(app) as url:
        assert await caller(url) == "caller:bye"
    assert seen[0] == End("bye")


async def test_a_silent_drop_stops_the_audio_but_leaves_the_line_open():
    """The robot must notice by itself (its 30 s rule); here it waits 0.4 s."""
    observed: dict[str, object] = {}

    async def app(line: RobotLine):
        events = line.events()
        await next_event(events, CallerAudio)
        try:
            while True:
                event = await asyncio.wait_for(anext(events), 0.4)
                if isinstance(event, LineClosed):
                    observed["closed"] = True
                    return
        except TimeoutError:
            observed["quiet_for_0.4s"] = True
        await line.end("no audio")

    async def caller(url):
        line = CallerLine(url, "c", SilentMic())
        call = asyncio.create_task(line.run())
        await asyncio.sleep(0.15)
        line.go_silent()
        return await asyncio.wait_for(call, 10)

    async with robot(app) as url:
        assert await caller(url) == "robot:no audio"
    assert observed == {"quiet_for_0.4s": True}


async def test_the_callers_voice_reaches_the_robot_piece_for_piece():
    got: list[bytes] = []

    async def app(line: RobotLine):
        async for event in line.events():
            if isinstance(event, CallerAudio) and event.piece != bytes(PIECE_BYTES):
                got.append(event.piece)
                if len(got) == 5:
                    await line.end("done")
                    return

    mic = QueueMic()
    mic.say(piece(n) for n in range(1, 6))
    async with robot(app) as url:
        await asyncio.wait_for(CallerLine(url, "c", mic).run(), 10)
    assert got == [piece(n) for n in range(1, 6)]


async def closed_code(url: str, *messages) -> int:
    async with connect(url, compression=None, proxy=None) as ws:
        for message in messages:
            await ws.send(message)
        with pytest.raises(ConnectionClosed) as closed:
            while True:
                await asyncio.wait_for(ws.recv(), 5)
    return closed.value.rcvd.code


async def test_audio_before_start_is_cut_off_with_1002():
    async def app(line):
        raise AssertionError("must never be reached")

    async with robot(app) as url:
        assert await closed_code(url, bytes(PIECE_BYTES)) == 1002


async def test_a_wrong_size_piece_is_cut_off_with_1002():
    async def app(line):
        async for _ in line.events():
            pass

    async with robot(app) as url:
        assert await closed_code(url, encode(Start("c")), bytes(100)) == 1002


async def test_a_second_start_is_cut_off_with_1002():
    async def app(line):
        async for _ in line.events():
            pass

    async with robot(app) as url:
        assert await closed_code(url, encode(Start("c")), encode(Start("c"))) == 1002


async def test_a_robot_that_breaks_the_rules_fails_the_call():
    async def app(line: RobotLine):
        await line._connection.send(encode(Start("robots may not send this")))
        async for _ in line.events():
            pass

    async with robot(app) as url:
        with pytest.raises(ProtocolError):
            await asyncio.wait_for(CallerLine(url, "c", SilentMic()).run(), 10)
