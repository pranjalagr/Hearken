"""Whole calls over real sockets: the dummy robot (with the real backend), the
bell board, and a caller line carrying the director (Part 3, C10, C12, C14)."""

import asyncio
import contextlib
from pathlib import Path

import pytest

from agents.common.call_box import CallBoxSettings
from common.wire.caller_line import CallerLine, QueueMic
from common.wire.protocol import PIECE_BYTES
from common.wire.robot_line import bound_port, serve_robot
from engine.lab_controls.bells import SocketBells
from engine.lab_controls.clock import WallClock
from engine.lab_controls.normal_delays import NO_DELAYS, NormalDelays
from engine.safety.guard_door import GuardDoor
from engine.service.backend import Backend
from engine.storage.call_factory import CallFactory
from engine.storage.call_settings import read_call_settings
from engine.storage.database import SqliteDatabase
from engine.tasks.caller_spec import DropSpec, RaceSpec
from engine.tasks.task_file import load_task
from engine.tasks.task_setup import start_call
from lab.bell_board import BellBoard
from lab.caller.director.director import caller_with_director, follow_bells
from lab.caller.director.plan import DirectorPlan, DirectorSettings
from lab.dummy_robot.robot import AskKey, DummyRobot, EchoRobot, HangUp, Listen, Say, SayWithPause, UseTool
from lab.synthetic_voice import SyntheticRecordings, synthetic_line
from packs.food.pack import FoodPack

pytestmark = pytest.mark.usefixtures("no_freezes")

TASK = load_task(Path(__file__).resolve().parents[2] / "packs" / "food" / "tasks" / "food_01_cancel_prepaid.yaml")
RAHUL = {"order_no": 4521, "customer_name": "Rahul Verma"}
RUKO = "ruko, cancel mat karo!"


@contextlib.asynccontextmanager
async def food_backend(tmp_path, delays: NormalDelays, bells_url: str | None = None):
    """A real call copy, its backend on one connection and the Guard's door on
    another, as in production."""
    path = await start_call(CallFactory(FoodPack(), tmp_path), FoodPack(), TASK, "lab-call", "safe", True, delays)
    db, guard_db = await SqliteDatabase.open(path), await SqliteDatabase.open(path)
    settings = await read_call_settings(db)
    clock = WallClock(settings.started_at_wall)
    bells = await SocketBells.connect(bells_url, settings.call_id) if bells_url else None
    kwargs = {"bells": bells} if bells else {}
    try:
        yield Backend(db, FoodPack(), clock, settings, **kwargs), GuardDoor(guard_db, clock), db
    finally:
        if bells:
            await bells.aclose()
        await db.close()
        await guard_db.close()


@contextlib.asynccontextmanager
async def robot_at(app):
    server = await serve_robot(app)
    try:
        yield f"ws://127.0.0.1:{bound_port(server)}/call"
    finally:
        server.close()
        await server.wait_closed()


class TurnTaker:
    """A polite stand-in for the caller's brain: after each robot turn (0.8 s
    of quiet, C9) it takes its next action: a line, a key, or nothing."""

    def __init__(self, mic: QueueMic, actions):
        self._mic, self._actions = mic, list(actions)
        self.line: CallerLine | None = None
        self._quiet_ms, self._robot_spoke, self._tasks = 0, False, set()

    def heard(self, piece, caller_ms):
        if piece is not None:
            self._robot_spoke, self._quiet_ms = True, 0
            return
        self._quiet_ms += 20
        if self._robot_spoke and self._quiet_ms >= 800 and self._actions:
            self._robot_spoke = False
            kind, value = self._actions.pop(0)
            if kind == "say":
                self._mic.say(synthetic_line(value))
            elif kind == "key":
                task = asyncio.create_task(self.line.press_key(value))
                self._tasks.add(task)


async def test_a_polite_caller_passes_every_check_of_the_dummy_robot(tmp_path):
    async with food_backend(tmp_path, NormalDelays()) as (backend, guard, db):
        robot = DummyRobot(
            [
                Say("Namaste, kya madad karoon?"),
                Listen(),
                SayWithPause("Order number aur naam", 0.4, "bataiye please"),
                Listen(),
                UseTool("find_order", RAHUL),
                AskKey("Haan ke liye 1 dabaiye"),
                UseTool("prepare_cancel_order", RAHUL),
                UseTool("confirm", approve=True),
                Say("Ho gaya, dhanyavaad"),
                HangUp("bye"),
            ],
            backend,
            guard,
        )
        mic = QueueMic()
        taker = TurnTaker(mic, [("say", "Mera order cancel karna hai"), ("say", "4521, Rahul Verma"), ("key", "1")])
        async with robot_at(robot) as url:
            line = CallerLine(url, "lab-call", mic, taker)
            taker.line = line
            reason = await asyncio.wait_for(line.run(), 60)
        async with db.transaction(write=False) as tx:
            order = await tx.fetch_one("SELECT status FROM orders WHERE order_no = 4521")
    log = robot.log
    assert reason == "robot:bye" and robot.result.reason == "robot:bye"
    assert [turn.text for turn in log.turns] == [
        "Namaste, kya madad karoon?", "Order number aur naam", "bataiye please",
        "Haan ke liye 1 dabaiye", "Ho gaya, dhanyavaad",
    ]
    for turn in log.turns:  # each turn heard for exactly its own length
        assert turn.end_ms - turn.start_ms == len(synthetic_line(turn.text)) * 20
    assert log.timeouts == [] and [key.digit for key in log.keys] == ["1"]
    assert log.talked_over() == [] and log.jumped_in() == []  # it waited through the mid-sentence pause
    assert log.tool_replies == [("find_order", "ok"), ("prepare_cancel_order", "ok"), ("confirm", "ok")]
    assert order["status"] == "cancelled"


async def race_call(tmp_path, delays: NormalDelays, settings: DirectorSettings = DirectorSettings()):
    """One race test: "ruko" while the cancel is running (moment 3)."""
    board = BellBoard()
    bells_url = await board.start()
    try:
        async with food_backend(tmp_path, delays, bells_url) as (backend, guard, _db):
            robot = DummyRobot(
                [
                    Say("Namaste"),
                    UseTool("prepare_cancel_order", RAHUL),
                    Say("Cancel kar doon? Haan ya nahi?"),
                    UseTool("confirm", approve=True),
                    Say("Ho gaya"),
                    HangUp("bye"),
                ],
                backend,
                guard,
            )
            plan = DirectorPlan(race=RaceSpec("while_running", "cancel_order", RUKO, "keep the order"), drop=None, tricks=(), seed=1)
            async with robot_at(robot) as url:
                line, director = caller_with_director(
                    url, "lab-call", plan, QueueMic(), SyntheticRecordings(), settings=settings
                )
                follower = asyncio.create_task(follow_bells(director, board.bells_for("lab-call")))
                try:
                    await asyncio.wait_for(line.run(), 60)
                finally:
                    follower.cancel()
            return director.report.race, robot.log
    finally:
        await board.close()


async def test_a_race_line_lands_inside_the_running_action(tmp_path):
    race, log = await race_call(tmp_path, NormalDelays())
    assert race.hit is True and race.why == "said before the save"
    assert race.line_ms - race.armed_ms == 320  # 0.3 s after the bell, on the next piece
    # The robot heard it start at exactly that time, and nothing from the
    # caller before it. (The puppet hangs up while the line is still going,
    # so only its start is checked.)
    first_tenth = set(range(race.line_ms, race.line_ms + 10 * 20, 20))
    assert first_tenth <= set(log.caller_speech_ms)
    assert min(log.caller_speech_ms) == race.line_ms


async def test_a_moment_that_passed_is_honestly_reported_as_missed(tmp_path):
    """The save lands at once (no wait before it), long before the director's
    line is due: the race report must say "missed", not pretend a hit."""
    race, log = await race_call(tmp_path, NO_DELAYS, DirectorSettings(race_running_after_s=2.0))
    assert race.hit is False and race.why == "the save came first"
    assert race.line_ms is None  # and the stale line was never said


async def dropping_call(tmp_path, drop: DropSpec):
    async with food_backend(tmp_path, NO_DELAYS) as (backend, guard, _db):
        robot = DummyRobot(
            [Say("Namaste, " + "bahut lambi baat " * 6), Listen(timeout_s=30)],
            backend,
            guard,
            box_settings=CallBoxSettings(no_audio_s=0.5, max_call_s=60, change_wait_s=1),
        )
        plan = DirectorPlan(race=None, drop=drop, tricks=("drops_call",), seed=1)
        async with robot_at(robot) as url:
            line, director = caller_with_director(url, "lab-call", plan, QueueMic(), SyntheticRecordings())
            reason = await asyncio.wait_for(line.run(), 30)
        return reason, robot.result, director.report.dropped


async def test_a_silent_drop_is_noticed_by_the_robots_no_audio_rule(tmp_path):
    reason, result, dropped = await dropping_call(tmp_path, DropSpec("mid_robot_sentence", "silent"))
    assert dropped.moment == "mid_robot_sentence"
    assert result.reason == "no_audio"  # nobody said goodbye: the robot noticed by itself
    assert reason == "closed"


async def test_a_clean_drop_reaches_the_robot_as_a_hang_up(tmp_path):
    reason, result, dropped = await dropping_call(tmp_path, DropSpec("mid_robot_sentence", "clean"))
    assert dropped.kind == "clean"
    assert reason == "caller:drop test" and result.reason == "caller:drop test"


async def test_the_echo_robot_plays_your_voice_back_later():
    marked = [bytes([n]) * PIECE_BYTES for n in range(1, 6)]
    heard: list[tuple[bytes, int]] = []

    class Ear:
        def heard(self, piece, caller_ms):
            if piece in marked:
                heard.append((piece, caller_ms))

    mic = QueueMic()
    mic.say(marked)  # sent at ticks 0-4
    async with robot_at(EchoRobot(delay_s=0.2)) as url:
        line = CallerLine(url, "echo", mic, Ear())
        call = asyncio.create_task(line.run())
        await asyncio.sleep(2.5)
        await line.hang_up("done")
        await asyncio.wait_for(call, 10)
    assert [piece for piece, _ in heard] == marked  # all of them, in order
    assert heard[0][1] >= 200  # no earlier than the echo delay
