"""The runner: start one lab call, run it, keep everything, say whose fault a
break was (Part 5, L3 and L4).

Per call:
1. make the call's database copy from its task, and keep a copy of the start
   (start.db, for "nothing else changed", L7);
2. start the robot as its own program, told only its copy and a phone port;
   wait for it to say "ready";
3. dial it with the caller (brain + director + recorder, all in this program);
4. when the call ends, stop the robot, keep the final records (final.db) and
   the caller's side of the black box;
5. label the call (L4): fine, the robot's failure, or a lab problem to rerun.

A robot is anything a RobotSpec can start: the puppet now, Agents A, B and C
later, all started the same way so the comparison is fair.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from websockets.exceptions import InvalidHandshake

from common.wire.protocol import ProtocolError
from engine.contract.pack import Pack
from engine.lab_controls.normal_delays import NormalDelays
from engine.storage.call_factory import CallFactory
from engine.storage.call_settings import SAFE
from engine.tasks.task_file import Task
from engine.tasks.task_setup import start_call
from lab.bell_board import BellBoard
from lab.caller.director.director import caller_with_director
from lab.caller.director.plan import DirectorPlan
from lab.caller.scripted import ScriptedCaller
from lab.puppets.food import PUPPETS
from lab.recorder.caller_side import CallerRecorder, Ears
from lab.settings import LabSettings

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# L4's labels.
FINE = "fine"
ROBOT_FAILURE = "robot_failure"
LAB_PROBLEM = "lab_problem"


class RobotSpec(Protocol):
    name: str  # e.g. "puppet:perfect"; the results table groups calls by it

    def command(self, db: Path, folder: Path, bells_url: str, settings: LabSettings, task: Task) -> list[str]:
        """The command line that starts this robot for one call."""


@dataclass(frozen=True)
class PuppetRobot:
    variant: str  # perfect or careless
    # "sarvam": real speech, so a caller with real ears can hear it. Beeps are enough
    # for the scripted caller, and they cost nothing (Part 3, C14).
    voice: str = "synthetic"

    @property
    def name(self) -> str:
        return f"puppet:{self.variant}"

    def command(self, db: Path, folder: Path, bells_url: str, settings: LabSettings, task: Task) -> list[str]:
        return [
            sys.executable, "-m", "cli.puppet_robot",
            "--db", str(db), "--task", task.id, "--variant", self.variant, "--port", "0",
            "--bells", bells_url, "--diary", str(folder / "robot.jsonl"),
            "--backend-events", str(folder / "backend.jsonl"),
            "--no-audio-s", str(settings.robot_no_audio_s), "--max-call-s", str(settings.robot_max_call_s),
            "--change-wait-s", str(settings.robot_change_wait_s), "--voice", self.voice,
        ]


class CallerMaker(Protocol):
    def __call__(self, plan: CallPlan, on_event: Any) -> Any:
        """The caller for one call: an object with the scripted caller's shape."""


@dataclass(frozen=True)
class CallPlan:
    call_id: str
    task: Task
    robot: RobotSpec
    mode: str
    seed: int
    attempt: int = 1
    # Who makes the caller; None is the scripted caller, for puppet lab runs.
    caller: CallerMaker | None = None


@dataclass
class CallOutcome:
    call_id: str
    task_id: str
    robot: str
    mode: str
    seed: int
    attempt: int
    label: str
    why: str
    end_reason: str = ""
    robot_end: str = ""
    max_late_ms: float = 0.0
    race: dict[str, Any] | None = None
    folder: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


async def run_call(plan: CallPlan, pack: Pack, run_dir: Path, board: BellBoard, bells_url: str,
                   settings: LabSettings, factory: CallFactory | None = None) -> CallOutcome:
    """`factory`: one per batch, shared by its calls, so the pack's template
    is built once (calls starting together must not each rebuild it)."""
    folder = run_dir / "calls" / plan.call_id
    folder.mkdir(parents=True, exist_ok=True)
    outcome = CallOutcome(plan.call_id, plan.task.id, plan.robot.name, plan.mode, plan.seed, plan.attempt,
                          label=FINE, why="", folder=str(folder))
    factory = factory or CallFactory(pack, run_dir / "copies")
    db = await start_call(factory, pack, plan.task, plan.call_id, plan.mode, plan.mode == SAFE, NormalDelays())
    await asyncio.to_thread(_backup, db, folder / "start.db")
    stderr = (folder / "robot.stderr.txt").open("wb")
    robot = await RobotProcess.start(plan.robot.command(db, folder, bells_url, settings, plan.task), stderr)
    recorder = CallerRecorder(plan.call_id)
    about_caller: dict[str, Any] = {}
    try:
        url = await _wait_ready(robot, settings.robot_ready_timeout_s)
        if url is None:
            return _label(outcome, ROBOT_FAILURE, "the robot never said ready")
        about_caller = await _talk(plan, url, board, recorder, outcome, settings)
    finally:
        outcome.robot_end, code = await _stop(robot, settings)
        stderr.close()
        await asyncio.to_thread(_backup, db, folder / "final.db")
        # Writing the audio is slow work: off this loop, which is still beating
        # 20 ms pieces for the other calls (on 1 Nov 2026, doing it here froze
        # every live call by up to 0.4 s).
        await asyncio.to_thread(recorder.save, folder, _about(plan, about_caller))
        if outcome.label == FINE:
            _judge_break(outcome, code, settings)
        (folder / "outcome.json").write_text(json.dumps(asdict(outcome), indent=1, default=str), encoding="utf-8")
    return outcome


def _scripted_caller(plan: CallPlan, on_event: Any) -> ScriptedCaller:
    english = plan.task.caller is not None and plan.task.caller.language == "mostly_english"
    return ScriptedCaller(PUPPETS[plan.task.id].caller, on_event, language="english" if english else "hinglish")


async def _talk(plan: CallPlan, url: str, board: BellBoard, recorder: CallerRecorder, outcome: CallOutcome,
                settings: LabSettings) -> dict[str, Any]:
    caller = (plan.caller or _scripted_caller)(plan, recorder.event)
    try:
        # Before dialling, so a caller that can't start never costs the robot anything.
        await caller.prepare()
    except Exception as exc:
        _label(outcome, LAB_PROBLEM, f"the caller couldn't start: {exc!r}")
        await caller.close()
        return caller.about()
    director_plan = DirectorPlan.from_task(plan.task, plan.seed)

    def director_said(happened: Any) -> None:
        recorder.director_event(happened)
        caller.director_event(happened)

    line, director = caller_with_director(
        url, plan.call_id, director_plan, caller.mic, caller.recordings,
        ear=Ears(recorder, caller), on_event=director_said, tap=recorder.tap,
        on_cleared=recorder.cleared, wrap_mic=caller.wrap_mic,
    )
    caller.attach(line, director)
    bells = board.bells_for(plan.call_id)

    async def follow() -> None:
        while True:
            heard = await bells.get()
            recorder.bell(heard)
            director.ring(heard.bell)

    late = _Heartbeat(lambda: recorder.last_ms)
    tasks = [asyncio.create_task(follow()), asyncio.create_task(late.run())]
    try:
        # Our own cap: a little over the robot's, so the robot's cap ends the call first.
        outcome.end_reason = await asyncio.wait_for(line.run(), settings.robot_max_call_s + 30)
    except ProtocolError as exc:
        _label(outcome, ROBOT_FAILURE, f"the robot broke the wire's rules: {exc}")
    except TimeoutError:  # before OSError: in Python a TimeoutError IS an OSError
        _label_never_ended(outcome, late.max_late_ms, settings)
    except (OSError, InvalidHandshake) as exc:
        # It said "ready" at an address that doesn't take the call.
        _label(outcome, ROBOT_FAILURE, f"the robot's phone didn't answer: {exc!r}")
    except Exception as exc:  # our caller's own code broke: not the robot's fault
        _label(outcome, LAB_PROBLEM, f"the caller crashed: {exc!r}")
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        outcome.max_late_ms = late.max_late_ms
        outcome.extra["worst_late_at_ms"] = late.worst_at_ms
        race = director.report.race
        outcome.race = asdict(race) if race is not None else None
        await caller.close()
        if caller.error is not None and outcome.label == FINE:
            # The caller broke (a limit, its ears, its voice): rerun it. L4: this
            # says nothing about the robot, so it never counts against one.
            _label(outcome, LAB_PROBLEM, f"the caller broke: {caller.error!r}")
        recorder.event("call.end", {"reason": outcome.end_reason, "caller_ms": recorder.last_ms})
    return caller.about()


def _label_never_ended(outcome: CallOutcome, late_ms: float, settings: LabSettings) -> None:
    """L4. If the lab itself stopped, the clock ran out on the lab, not on the robot: a
    laptop in standby for 43 minutes was marked as the robot's failure (5 Nov 2026)."""
    if late_ms > settings.freeze_limit_ms:
        _label(outcome, LAB_PROBLEM, f"the lab fell behind by {late_ms:.0f} ms")
    else:
        _label(outcome, ROBOT_FAILURE, "the call never ended")


def _judge_break(outcome: CallOutcome, robot_code: int | None, settings: LabSettings) -> None:
    """L4, for a call nothing labelled yet. When unsure, it counts against the robot (C15)."""
    if robot_code not in (0, None):
        _label(outcome, ROBOT_FAILURE, f"the robot program exited with code {robot_code}")
    elif outcome.max_late_ms > settings.freeze_limit_ms:
        _label(outcome, LAB_PROBLEM, f"the lab fell behind by {outcome.max_late_ms:.0f} ms")
    elif outcome.race and outcome.race.get("hit") is False and outcome.race.get("armed_ms") is not None:
        # The moment came, but the line missed it: our timing, not the robot.
        _label(outcome, LAB_PROBLEM, f"race line missed its moment: {outcome.race.get('why')}")
    # A race never armed means the robot never did the action: scored as usual.


def _label(outcome: CallOutcome, label: str, why: str) -> CallOutcome:
    outcome.label, outcome.why = label, why
    return outcome


class RobotProcess:
    """The robot's program, started and read from worker threads.

    asyncio's own subprocess start blocks the event loop while Windows starts
    the process, and that loop beats 20 ms pieces for every other live call:
    on 1 Nov 2026 each robot start froze the live calls by ~0.26 s. Started
    from a thread, the loop never waits for it."""

    def __init__(self, process: subprocess.Popen, loop: asyncio.AbstractEventLoop) -> None:
        self._process = process
        self._lines: asyncio.Queue[str | None] = asyncio.Queue()
        threading.Thread(target=self._read, args=(loop,), daemon=True).start()

    @classmethod
    async def start(cls, command: list[str], stderr: Any) -> RobotProcess:
        process = await asyncio.to_thread(
            subprocess.Popen, command, cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE, stderr=stderr
        )
        return cls(process, asyncio.get_running_loop())

    def _read(self, loop: asyncio.AbstractEventLoop) -> None:
        for raw in self._process.stdout:
            loop.call_soon_threadsafe(self._lines.put_nowait, raw.decode("utf-8", "replace").strip())
        self._process.stdout.close()
        loop.call_soon_threadsafe(self._lines.put_nowait, None)  # end of output

    async def line(self) -> str | None:
        """Its next line of output; None once it has exited."""
        return await self._lines.get()

    async def wait(self) -> int:
        return await asyncio.to_thread(self._process.wait)

    def kill(self) -> None:
        self._process.kill()


async def _wait_ready(robot: RobotProcess, timeout_s: float) -> str | None:
    try:
        async with asyncio.timeout(timeout_s):
            while True:
                text = await robot.line()
                if text is None:
                    return None  # it exited first
                if text.startswith("ready "):
                    return text.removeprefix("ready ")
    except TimeoutError:
        return None


async def _stop(robot: RobotProcess, settings: LabSettings) -> tuple[str, int | None]:
    """Let the robot finish on its own (a silent drop takes its no-audio rule
    to notice), then make sure it's gone."""
    end = ""
    wait_s = settings.robot_no_audio_s + settings.robot_change_wait_s + 10
    try:
        async with asyncio.timeout(wait_s):
            while (text := await robot.line()) is not None:
                if text.startswith("ended "):
                    end = text.removeprefix("ended ")
            code = await robot.wait()
    except TimeoutError:
        robot.kill()
        await robot.wait()
        return "killed: it never ended", None
    return end, code


class _Heartbeat:
    """How late this program's event loop runs (L4): a late beat means our
    laptop fell behind, which would make the robot look slower than it is.
    Also notes when the worst beat was, on the call clock, to find its cause."""

    def __init__(self, now_ms, every_s: float = 0.01) -> None:
        self._every_s = every_s
        self._now_ms = now_ms
        self.max_late_ms = 0.0
        self.worst_at_ms = 0

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        due = loop.time() + self._every_s
        while True:
            await asyncio.sleep(max(0.0, due - loop.time()))
            late = (loop.time() - due) * 1000
            if late > self.max_late_ms:
                self.max_late_ms, self.worst_at_ms = late, self._now_ms()
            due = max(due + self._every_s, loop.time())


def _backup(source: Path, target: Path) -> None:
    # SQLite's own backup, not a file copy: correct even with data in the WAL file.
    with sqlite3.connect(source) as src, sqlite3.connect(target) as dst:
        src.backup(dst)
    dst.close()
    src.close()


def _about(plan: CallPlan, about_caller: dict[str, Any]) -> dict[str, Any]:
    caller = plan.task.caller
    return {
        "call_id": plan.call_id,
        "task_id": plan.task.id,
        "robot": plan.robot.name,
        "mode": plan.mode,
        "seed": plan.seed,
        "attempt": plan.attempt,
        "caller": asdict(caller) if caller is not None else None,
        **about_caller,
        "race": asdict(plan.task.race) if plan.task.race else None,
        "drop": asdict(plan.task.drop) if plan.task.drop else None,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
