"""Agent C's robot, as two graphs (17 Nov 2026): started like Agent A, no brain inside the
pipeline, the Guard's door nowhere an AI can reach, and the call's wiring between the
pipeline, the Talker graph, the Thinker graph and the Guard."""

import ast
import asyncio
from pathlib import Path

from agents.agent_c.notebook import FAILED
from agents.agent_c.robot import AgentC, CallParts, Keys
from agents.agent_c.settings import load_settings
from agents.common.diary import Diary
from agents.common.pipecat_line import LineTransport
from common.wire.protocol import Key
from engine.contract.spoken import SpokenWords

ROOT = Path(__file__).resolve().parents[2]


class ListSink:
    def __init__(self) -> None:
        self.records: list[dict] = []

    def write(self, record: dict) -> None:
        self.records.append(record)


class FakeLine:
    async def send_audio(self, piece: bytes) -> None: ...

    async def clear(self) -> None: ...


class FakeTools:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def call(self, tool, inputs):
        self.calls.append((tool, inputs))

    async def catalog(self):
        return []


def a_call() -> CallParts:
    return CallParts(load_settings(), Diary("c1", ListSink()), SpokenWords())


def test_settings_load_with_the_same_ears_voice_and_greeting_as_agent_a():
    import yaml

    c = load_settings()
    a = yaml.safe_load((ROOT / "agents" / "agent_a" / "settings.yaml").read_text(encoding="utf-8"))
    assert c.greeting == a["greeting"] and c.ears == a["ears"] and c.voice == a["voice"]


def test_the_pipeline_has_no_brain_inside_ears_to_voice():
    # The Talker graph decides what to say; the pipeline only hears and speaks.
    robot = AgentC(load_settings(), Keys("fake-gemini", "fake-sarvam"), FakeTools())
    pipeline = robot.pipeline(LineTransport(FakeLine()), a_call())
    names = [type(p).__name__ for p in pipeline.processors if type(p).__name__ not in ("PipelineSource", "PipelineSink")]
    assert names[0] == "LineInput" and names[-1] == "LineOutput", names
    assert not any("LLMService" in name for name in names), names
    assert names.index("SarvamRealtimeSTTService") < names.index("SarvamTTSService"), names


def test_the_cli_takes_the_labs_arguments():
    from cli.agent_c import parse

    args = parse(["--db", "x.db", "--max-brain-requests", "12", "--change-wait-s", "7"])
    assert (args.db, args.max_brain_requests, args.change_wait_s, args.port) == ("x.db", 12, 7.0, 0)


def test_no_agent_c_code_imports_the_guard_door_except_the_guard():
    # Only plain code holding GuardDoor may get a yes-ticket (decision 7): the AI parts
    # (Talker, Thinker) must not even import it.
    allowed = {"guard.py", "guard_wiring.py"}
    for path in (ROOT / "agents" / "agent_c").rglob("*.py"):
        if path.name in allowed:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        assert "engine.safety.guard_door" not in imported, path


def test_the_greeting_is_the_settings_fixed_line():
    # Same opening as Agent A, said by the Talker graph's first step: no brain request.
    source = (ROOT / "agents" / "agent_c" / "robot.py").read_text(encoding="utf-8")
    assert "self._settings.greeting" in source


def test_agent_c_never_asks_gemini_for_a_deadline_under_its_minimum(monkeypatch):
    # 16 Nov 2026: the Thinker's client was made with its 8 s step limit, and Gemini refused
    # every request ("Minimum allowed deadline is 10s"); the 1 s judge would have too.
    import agents.agent_c.robot as robot_module
    from agents.common.pipecat_brain_cap import BrainCap
    from common.google import GOOGLE_MIN_TIMEOUT_S

    asked: list[float] = []
    monkeypatch.setattr(robot_module, "google_client", lambda key, timeout_s: asked.append(timeout_s) or object())
    robot = AgentC(load_settings(), Keys("g", "s"), FakeTools(), stop_model=object())

    async def reached(n):
        pass

    robot._yes_judge(BrainCap(40, reached))
    robot._gemini_thinker()
    robot._gemini_talker()
    assert len(asked) == 3 and min(asked) >= GOOGLE_MIN_TIMEOUT_S


def test_with_several_keys_the_talker_keeps_one_and_the_back_office_takes_turns_over_the_rest(monkeypatch):
    # The owner's choice, 16 Nov 2026: one key went over Gemini's 15 requests a minute.
    import agents.agent_c.robot as robot_module

    made: list[str] = []

    class Client:
        def __init__(self, key):
            made.append(key)
            self.aio = key

    monkeypatch.setattr(robot_module, "google_client", lambda key, timeout_s: Client(key))
    keys = Keys("k1", "s", gemini_pool=("k1", "k2", "k3"))
    assert keys.back_office() == ["k2", "k3"] and Keys("k1", "s").back_office() == ["k1"]
    robot = AgentC(load_settings(), keys, FakeTools(), stop_model=object())
    ring = robot._ring(10.0)
    assert made == ["k2", "k3"] and [ring.aio for _ in range(3)] == ["k2", "k3", "k2"]


async def test_the_last_line_is_said_again_if_cut_then_code_ends_the_call():
    # 16 Nov 2026, food_05: the caller spoke over the hand-over line, the cut dropped the
    # end frame queued behind it, and the call stayed open.
    call = a_call()
    line_text = "Main aapko hamare team member se connect kar raha hoon."
    queued, ended = [], []

    class Task:
        async def queue_frames(self, frames):
            queued.append(frames[0].text)
            # The first time the caller cuts it off; the second time it plays to the end.
            asyncio.get_running_loop().call_soon(call.speech_ended, len(queued) == 1, line_text)

    class Line:
        async def end(self, reason):
            ended.append(("line", reason))

    class Box:
        def end(self, reason):
            ended.append(("box", reason))

        def start(self, name, work):
            return asyncio.ensure_future(work)

    call.task, call.line, call.box = Task(), Line(), Box()
    await call.say(line_text)
    await asyncio.wait_for(call.end_call("handed_over"), 5)
    spoken = "Mai aapko hamare team member se connect kar raha hoon."  # spelled for the voice (2 Nov 2026)
    assert queued == [spoken, spoken] and ended == [("line", "handed_over"), ("box", "handed_over")]


async def test_the_hand_over_presses_the_tool_once_and_tells_the_talker():
    call, tools = a_call(), FakeTools()
    told: list[str] = []

    class Runner:
        def hand_over(self, reason):
            told.append(reason)

    call.tools, call.runner = tools, Runner()
    await call.hand_over("the caller asked for a person")
    await call.hand_over("again")  # a second hand-over is ignored
    assert [tool for tool, _ in tools.calls] == ["hand_over_to_human"] and told == ["the caller asked for a person"]


async def test_a_thinker_crash_writes_failed_then_hands_over():
    call, tools = a_call(), FakeTools()
    started: list[asyncio.Future] = []

    class Box:
        def start(self, name, work):
            started.append(asyncio.ensure_future(work))

    class Runner:
        def hand_over(self, reason):
            pass

    call.tools, call.runner, call.box = tools, Runner(), Box()
    call.crashed("thinker", RuntimeError("boom"))
    await asyncio.gather(*started)
    assert call.notebook.notes[-1].kind == FAILED and [tool for tool, _ in tools.calls] == ["hand_over_to_human"]


def test_a_finished_turn_goes_to_the_guard_and_the_talker():
    call = a_call()
    seen: list[tuple[str, str]] = []

    class Guard:
        asking = False

        def turn_finished(self, words):
            seen.append(("guard", words))

    class Runner:
        def turn(self, words):
            seen.append(("talker", words))

    call.guard, call.runner = Guard(), Runner()
    call.caller_finished("4521")
    assert seen == [("guard", "4521"), ("talker", "4521")]


def test_a_key_goes_to_the_guard_during_a_yes_and_to_the_talker_otherwise():
    call = a_call()
    seen: list[tuple[str, str]] = []

    class Guard:
        asking = True

        def key(self, digit, caller_ms):
            seen.append(("guard", digit))

    class Runner:
        def key(self, digit, caller_ms):
            seen.append(("talker", digit))

    call.guard, call.runner = Guard(), Runner()
    call.receipt(Key("1", 100))
    Guard.asking = False
    call.receipt(Key("4", 200))
    assert seen == [("guard", "1"), ("talker", "4")]


async def test_a_line_waits_while_the_caller_is_talking():
    # Decision 8: the robot never speaks over the caller (missed in step 5; run 2, 17 Nov 2026).
    call = a_call()
    queued = []

    class Task:
        async def queue_frames(self, frames):
            queued.append(frames[0].text)

    call.task = Task()
    call.caller_talking = True
    saying = asyncio.ensure_future(call.say("Ji, ek second."))
    await asyncio.sleep(0.2)
    assert queued == []  # held while the caller talks
    call.caller_talking = False
    await asyncio.wait_for(saying, 2)
    assert queued == ["Ji, ek second."]
