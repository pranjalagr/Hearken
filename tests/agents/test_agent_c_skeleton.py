"""Agent C's skeleton (build step 1): started like Agent A, the Talker holds only
ask_thinker, and the Guard's door is nowhere the AI can reach."""

import ast
from pathlib import Path

from pipecat.frames.frames import TTSSpeakFrame

from agents.agent_c.robot import AgentC, CallParts, Keys, talker_tools
from agents.agent_c.settings import load_settings
from agents.agent_c.talker.ask_thinker import ASK_THINKER, AskThinker
from agents.common.diary import Diary
from engine.contract.spoken import SpokenWords
from agents.common.pipecat_line import LineTransport

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
    async def call(self, tool, inputs):
        raise AssertionError("the skeleton's Talker must never reach the backend")

    async def catalog(self):
        return []


def test_settings_load_with_the_same_ears_voice_and_greeting_as_agent_a():
    import yaml

    c = load_settings()
    a = yaml.safe_load((ROOT / "agents" / "agent_a" / "settings.yaml").read_text(encoding="utf-8"))
    assert c.greeting == a["greeting"] and c.ears == a["ears"] and c.voice == a["voice"]
    assert set(c.models) == {"talker", "thinker", "ears", "voice"}
    assert "ask_thinker" in c.talker_prompt and "[amount]" in c.talker_prompt  # role, then style guide


def test_the_talker_holds_exactly_ask_thinker_and_end_call():
    # end_call added 16 Nov 2026: without it a solved call ran on until the caller gave up.
    from agents.common.pipecat_hangup import END_CALL

    assert [tool.name for tool in talker_tools().standard_tools] == [ASK_THINKER, END_CALL]


async def test_the_talker_may_not_hang_up_while_the_back_office_works():
    from agents.common.pipecat_hangup import END_CALL

    sink = ListSink()
    call = CallParts(load_settings(), Diary("c1", sink), SpokenWords())

    class Brain:
        def __init__(self):
            self.handlers, self.pushed = {}, []

        def register_function(self, name, handler, **_):
            self.handlers[name] = handler

        async def push_frame(self, frame, *args):
            self.pushed.append(frame)

    class Params:
        def __init__(self, brain):
            self.function_name, self.arguments, self.llm, self.results = END_CALL, {"reason": "bye"}, brain, []

        async def result_callback(self, result, *, properties=None):
            self.results.append(result)

    brain = Brain()
    call.hang_up.connect(brain)
    call.desk.start_thinker = lambda job, desk: None  # a Thinker that never answers
    call.desk.open_job("cancel my order")  # the Thinker is still working
    params = Params(brain)
    await brain.handlers[END_CALL](params)
    assert params.results[0]["status"] == "refused" and call.hang_up.reason is None and brain.pushed == []
    call.desk.job.state = "answer given"
    call.desk.job.change_sent()  # a confirm still saving
    await brain.handlers[END_CALL](Params(brain))
    assert call.hang_up.reason is None and brain.pushed == []
    call.desk.job.change_answered()
    await brain.handlers[END_CALL](Params(brain))
    assert call.hang_up.reason == "bye" and len(brain.pushed) == 1  # the end frame, after the goodbye


def test_the_pipeline_builds_without_a_network_and_in_agent_as_order():
    robot = AgentC(load_settings(), Keys("fake-gemini", "fake-sarvam"), FakeTools())
    pipeline = robot.pipeline(LineTransport(FakeLine()), CallParts(load_settings(), Diary("c1", ListSink()), SpokenWords()))
    names = [type(p).__name__ for p in pipeline.processors if type(p).__name__ not in ("PipelineSource", "PipelineSink")]
    assert names[0] == "LineInput" and names[-1] == "LLMAssistantAggregator", names
    assert names.index("GoogleLLMService") < names.index("SarvamTTSService") < names.index("LineOutput"), names


async def test_ask_thinker_opens_a_job_and_answers_at_once_for_the_filler_line():
    class Desk:
        def __init__(self):
            self.requests = []

        def open_job(self, request):
            self.requests.append(request)
            return "started"

    class LLM:
        def register_function(self, name, handler, *, cancel_on_interruption=True, **_):
            self.name, self.handler, self.cancel = name, handler, cancel_on_interruption

    class Params:
        arguments = {"request": "cancel order 4521, Rahul"}

        def __init__(self):
            self.results = []

        async def result_callback(self, result, **_):
            self.results.append(result)

    desk, llm, sink, params = Desk(), LLM(), ListSink(), Params()
    AskThinker(desk, Diary("c1", sink)).connect(llm)
    assert llm.name == ASK_THINKER and llm.cancel is False  # a caller cutting in never half-opens a job
    await llm.handler(params)
    assert desk.requests == ["cancel order 4521, Rahul"] and params.results == [{"status": "started"}]
    assert [r["what"] for r in sink.records if r["kind"] == "decision"] == ["ask thinker"]


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


def test_the_greeting_is_a_fixed_line_not_a_brain_request():
    # Same opening as Agent A: queued as a TTSSpeakFrame, so no Talker request is spent.
    source = (ROOT / "agents" / "agent_c" / "robot.py").read_text(encoding="utf-8")
    assert "TTSSpeakFrame(self._settings.greeting)" in source and TTSSpeakFrame


def test_agent_c_never_asks_gemini_for_a_deadline_under_its_minimum(monkeypatch):
    # 16 Nov 2026: the Thinker's client was made with its 8 s step limit, and Gemini refused
    # every request ("Minimum allowed deadline is 10s"); the 1 s judge would have too.
    import agents.agent_c.robot as robot_module
    from agents.common.pipecat_brain_cap import BrainCap
    from common.google import GOOGLE_MIN_TIMEOUT_S

    asked: list[float] = []
    monkeypatch.setattr(robot_module, "google_client", lambda key, timeout_s: asked.append(timeout_s) or object())
    monkeypatch.setattr(robot_module, "GeminiThinker", lambda *args, **kwargs: object())
    robot = AgentC(load_settings(), Keys("g", "s"), FakeTools(), stop_model=object())

    async def reached(n):
        pass

    robot._yes_judge(BrainCap(40, reached))
    robot._gemini_thinker([])
    assert len(asked) == 2 and min(asked) >= GOOGLE_MIN_TIMEOUT_S
