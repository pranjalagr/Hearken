"""The Pipecat robots' tool bridge and diary: tools reach the backend as asked,
every call is written down, and what was heard and said is recorded once."""

import asyncio

import pytest
from pipecat.frames.frames import (InterruptionFrame, MetricsFrame, TranscriptionFrame, TTSStartedFrame,
                                   TTSStoppedFrame, TTSTextFrame)
from pipecat.metrics.metrics import LLMTokenUsage, LLMUsageMetricsData, TTSUsageMetricsData
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection

from agents.common.diary import Diary
from agents.common.pipecat_diary import DiaryObserver
from agents.common.pipecat_tools import CANCELLED, connect_tools, tools_schema
from engine.service.replies import ToolReply
from engine.service.tool_catalog import ToolSpec


class ListSink:
    def __init__(self) -> None:
        self.records: list[dict] = []

    def write(self, record: dict) -> None:
        self.records.append(record)


def _kinds(sink: ListSink, kind: str) -> list[dict]:
    return [r for r in sink.records if r["kind"] == kind]


class FakeLLM:
    def __init__(self) -> None:
        self.handler = None
        self.cancel_on_interruption = None

    def register_function(self, name, handler, *, cancel_on_interruption=None, **_):
        assert name is None  # one handler for every tool
        self.handler, self.cancel_on_interruption = handler, cancel_on_interruption


class Params:
    def __init__(self, name, arguments):
        self.function_name, self.arguments, self.results = name, arguments, []

    async def result_callback(self, result):
        self.results.append(result)


def test_the_brain_sees_the_backends_tools_without_additionalproperties():
    schema = tools_schema([ToolSpec("find_order", "Look up an order.", {
        "type": "object", "properties": {"order_number": {"type": "string", "description": "n"}},
        "required": ["order_number"], "additionalProperties": False})])
    [tool] = schema.standard_tools
    assert tool.name == "find_order" and tool.required == ["order_number"]
    assert "additionalProperties" not in tool.properties and set(tool.properties) == {"order_number"}


@pytest.mark.parametrize("background", [False, True])
async def test_a_tool_call_goes_to_the_backend_and_is_written_down(background):
    class Tools:
        async def call(self, tool, inputs):
            return ToolReply("done", "found", {"status": "preparing"}) if (tool, inputs) == ("find_order", {"order_number": "4521"}) else None

    llm, sink = FakeLLM(), ListSink()
    connect_tools(llm, Tools(), Diary("c1", sink), background=background)
    assert llm.cancel_on_interruption is (not background)  # Pipecat's own background switch
    params = Params("find_order", {"order_number": "4521"})
    await llm.handler(params)
    assert params.results == [{"status": "done", "message": "found", "data": {"status": "preparing"}}]
    [line] = _kinds(sink, "tool")
    assert line["name"] == "find_order" and line["inputs"] == {"order_number": "4521"} and line["status"] == "done"


async def test_a_tool_cut_off_by_the_caller_is_written_as_cancelled():
    class SlowTools:
        async def call(self, tool, inputs):
            await asyncio.sleep(10)

    llm, sink = FakeLLM(), ListSink()
    connect_tools(llm, SlowTools(), Diary("c1", sink), background=False)
    running = asyncio.create_task(llm.handler(Params("cancel_order", {"order_number": "4521"})))
    await asyncio.sleep(0.05)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert [line["status"] for line in _kinds(sink, "tool")] == [CANCELLED]


async def test_the_diary_records_heard_said_cut_and_costs_once_each():
    sink = ListSink()
    observer = DiaryObserver(Diary("c1", sink))

    async def push(frame, times=1):
        for _ in range(times):  # a frame passes several processors
            await observer.on_push_frame(FramePushed(source=None, destination=None, frame=frame,
                                                     direction=FrameDirection.DOWNSTREAM, timestamp=0))

    await push(TranscriptionFrame("mera order 4521", "caller", "t"), times=3)
    await push(TTSStartedFrame())
    await push(TTSTextFrame("Ek second,", aggregated_by="sentence"), times=2)
    await push(TTSTextFrame("check karta hoon.", aggregated_by="sentence"))
    await push(TTSStoppedFrame())
    await push(TTSStartedFrame())
    await push(TTSTextFrame("Aapka order", aggregated_by="sentence"))
    await push(InterruptionFrame())  # the caller cut in mid-reply
    await push(MetricsFrame(data=[LLMUsageMetricsData(processor="llm", model="gemini-3.5-flash-lite",
                                                      value=LLMTokenUsage(prompt_tokens=900, completion_tokens=40, total_tokens=940)),
                                  TTSUsageMetricsData(processor="tts", model="bulbul:v4-flash", value=27)]))
    assert [r["text"] for r in _kinds(sink, "heard")] == ["mera order 4521"]
    assert [(r["text"], r["cut"]) for r in _kinds(sink, "said")] == [("Ek second, check karta hoon.", False), ("Aapka order", True)]
    costs = {(r["service"], r["unit"]): r["amount"] for r in _kinds(sink, "cost")}
    assert costs == {("gemini-3.5-flash-lite", "input tokens"): 900, ("gemini-3.5-flash-lite", "output tokens"): 40,
                     ("bulbul:v4-flash", "characters"): 27}


async def test_the_mcp_tool_list_reads_as_tool_specs():
    # Uses the MCP library's own types: on 15 Nov 2026 the first real call crashed on a
    # wrong field name ("inputSchema"; mcp 2.x calls it input_schema).
    from mcp import types

    from agents.common.tools import McpTools

    schema = {"type": "object", "properties": {"order_number": {"type": "string"}}, "required": ["order_number"]}

    class Session:
        async def list_tools(self):
            return types.ListToolsResult(tools=[types.Tool(name="find_order", description="Look up.", input_schema=schema)])

    assert await McpTools(Session()).catalog() == [ToolSpec("find_order", "Look up.", schema)]


async def test_hanging_up_ends_the_pipeline_after_the_goodbye_with_no_extra_brain_request():
    # Decision 13; 15 Nov 2026: with no way to hang up, one call ran to 50 brain requests.
    from pipecat.frames.frames import EndWorkerFrame

    from agents.common.pipecat_hangup import END_CALL, END_CALL_SCHEMA, HangUp

    sink = ListSink()
    hang_up = HangUp(Diary("c1", sink))

    class Brain:
        def __init__(self):
            self.handlers, self.pushed = {}, []

        def register_function(self, name, handler, **_):
            self.handlers[name] = handler

        async def push_frame(self, frame, *args):
            self.pushed.append(frame)

    class HangUpParams(Params):
        async def result_callback(self, result, *, properties=None):
            self.results.append((result, properties))

    brain = Brain()
    hang_up.connect(brain)
    params = HangUpParams(END_CALL, {"reason": "handed over"})
    params.llm = brain
    await brain.handlers[END_CALL](params)
    assert hang_up.reason == "handed over"
    [(result, properties)] = params.results
    assert result == {"status": "ok"} and properties.run_llm is False  # no request after the hang-up
    assert isinstance(brain.pushed[0], EndWorkerFrame)  # sent after the goodbye, so it plays first
    assert [r["what"] for r in _kinds(sink, "decision")] == ["hang up"]
    assert [t.name for t in tools_schema([], (END_CALL_SCHEMA,)).standard_tools] == [END_CALL]


async def test_the_brain_cap_counts_each_request_once_and_ends_the_call_once():
    # 15 Nov 2026: one Agent A call ran to 56 brain requests going round in circles.
    from agents.common.pipecat_brain_cap import BrainCap

    reached: list[int] = []

    async def on_reached(requests):
        reached.append(requests)

    cap = BrainCap(3, on_reached)

    async def push(frame, times=1):
        for _ in range(times):  # a frame passes several processors
            await cap.on_push_frame(FramePushed(source=None, destination=None, frame=frame,
                                                direction=FrameDirection.DOWNSTREAM, timestamp=0))

    def usage():
        return MetricsFrame(data=[LLMUsageMetricsData(processor="llm", model="m", value=LLMTokenUsage(
            prompt_tokens=900, completion_tokens=40, total_tokens=940))])

    await push(usage(), times=3)
    await push(MetricsFrame(data=[TTSUsageMetricsData(processor="tts", model="v", value=27)]))  # not the brain
    await push(usage())
    assert cap.requests == 2 and reached == []
    await push(usage())
    await push(usage())  # usage still arriving while the call ends
    assert reached == [3]


def test_every_ai_robot_is_given_the_labs_brain_cap(tmp_path):
    from lab.runner.run_call import AgentRobot
    from lab.settings import load_settings

    settings = load_settings().with_changes(robot_max_brain_requests=7)
    robot = AgentRobot("agent:a", "cli.agent_a", "safe", ticket_check=False, extra_args=("--background-tools",))

    class Task:
        id = "food_01"

    command = robot.command(tmp_path / "c.db", tmp_path, "http://bells", settings, Task())
    assert command[command.index("--max-brain-requests") + 1] == "7"
    assert command[-1] == "--background-tools"  # the robot's own switches still come last
