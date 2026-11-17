"""A whole Agent C call through the robot's own wiring (17 Nov 2026): the real call box, both
graphs, the real Guard and the real food backend. Only the network parts are stand-ins:
the AIs and the judge are scripts, and the voice "plays" each line at once and sends the
caller's "played" receipt back, as the caller's side does. So the yes moment runs as in a
real call: the question, its mark, the receipt, then a "haan" that counts.
"""

from __future__ import annotations

import asyncio

from pipecat.frames.frames import TTSSpeakFrame

from agents.agent_c.notebook import NEED_YES, RESULT
from agents.agent_c.robot import AgentC, CallParts, Keys, thinker_tools
from agents.agent_c.settings import load_settings
from agents.common.call_box import CallBox
from agents.common.diary import Diary
from agents.common.pipecat_brain_cap import BrainCap
from common.wire.protocol import Played
from engine.service.tool_catalog import build_catalog
from packs.food.pack import FoodPack
from test_talker_graph import ScriptedTalkerAI, finder, order  # pytest puts this folder on the path
from test_thinker_graph import ScriptedAI


class ListSink:
    def __init__(self) -> None:
        self.records: list[dict] = []

    def write(self, record: dict) -> None:
        self.records.append(record)


class BackendTools:
    """The real backend as the robot's tools, with its catalog."""

    def __init__(self, kit) -> None:
        self.kit, self.calls = kit, []

    async def call(self, name, inputs):
        self.calls.append(name)
        return await self.kit.backend.call_tool(name, inputs)

    async def catalog(self):
        return build_catalog(FoodPack().actions(), self.kit.settings)


class YesJudge:
    """The one-word judge, scripted: "haan" is yes."""

    async def label(self, question, words):
        return "yes" if "haan" in words.lower() else "other"


class PlayingVoice:
    """Stands in for the voice and the caller's side: each line "plays" at once (the speech
    watch reports it said), and a mark sent down the line comes back as played."""

    def __init__(self, call: CallParts) -> None:
        self.call, self.said = call, []

    async def queue_frames(self, frames):
        for frame in frames:
            if isinstance(frame, TTSSpeakFrame):
                self.said.append(frame.text)
                asyncio.get_running_loop().call_soon(self.call.speech_ended, False, frame.text)


class Line:
    def __init__(self, call: CallParts) -> None:
        self.call, self.ended = call, []

    async def send_mark(self, name):
        self.call.caller_ms += 10
        self.call.receipt(Played(name, self.call.caller_ms))

    async def end(self, reason):
        self.ended.append(reason)


async def test_a_whole_cancel_call_through_the_robots_wiring(open_call):
    kit = await open_call(FoodPack())
    tools = BackendTools(kit)
    opening = "Mujhe order 4521 cancel karna hai"
    robot = AgentC(load_settings(), Keys("g", "s"), tools, diary=Diary("c1", ListSink()), words=FoodPack().spoken_words(),
                   door=kit.guard, judge=YesJudge(), stop_model=object(),
                   talker_ai=ScriptedTalkerAI(no_job=[("open_job", [])], collect=[("heard_number", [])],
                                              job_open=[("reply", ["Ji, theek hai."])]),
                   thinker_ai=ScriptedAI(check_number=[{"exit": "ok", "number": "4521"}], work_left=[{"exit": "done"}],
                                         decide=[{"exit": "prepare", "tool": "prepare_cancel_order", "inputs": {},
                                                  "plan": "Order cancel kar raha hoon."}]),
                   find_numbers=finder(**{opening: order("4521")}))
    call = CallParts(load_settings(), Diary("c1", ListSink()), FoodPack().spoken_words())
    call.stop_listener = None
    voice, line = PlayingVoice(call), Line(call)

    async def reached(n):
        raise AssertionError("over the brain cap")

    async def wait_for(kind: str) -> None:
        for _ in range(500):
            if any(note.kind == kind for note in call.notebook.notes):
                return
            await asyncio.sleep(0.01)
        raise AssertionError(f"no {kind} note; notes: {[n.kind for n in call.notebook.notes]}")

    async def caller_turn(words: str) -> None:
        call.caller_ms += 1000  # the caller starts well after the last line played
        call.caller_began()
        call.caller_finished(words)

    async def main(box: CallBox) -> None:
        robot.wire(call, box, line, voice, BrainCap(40, reached), thinker_tools(await tools.catalog()))
        box.start("talker", call.runner.run())
        await asyncio.sleep(0.05)
        await caller_turn(opening)
        await wait_for(NEED_YES)
        # The Guard asks once the robot is quiet (its own turn, 17 Nov 2026); the caller answers
        # after hearing it, as a real one does: a "haan" before it never counts (decision 22).
        for _ in range(800):
            if any(text.endswith("Haan ya nahi?") for text in voice.said):
                break
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.2)  # its mark comes back
        await caller_turn("Haan, kar dijiye")
        await wait_for(RESULT)
        await asyncio.sleep(0.3)
        await caller_turn("Nahi bas, thank you")
        await asyncio.sleep(3)  # the Talker says goodbye and ends the call (the box ends main)

    result = await asyncio.wait_for(CallBox().run(main), 15)
    assert result.reason == "the caller said goodbye" and line.ended == ["the caller said goodbye"], result.error
    assert tools.calls == ["find_order", "prepare_cancel_order", "confirm"]
    assert (await kit.one("SELECT status FROM orders WHERE order_no = 4521"))["status"] == "cancelled"
    question = await kit.guard.question(call.notebook.notes[[n.kind for n in call.notebook.notes].index(NEED_YES)]
                                        .data["data"]["token"], "hinglish")
    assert question in voice.said  # the Guard asked the backend's own yes-question
    assert voice.said[-1] == call.words.phrase("talk", "goodbye", "hinglish")
