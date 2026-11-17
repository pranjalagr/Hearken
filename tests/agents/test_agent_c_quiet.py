"""Waiting lines, the silent caller and pre-made clips, for the two graphs (step 6, 17 Nov 2026)."""

from __future__ import annotations

import asyncio
import dataclasses

from pipecat.frames.frames import TTSAudioRawFrame, TTSSpeakFrame, TTSStartedFrame, TTSStoppedFrame, TTSTextFrame

from agents.agent_c.notebook import NEED_YES, RESULT
from agents.agent_c.robot import AgentC, CallParts, Keys, thinker_tools
from agents.agent_c.settings import load_settings
from agents.agent_c.talker.lines import ClipOut
from agents.agent_c.talker.quiet import QuietSettings
from agents.agent_c.talker.quiet_watch import CallView, QuietWatch
from agents.common.call_box import CallBox
from agents.common.diary import Diary
from agents.common.pipecat_brain_cap import BrainCap
from packs.food.pack import FoodPack
from test_agent_c_wiring import BackendTools, Line, ListSink, PlayingVoice, YesJudge
from test_talker_graph import ScriptedTalkerAI, finder, order
from test_thinker_graph import ScriptedAI

SETTINGS = QuietSettings(first_s=1.0, every_s=4.0, hello_s=6.0, line_gone_s=15.0)
WAITING = ["dekh_raha", "check_kar_raha", "line_par_rahiye", "thoda_samay"]


class State:
    """A made-up call, changed by the test between ticks."""

    def __init__(self) -> None:
        self.robot = self.caller = self.saving = self.asking = False
        self.working = True
        self.now = 0.0

    def view(self) -> CallView:
        return CallView(lambda: self.robot, lambda: self.caller, lambda: self.working, lambda: self.saving,
                        lambda: self.asking)


def watch(state: State) -> tuple[QuietWatch, list, list]:
    said, ended = [], []

    async def say_line(group, value):
        said.append((group, value))

    async def say_text(text):
        said.append(("plan", text))

    async def end_call(reason):
        ended.append(reason)

    return QuietWatch(state.view(), SETTINGS, WAITING, say_line, say_text, end_call, 0.1, clock=lambda: state.now), said, ended


async def run_until(state: State, w: QuietWatch, until: float, step: float = 0.1) -> None:
    while state.now < until:
        state.now = round(state.now + step, 3)
        await w.tick()


# ---- waiting lines (decision 11) ------------------------------------------------------------

async def test_waiting_lines_come_after_1_s_then_every_4_s_never_twice_in_a_row():
    state = State()
    w, said, _ = watch(state)
    await run_until(state, w, 0.9)
    assert said == []
    await run_until(state, w, 1.2)
    assert len(said) == 1 and said[0][0] == "waiting"
    await run_until(state, w, 4.0)
    assert len(said) == 1  # not yet: 4 s apart, start to start
    await run_until(state, w, 12.0)  # lines at ~1, ~5 and ~9 s
    assert len(said) == 3 and all(a != b for a, b in zip(said, said[1:]))


async def test_the_plan_takes_a_waiting_lines_place_and_keeps_the_pace():
    state = State()
    w, said, _ = watch(state)
    await run_until(state, w, 1.2)  # a waiting line at ~1 s
    state.now = 2.0
    w.hold_plan("Dekh raha hoon order nikla ya nahi.")
    await run_until(state, w, 2.1)
    assert said[-1] == ("plan", "Dekh raha hoon order nikla ya nahi.")  # as soon as the robot is quiet
    await run_until(state, w, 5.9)
    assert len(said) == 2  # the next waiting line keeps its pace after the plan
    await run_until(state, w, 6.5)
    assert said[-1][0] == "waiting"


async def test_no_waiting_line_while_the_thinker_waits_on_the_caller_or_the_guard_asks():
    state = State()
    w, said, _ = watch(state)
    state.working = False  # e.g. collecting the order number
    await run_until(state, w, 3)
    state.working, state.asking = True, True  # the yes-question is open
    await run_until(state, w, 6)
    assert [line for line in said if line[0] == "waiting"] == []


# ---- the silent caller (decision 13) --------------------------------------------------------

async def test_a_silent_caller_hears_hello_then_line_gone_and_the_call_ends():
    state = State()
    state.working = False
    w, said, ended = watch(state)
    state.robot = True
    await run_until(state, w, 1)  # the robot spoke
    state.robot = False
    await run_until(state, w, 7.5)  # ~6 s of quiet after the robot's line
    assert said == [("silent", "hello")]
    state.robot = True  # the hello plays
    await run_until(state, w, 8)
    state.robot = False
    await run_until(state, w, 24)  # ~15 s more
    assert said == [("silent", "hello"), ("silent", "line_gone")] and ended == ["the caller went silent"]


async def test_no_silent_caller_lines_while_the_thinker_works():
    state = State()
    w, said, ended = watch(state)
    state.robot = True
    await run_until(state, w, 1)
    state.robot = False
    await run_until(state, w, 30)
    assert [line for line in said if line[0] == "silent"] == [] and ended == []


# ---- clips ----------------------------------------------------------------------------------

class Pushed:
    def __init__(self) -> None:
        self.frames, self.queued = [], []

    async def push(self, frame, direction=None):
        self.frames.append(frame)

    async def queue_frames(self, frames):
        self.queued.extend(frames)


async def test_a_line_with_a_clip_plays_it_with_its_words_and_others_go_live():
    sound = b"\x00\x01" * 400
    call = CallParts(load_settings(), Diary("c1", ListSink()), FoodPack().spoken_words(),
                     clips={"Ji, ek second.": sound})
    out = Pushed()
    call.clip_out.push_frame = out.push
    call.task = out
    await call.say("Ji, ek second.")
    await call.say("4521, dekh raha hoon.")
    # Sent as a voice service sends a line, so the output counts it as robot speech.
    assert [type(f) for f in out.frames] == [TTSStartedFrame, TTSAudioRawFrame, TTSTextFrame, TTSStoppedFrame]
    assert out.frames[2].text == "Ji, ek second."
    assert [f.text for f in out.queued if isinstance(f, TTSSpeakFrame)] == ["4521, dekh raha hoon."]


async def test_the_clip_player_lets_every_frame_through():
    # A processor that forwards nothing swallows the StartFrame and the call never starts
    # (the stop switch did, 16 Nov 2026).
    from pipecat.frames.frames import TextFrame
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.tests.utils import run_test

    await asyncio.wait_for(run_test(Pipeline([ClipOut()]), frames_to_send=[TextFrame("hi")],
                                    expected_down_frames=[TextFrame]), 10)


# ---- a whole call with waiting lines and a silent caller -------------------------------------

class SlowAI:
    """The scripted Thinker AI, taking `seconds` to decide, as a slow Flash-Lite would."""

    def __init__(self, ai, seconds) -> None:
        self.ai, self.seconds = ai, seconds

    async def ask(self, step, facts):
        if step == "decide":
            await asyncio.sleep(self.seconds)
        return await self.ai.ask(step, facts)


async def test_a_whole_call_with_waiting_lines_a_plan_and_a_silent_caller(open_call):
    kit = await open_call(FoodPack())
    tools = BackendTools(kit)
    raw = dict(load_settings().raw)
    # The same rules, shortened so the test runs in seconds.
    raw.update(speed={"tick_s": 0.02, "reply_grace_s": 0.1, "question_wait_s": 2.0, "question_gap_s": 0.1},
               waiting_lines={**raw["waiting_lines"], "first_s": 0.2, "every_s": 0.5},
               silent_caller={"hello_s": 0.5, "line_gone_s": 0.5}, hand_over_end={"wait_s": 1.0, "tries": 2})
    settings = dataclasses.replace(load_settings(), raw=raw)
    opening = "Mujhe order 4521 cancel karna hai"
    thinker = ScriptedAI(check_number=[{"exit": "ok", "number": "4521"}],
                         decide=[{"exit": "prepare", "tool": "prepare_cancel_order", "inputs": {},
                                  "plan": "Order cancel kar raha hoon."}])
    robot = AgentC(settings, Keys("g", "s"), tools, diary=Diary("c1", ListSink()), words=FoodPack().spoken_words(),
                   door=kit.guard, judge=YesJudge(), stop_model=object(),
                   talker_ai=ScriptedTalkerAI(no_job=[("open_job", [])], collect=[("heard_number", [])]),
                   thinker_ai=SlowAI(thinker, 1.6), find_numbers=finder(**{opening: order("4521")}))
    call = CallParts(settings, Diary("c1", ListSink()), FoodPack().spoken_words())
    voice, line = PlayingVoice(call), Line(call)

    async def reached(n):
        raise AssertionError("over the brain cap")

    async def main(box: CallBox) -> None:
        robot.wire(call, box, line, voice, BrainCap(40, reached), thinker_tools(await tools.catalog()))
        box.start("talker", call.runner.run())
        box.start("quiet", call.quiet.run())
        await asyncio.sleep(0.05)
        call.caller_ms += 1000
        call.caller_began()
        call.caller_finished(opening)
        for _ in range(800):  # the caller answers once the yes-question has been asked
            if any(text.endswith("Haan ya nahi?") for text in voice.said):
                break
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.2)
        call.caller_ms += 1000
        call.caller_began()
        call.caller_finished("Haan, kar dijiye")
        await asyncio.sleep(10)  # then silence: hello, line gone, and code ends the call

    result = await asyncio.wait_for(CallBox().run(main), 20)
    words = call.words
    waiting = {words.phrase("waiting", name, "hinglish") for name in words.values("waiting")}
    said = voice.said
    # A plan is never said once its work moved on: before the yes-question, or not at all
    # (here the tool answers at once, so the plan is old news before the robot is quiet).
    question_at = next(i for i, text in enumerate(said) if text.endswith("Haan ya nahi?"))
    assert all(i < question_at for i, text in enumerate(said) if text == "Order cancel kar raha hoon.")
    assert any(text in waiting for text in said)  # a waiting line while the Thinker decided
    assert any(n.kind == RESULT for n in call.notebook.notes)
    assert said[-2:] == [words.phrase("silent", "hello", "hinglish"), words.phrase("silent", "line_gone", "hinglish")]
    assert result.reason == "the caller went silent" and line.ended == ["the caller went silent"]
