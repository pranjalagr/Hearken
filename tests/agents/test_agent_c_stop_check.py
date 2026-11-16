"""Agent C's stop-or-keep check (build step 5): the haan-ruko model on the caller's sound
while the robot talks is the only thing that stops the voice; a stop bumps the job
version and stops a ticket being confirmed."""

import asyncio
import time

import numpy as np
import pytest

from agents.agent_c.robot import AgentC, CallParts, Keys
from agents.agent_c.settings import load_settings
from agents.agent_c.stop_check import OnnxStopModel, StopListener
from agents.common.diary import Diary
from agents.common.pipecat_line import LineTransport
from common.wire.protocol import PIECE_BYTES
from engine.contract.spoken import SpokenWords
from packs.food.pack import FoodPack

LOUD = (np.full(PIECE_BYTES // 2, 8000, dtype="<i2")).tobytes()
QUIET = bytes(PIECE_BYTES)


class ListSink:
    def __init__(self) -> None:
        self.records: list[dict] = []

    def write(self, record: dict) -> None:
        self.records.append(record)


class LoudnessModel:
    """Says stop when the window holds loud sound; records each window it saw."""

    def __init__(self):
        self.windows = []

    def chance(self, audio):
        self.windows.append(len(audio))
        return float(np.abs(audio).mean() > 0.1)


async def test_the_model_is_asked_every_tenth_of_a_second_only_while_the_robot_talks():
    model, stops = LoudnessModel(), []

    async def on_stop(chance):
        stops.append(chance)

    listener = StopListener(model, line=0.30, window_s=2.0, every_s=0.1, on_stop=on_stop)
    for _ in range(20):
        await listener.heard(QUIET, robot_talking=False)
    assert model.windows == []  # the robot is quiet: nothing to stop
    for _ in range(10):
        await listener.heard(QUIET, robot_talking=True)
    assert len(model.windows) == 2 and stops == []  # every 5 pieces = 0.1 s
    for _ in range(100):
        await listener.heard(QUIET, robot_talking=True)
    assert max(model.windows) == 16000  # never more than the last 2 s
    for _ in range(100):
        await listener.heard(LOUD, robot_talking=True)
    assert stops == [1.0] and listener.stopped  # once, then quiet until the caller finishes
    listener.turn_finished()
    await asyncio.gather(*(listener.heard(LOUD, robot_talking=True) for _ in range(5)))
    assert stops == [1.0, 1.0]


def tiny_onnx(path, bias: float):
    """A stand-in for the trained model with the same input and output: the stop logit is
    the mean of the log-mel picture plus a bias. Built with onnx alone (no torch here)."""
    import onnx
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [helper.make_node("ReduceMean", ["log_mel"], ["mean"], axes=[1, 2, 3], keepdims=0),
         helper.make_node("Add", ["mean", "bias"], ["stop_logit"])],
        "tiny", [helper.make_tensor_value_info("log_mel", TensorProto.FLOAT, ["batch", 1, 40, 200])],
        [helper.make_tensor_value_info("stop_logit", TensorProto.FLOAT, ["batch"])],
        [helper.make_tensor("bias", TensorProto.FLOAT, [], [bias])])
    # IR version 10: what the onnxruntime in requirements.txt reads (newer onnx writes 14).
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10), str(path))


def test_the_onnx_runner_hears_2_s_pads_a_short_start_and_is_fast(tmp_path):
    tiny_onnx(tmp_path / "m.onnx", bias=0.0)
    model = OnnxStopModel(tmp_path / "m.onnx")
    quiet = model.chance(np.zeros(16000, dtype=np.float32))
    loud = model.chance(np.random.default_rng(1).uniform(-0.5, 0.5, 16000).astype(np.float32))
    short = model.chance(np.zeros(4000, dtype=np.float32))
    assert 0.0 <= quiet < loud <= 1.0 and short == pytest.approx(quiet, abs=1e-3)
    started = time.perf_counter()
    for _ in range(20):
        model.chance(np.zeros(16000, dtype=np.float32))
    assert (time.perf_counter() - started) / 20 < 0.02  # one check fits between two 20 ms pieces


async def test_a_stop_stops_the_voice_bumps_the_version_and_stops_the_ticket():
    call = CallParts(load_settings(), Diary("c1", ListSink()), FoodPack().spoken_words(), stop_model=LoudnessModel())
    stopped, tickets = [], []

    async def stop_voice():
        stopped.append(1)

    class Guard:
        def stop_ticket(self, reason):
            tickets.append(reason)

        def turn_finished(self, words):
            pass

    call.switch.stop_voice = stop_voice
    call.guard = Guard()
    call.desk.start_thinker = lambda job, desk: None
    call.desk.open_job("cancel 4521")
    call.speech.speaking = True
    for _ in range(100):
        await call.heard_piece(LOUD)
    assert stopped == [1] and call.desk.job.version == 2 and tickets
    call.caller_finished("ruko, address bhi badalna hai")
    assert not call.stop_listener.stopped


def test_with_the_model_the_callers_sound_never_interrupts_by_itself():
    class FakeLine:
        async def send_audio(self, piece): ...

        async def clear(self): ...

    settings = load_settings()
    robot = AgentC(settings, Keys("g", "s"), None, stop_model=LoudnessModel())
    parts = CallParts(settings, Diary("c1", ListSink()), SpokenWords(), stop_model=LoudnessModel())
    pipeline = robot.pipeline(LineTransport(FakeLine()), parts)
    [user] = [p for p in pipeline.processors if type(p).__name__ == "LLMUserAggregator"]
    starts = user._params.user_turn_strategies.start
    assert starts and all(strategy._enable_interruptions is False for strategy in starts)
    assert parts.switch in pipeline.processors


def test_without_a_model_file_agent_c_says_so_and_falls_back_to_pipecats_own_stop():
    import dataclasses

    sink = ListSink()
    # A path that is never there: on the owner's laptop the real model file exists (16 Nov 2026).
    settings = load_settings()
    raw = {**settings.raw, "stop_check": {**settings.section("stop_check"), "model_path": "runs/no_such_dir/none.onnx"}}
    robot = AgentC(dataclasses.replace(settings, raw=raw), Keys("g", "s"), None, diary=Diary("c1", sink))
    assert robot._stop_model is None and "no stop-or-keep model" in sink.records[0]["text"]


async def test_agent_cs_own_pipeline_parts_let_the_call_start_and_frames_through():
    # 16 Nov 2026: the stop switch forwarded nothing, so the StartFrame never reached the
    # end of the pipeline and Agent C's first real call never began. Every part tested
    # alone had passed; this runs Agent C's own parts together, as a call does.
    from pipecat.frames.frames import TextFrame
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.tests.utils import run_test

    from agents.agent_c.robot import CallParts
    from agents.agent_c.talker.mouth import Mouth
    from engine.contract.spoken import SpokenWords

    call = CallParts(load_settings(), Diary("c1", ListSink()), SpokenWords())
    parts = Pipeline([call.switch, Mouth(call.speaker, LLMContext(), Diary("c1", ListSink()))])
    # A time limit: a part that swallows frames makes run_test wait forever, not fail.
    await asyncio.wait_for(run_test(parts, frames_to_send=[TextFrame("hello")], expected_down_frames=[TextFrame]), 10)
