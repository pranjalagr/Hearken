"""Pipecat on our phone wire: whole pieces, real speed, "clear" on interruption."""

import asyncio
import time

from pipecat.frames.frames import EndFrame, InputAudioRawFrame, InterruptionFrame, OutputAudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.tests.utils import SleepFrame, run_test
from websockets.exceptions import ConnectionClosedOK

from agents.common.pipecat_line import LEAD_S, LineOutput, LineTransport, line_params, pieces
from common.wire.protocol import PIECE_BYTES, SAMPLE_RATE

LINE_RATES = PipelineParams(audio_in_sample_rate=SAMPLE_RATE, audio_out_sample_rate=SAMPLE_RATE)


class FakeLine:
    def __init__(self) -> None:
        self.sent: list[bytes] = []
        self.times: list[float] = []
        self.cleared = 0

    async def send_audio(self, piece: bytes) -> None:
        self.sent.append(piece)
        self.times.append(time.monotonic())

    async def clear(self) -> None:
        self.cleared += 1


def _speech(seconds: float) -> bytes:
    return bytes(i % 251 for i in range(int(seconds * SAMPLE_RATE) * 2))


def test_audio_is_cut_into_whole_pieces_with_only_the_last_filled_with_silence():
    cut = pieces(b"\x01" * (PIECE_BYTES * 2 + 10))
    assert [len(p) for p in cut] == [PIECE_BYTES] * 3
    assert cut[2] == b"\x01" * 10 + b"\x00" * (PIECE_BYTES - 10)


async def test_a_reply_goes_out_whole_in_order_at_about_real_speed():
    line = FakeLine()
    audio = _speech(1.0)
    await run_test(LineOutput(line, line_params()), frames_to_send=[OutputAudioRawFrame(audio, SAMPLE_RATE, 1)],
                   pipeline_params=LINE_RATES)
    assert all(len(p) == PIECE_BYTES for p in line.sent)
    assert b"".join(line.sent) == audio  # nothing lost, added or reordered
    # Real speed, less the small lead: Pipecat's own "still speaking" timing depends on it.
    spread = line.times[-1] - line.times[0]
    assert 1.0 - LEAD_S - 0.15 <= spread <= 1.0 + 0.15, spread


async def test_an_interruption_clears_the_caller_queue_and_stops_the_rest_of_the_reply():
    line = FakeLine()
    await run_test(LineOutput(line, line_params()),
                   frames_to_send=[OutputAudioRawFrame(_speech(3.0), SAMPLE_RATE, 1), SleepFrame(0.5), InterruptionFrame()],
                   pipeline_params=LINE_RATES)
    assert line.cleared >= 1
    assert len(line.sent) < 3.0 / 0.02 / 2, len(line.sent)  # well under the 150 pieces of the full reply


async def test_a_hung_up_line_ends_writing_without_an_error():
    class GoneLine(FakeLine):
        async def send_audio(self, piece: bytes) -> None:
            raise ConnectionClosedOK(None, None)

    output = LineOutput(GoneLine(), line_params())
    assert await output.write_audio_frame(OutputAudioRawFrame(_speech(0.1), SAMPLE_RATE, 1)) is False


async def test_caller_pieces_enter_the_pipeline_as_8k_audio():
    transport = LineTransport(FakeLine())
    got: list[InputAudioRawFrame] = []

    class Collect(FrameProcessor):
        async def process_frame(self, frame, direction: FrameDirection) -> None:
            await super().process_frame(frame, direction)
            if isinstance(frame, InputAudioRawFrame):
                got.append(frame)
            await self.push_frame(frame, direction)

    task = PipelineTask(Pipeline([transport.input(), Collect()]), params=LINE_RATES)
    running = asyncio.create_task(PipelineRunner(handle_sigint=False).run(task))
    for n in range(3):
        await transport.input().feed(bytes([n]) * PIECE_BYTES)  # waits for the pipeline to start
    await asyncio.sleep(0.2)
    await task.queue_frame(EndFrame())
    await asyncio.wait_for(running, 5)
    assert [f.audio[0] for f in got] == [0, 1, 2] and {f.sample_rate for f in got} == {SAMPLE_RATE}
