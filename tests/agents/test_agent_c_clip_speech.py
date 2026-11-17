"""A clip line counts as the robot speaking, end to end through the real line output
(17 Nov 2026, food_08: plain audio frames were never seen as robot speech, so a clip line's
start and end were invisible to the Guard and to the call's ending)."""

import asyncio

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask

from agents.agent_c.talker.lines import ClipOut
from agents.agent_c.talker.speech_watch import SpeechWatch
from agents.common.pipecat_line import LineTransport
from common.wire.protocol import SAMPLE_RATE


class Line:
    def __init__(self) -> None:
        self.sent = 0

    async def send_audio(self, piece: bytes) -> None:
        self.sent += len(piece)

    async def clear(self) -> None: ...


async def test_a_clip_line_is_seen_starting_and_ending_with_its_words():
    line, ended = Line(), []
    transport = LineTransport(line)
    clip_out = ClipOut()
    watch = SpeechWatch(lambda: None, lambda cut, words: ended.append((cut, words)))
    watch.output = transport.output()
    task = PipelineTask(Pipeline([clip_out, transport.output()]),
                        params=PipelineParams(audio_out_sample_rate=SAMPLE_RATE), observers=[watch])
    runner = asyncio.create_task(PipelineRunner(handle_sigint=False).run(task))
    await asyncio.sleep(0.3)
    sound = b"\x10\x00" * (SAMPLE_RATE // 2)  # half a second of sound
    await clip_out.play(sound, "Haan ya nahi?")
    for _ in range(60):
        if ended:
            break
        await asyncio.sleep(0.05)
    await task.cancel()
    await asyncio.wait_for(runner, 5)
    assert ended == [(False, "Haan ya nahi?")] and line.sent >= len(sound)
