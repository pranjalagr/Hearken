"""Pipecat on our phone wire (Part 2 decision 4).

The lab's wire carries 20 ms pieces of 8 kHz audio both ways, plus a few JSON
messages (common/wire/protocol.py). Pipecat has no transport for it, so this is
one: the caller's pieces go into a Pipecat pipeline, and the pipeline's voice
comes back out as pieces.

How fast the robot's audio is sent. Our wire lets a robot send as fast as its
voice service makes audio; the caller's side plays it at real speed (decision
22). Pipecat, though, decides "the robot is speaking" and "the robot stopped"
from how long its output takes to write, as a sound card would. Sent all at
once, Pipecat would think every reply was over at once, and a caller talking
over the rest of it would not count as an interruption. So the output keeps
real speed, holding at most LEAD_S of audio queued at the caller, like a
phone's small playback buffer; on an interruption that queue is dropped with
"clear". Agent A is meant to be Pipecat as it comes (Part 2), and this keeps
Pipecat's own timing intact.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Protocol

from pipecat.frames.frames import Frame, InputAudioRawFrame, InterruptionFrame, OutputAudioRawFrame, StartFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from websockets.exceptions import ConnectionClosed

from common.wire.protocol import PIECE_BYTES, PIECE_MS, SAMPLE_RATE

PIECE_S = PIECE_MS / 1000
# At most this much robot audio waits at the caller: enough to ride over a late
# write, small enough that a "clear" on interruption throws little away.
LEAD_S = 0.2


def line_params() -> TransportParams:
    return TransportParams(
        audio_in_enabled=True,
        audio_in_sample_rate=SAMPLE_RATE,
        audio_out_enabled=True,
        audio_out_sample_rate=SAMPLE_RATE,
        # 2 x 10 ms = one wire piece, so Pipecat hands over whole pieces.
        audio_out_10ms_chunks=PIECE_MS // 10,
        # The wire needs no robot silence: the caller's side plays silence itself
        # when no piece arrives, as with every other robot. Pipecat would otherwise
        # also send 2 s of silence after the end (found by the tests, 15 Nov 2026).
        audio_out_auto_silence=False,
        audio_out_end_silence_secs=0,
    )


class LineOut(Protocol):
    """The robot's sending side of the wire (common/wire/robot_line.RobotLine)."""

    async def send_audio(self, piece: bytes) -> None: ...

    async def clear(self) -> None: ...


def pieces(audio: bytes) -> list[bytes]:
    """Whole 20 ms pieces. The wire refuses any other size, so a short last piece is
    filled with silence: at most 20 ms of quiet at the end of a reply."""
    return [audio[at:at + PIECE_BYTES].ljust(PIECE_BYTES, b"\x00") for at in range(0, len(audio), PIECE_BYTES)]


class LineInput(BaseInputTransport):
    def __init__(self, params: TransportParams, **kwargs) -> None:
        super().__init__(params, **kwargs)
        self._ready = asyncio.Event()

    async def start(self, frame: StartFrame) -> None:
        await super().start(frame)
        await self.set_transport_ready(frame)
        # Pipecat makes its input queue only now; audio pushed before would be lost.
        self._ready.set()

    async def feed(self, piece: bytes) -> None:
        """One caller piece into the pipeline. Waits until the pipeline has started."""
        await self._ready.wait()
        await self.push_audio_frame(InputAudioRawFrame(audio=piece, sample_rate=SAMPLE_RATE, num_channels=1))


class LineOutput(BaseOutputTransport):
    def __init__(self, line: LineOut, params: TransportParams, *, lead_s: float = LEAD_S,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
                 **kwargs) -> None:
        super().__init__(params, **kwargs)
        self._line = line
        self._lead_s = lead_s
        self._clock = clock
        self._sleep = sleep
        self._next_s = 0.0  # when the caller will have played everything sent so far
        self._open = True

    async def start(self, frame: StartFrame) -> None:
        await super().start(frame)
        await self.set_transport_ready(frame)

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, InterruptionFrame):
            # The caller cut in: throw away what is still queued at their end too.
            self._next_s = 0.0
            await self._send(self._line.clear)

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        for piece in pieces(frame.audio):
            await self._pace()
            if not await self._send(lambda piece=piece: self._line.send_audio(piece)):
                return False
        return True

    async def _pace(self) -> None:
        now = self._clock()
        if self._next_s < now:
            self._next_s = now  # the caller's queue ran dry: a new burst starts now
        wait = self._next_s - self._lead_s - now
        if wait > 0:
            await self._sleep(wait)
        self._next_s += PIECE_S

    async def _send(self, send: Callable[[], Awaitable[None]]) -> bool:
        """False once the caller has hung up: the pipeline is being torn down anyway."""
        if not self._open:
            return False
        try:
            await send()
            return True
        except ConnectionClosed:
            self._open = False
            return False


class LineTransport(BaseTransport):
    """One call's line as a Pipecat transport: `input()` first in the pipeline,
    `output()` after the voice."""

    def __init__(self, line: LineOut, params: TransportParams | None = None, **output_options) -> None:
        super().__init__()
        params = params or line_params()
        self._input = LineInput(params, name="line-in")
        self._output = LineOutput(line, params, name="line-out", **output_options)

    def input(self) -> LineInput:
        return self._input

    def output(self) -> LineOutput:
        return self._output
