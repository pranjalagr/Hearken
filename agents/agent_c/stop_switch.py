"""The one way Agent C's voice is stopped: the stop-or-keep model says stop (decision 26).

A pipeline part with one job: when told, it sends Pipecat's interruption both ways, so
the voice stops, the queue at the caller is cleared, and a reply being written is
dropped, exactly as a Pipecat interruption from the caller's own sound would.
"""

from __future__ import annotations

from pipecat.frames.frames import Frame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class StopSwitch(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        # Pass everything on. Pipecat's own process_frame forwards nothing: without this,
        # the switch swallowed the StartFrame and Agent C's first real call (16 Nov 2026)
        # never started ("timeout waiting for StartFrame ... being blocked somewhere?").
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

    async def stop_voice(self) -> None:
        await self.broadcast_interruption()
