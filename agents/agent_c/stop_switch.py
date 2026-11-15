"""The one way Agent C's voice is stopped: the stop-or-keep model says stop (decision 26).

A pipeline part with one job: when told, it sends Pipecat's interruption both ways, so
the voice stops, the queue at the caller is cleared, and a reply being written is
dropped, exactly as a Pipecat interruption from the caller's own sound would.
"""

from __future__ import annotations

from pipecat.processors.frame_processor import FrameProcessor


class StopSwitch(FrameProcessor):
    async def stop_voice(self) -> None:
        await self.broadcast_interruption()
