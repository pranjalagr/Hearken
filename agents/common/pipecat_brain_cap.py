"""The most brain requests a Pipecat robot may send in one call.

On 15 Nov 2026 one Agent A call (food_09) went round in circles: the ears turned
background noise into turns and the pretend caller kept asking the same thing, so
the brain answered 56 times, a ninth of a free Gemini key's day. A cap ends such
a call instead of letting it run to the 15-minute limit. The lab gives every AI
robot the same number (lab/settings.yaml), so the cap is fair to all of them.

It watches the frames as an observer, like the diary, so the pipeline stays
exactly as Pipecat ships it. A brain request is counted when its token usage
arrives: that is the one record every finished request leaves.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from pipecat.frames.frames import MetricsFrame
from pipecat.metrics.metrics import LLMUsageMetricsData
from pipecat.observers.base_observer import BaseObserver, FramePushed

BRAIN_CAP = "brain_cap"  # the call's end reason when the cap is reached


class BrainCap(BaseObserver):
    def __init__(self, limit: int, on_reached: Callable[[int], Awaitable[None]]) -> None:
        super().__init__()
        self._limit = limit
        self._on_reached = on_reached
        self._seen: set[int] = set()  # a frame passes many processors; count it once
        self.requests = 0
        self._reached = False

    async def on_push_frame(self, data: FramePushed) -> None:
        frame = data.frame
        if not isinstance(frame, MetricsFrame) or frame.id in self._seen:
            return
        self._seen.add(frame.id)
        await self.add(sum(1 for item in frame.data if isinstance(item, LLMUsageMetricsData)))

    async def add(self, requests: int) -> None:
        """Count requests sent outside the pipeline (Agent C's Thinker and Guard judge):
        the cap is for every brain request of the call, wherever it is sent from."""
        self.requests += requests
        if self.requests >= self._limit and not self._reached:
            self._reached = True  # end the call once, even if more usage arrives while it ends
            await self._on_reached(self.requests)
