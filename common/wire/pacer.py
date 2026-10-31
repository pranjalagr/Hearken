"""A steady 20 ms beat for the caller's side.

Sleeping 20 ms in a loop drifts: every sleep runs a little long, and the
errors add up. Instead, tick n is due at start + n x 20 ms, measured from one
fixed start. If the computer was busy and ticks are overdue, they are handed
out at once, one after another, so the caller never skips a piece: its clock
(pieces sent x 20 ms) must keep up with real time.

On Windows the event loop wakes up with roughly 15 ms granularity, so single
ticks can be a few ms late; thanks to the fixed start they never drift.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable

from common.wire.protocol import PIECE_MS


class Pacer:
    def __init__(
        self,
        period_s: float = PIECE_MS / 1000,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if period_s <= 0:
            raise ValueError("period_s must be positive")
        self._period = period_s
        self._clock = clock
        self._sleep = sleep

    async def ticks(self) -> AsyncIterator[int]:
        """Tick numbers 0, 1, 2, ... each when it is due. Never skips one."""
        clock = self._clock or asyncio.get_running_loop().time
        start = clock()
        n = 0
        while True:
            wait = start + n * self._period - clock()
            if wait > 0:
                await self._sleep(wait)
            yield n
            n += 1
