"""The pretend clock. Every engine time is seconds since the call started."""

from __future__ import annotations

import time
from typing import Protocol


class Clock(Protocol):
    def now(self) -> float: ...


class ManualClock:
    """For tests: time moves only when the test moves it.

    "Two minutes later" then takes no real time, and every run sees exactly
    the same times, so the same task gives the same result.
    """

    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def now(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("time only moves forward")
        self._now += seconds


class WallClock:
    """For live calls: real seconds since the call started.

    Built from the start time stored in call_info, so every process working on
    the same call (the backend, the Guard) agrees on what "now" is.
    """

    def __init__(self, started_at_wall: float) -> None:
        self._started = started_at_wall

    def now(self) -> float:
        return time.time() - self._started
