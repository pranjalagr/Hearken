"""The two kinds of quiet on a call, and what code says into each (decisions 11 and 13).

- The caller waits on the back office (a job is working): waiting lines, the first after
  ~1 s of robot silence, then one every ~4 s, never the same twice in a row.
- The robot spoke and the caller says nothing: ~6 s -> "Hello, aap sun rahe hain?";
  ~15 s more -> "Lagta hai line kat gayi", then hang up.

The two never overlap: while the back office works the quiet is the robot's, not the
caller's. Both look at the call every tick (settings: speed.tick_s) through a Moment,
so tests drive them with a made-up clock and no sleeping.
"""

from __future__ import annotations

import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Moment:
    """What the call looks like at one tick."""

    robot_speaking: bool  # the voice is playing, or a written reply is on its way to it
    caller_speaking: bool
    working: bool  # the Thinker graph is running a step, and the caller waits on it
    saving: bool  # a change is in flight
    guard_asking: bool  # the yes-question is open: the Guard has its own asking ladder


@dataclass(frozen=True)
class QuietSettings:
    first_s: float
    every_s: float
    hello_s: float
    line_gone_s: float

    @classmethod
    def from_settings(cls, waiting: dict[str, Any], silent: dict[str, Any]) -> QuietSettings:
        return cls(float(waiting["first_s"]), float(waiting["every_s"]), float(silent["hello_s"]),
                   float(silent["line_gone_s"]))


class WaitingLines:
    def __init__(self, names: Sequence[str], settings: QuietSettings, play: Callable[[str], Awaitable[Any]],
                 chooser: random.Random | None = None) -> None:
        if len(names) < 2:
            # With one line, "never the same twice in a row" could only be kept by silence.
            raise ValueError("waiting lines need at least two lines")
        self._names = list(names)
        self._settings = settings
        self._play = play
        self._chooser = chooser or random.Random()
        self._quiet_since: float | None = None
        self._last_at: float | None = None  # when the last waiting line was started
        self._last: str | None = None
        self.played = 0

    async def tick(self, now: float, moment: Moment) -> None:
        if not moment.working or moment.caller_speaking or moment.guard_asking or moment.robot_speaking:
            # Only robot silence while the back office works counts, and never over the
            # caller or the yes-question.
            self._quiet_since = None
            return
        if self._quiet_since is None:
            self._quiet_since = now
            return
        if now - self._quiet_since < self._settings.first_s:
            return
        if self._last_at is not None and now - self._last_at < self._settings.every_s:
            return
        name = self._chooser.choice([name for name in self._names if name != self._last])
        self._last, self._last_at, self._quiet_since = name, now, None
        self.played += 1
        await self._play(name)

    def played_instead(self, now: float) -> None:
        """Another line took a waiting line's place (the Thinker's plan): the next waiting
        line keeps the same pace after it."""
        self._last_at, self._quiet_since = now, None
        self.played += 1


class SilentCaller:
    def __init__(self, settings: QuietSettings, play: Callable[[str], Awaitable[Any]],
                 end: Callable[[], Awaitable[Any]]) -> None:
        self._settings = settings
        self._play = play
        self._end = end
        self._quiet_since: float | None = None
        self._robot_spoke_last = False  # the robot spoke after the caller's last words: the caller's turn now
        self._asked = False  # "Hello, aap sun rahe hain?" was said in this quiet
        self.ended = False

    async def tick(self, now: float, moment: Moment) -> None:
        if self.ended:
            return
        if moment.caller_speaking:
            self._robot_spoke_last, self._asked = False, False
        elif moment.robot_speaking:
            self._robot_spoke_last = True
        if (moment.robot_speaking or moment.caller_speaking or moment.working or moment.saving
                or moment.guard_asking or not self._robot_spoke_last):
            # Not the caller's silence: someone is speaking, the caller waits on us, a
            # change is saving, or the Guard's question is open (its ladder asks).
            self._quiet_since = None
            return
        if self._quiet_since is None:
            self._quiet_since = now
            return
        quiet = now - self._quiet_since
        if not self._asked and quiet >= self._settings.hello_s:
            self._asked, self._quiet_since = True, None
            await self._play("hello")
        elif self._asked and quiet >= self._settings.line_gone_s:
            self.ended = True
            await self._play("line_gone")
            await self._end()
