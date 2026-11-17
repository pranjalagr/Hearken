"""Watches one call's quiet and fills it (decisions 11 and 13), for the two graphs.

Every tick (settings: speed.tick_s) it looks at the call: is the robot or the caller
talking, is the Thinker graph working, is a change saving, is the Guard asking? Then:
- the Thinker works and the robot is quiet: a waiting line; the Thinker's plan takes a
  waiting line's place (decided 17 Nov 2026), said as soon as the robot is quiet, and the
  next waiting line keeps the same pace after it. Held for the next waiting slot instead
  (~4 s), the plan was almost never heard: the tool after it takes 0.3-0.8 s and then
  the yes-question stops the waiting lines (found in step 6's test, 17 Nov 2026);
- the robot spoke and the caller says nothing: "Hello, aap sun rahe hain?", then
  "Lagta hai line kat gayi" and the call ends.

The rules themselves are in quiet.py (WaitingLines, SilentCaller); this only reads the
call and plays what they choose. Every look is a plain read, so nothing here holds up
the event loop.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from agents.agent_c.talker.quiet import Moment, QuietSettings, SilentCaller, WaitingLines


@dataclass
class CallView:
    """What the watch reads from the call, as small functions (the robot's CallParts
    gives them; tests give plain ones)."""

    robot_speaking: Callable[[], bool]
    caller_speaking: Callable[[], bool]
    working: Callable[[], bool]
    saving: Callable[[], bool]
    guard_asking: Callable[[], bool]


class QuietWatch:
    """One call's waiting lines and silent-caller lines."""

    def __init__(self, view: CallView, settings: QuietSettings, waiting_names: list[str],
                 say_line: Callable[[str, str], Awaitable[Any]], say_text: Callable[[str], Awaitable[Any]],
                 end_call: Callable[[str], Awaitable[Any]], tick_s: float,
                 clock: Callable[[], float] | None = None) -> None:
        self._view = view
        self._say_line = say_line  # (group, value): a fixed line from words.yaml
        self._say_text = say_text  # a plan, already made safe by the Talker graph
        self._tick_s = tick_s
        self._clock = clock or (lambda: asyncio.get_running_loop().time())
        self._plan: str | None = None
        self.waiting = WaitingLines(waiting_names, settings, self._play_waiting)
        self.silent = SilentCaller(settings, lambda name: self._say_line("silent", name),
                                   lambda: end_call("the caller went silent"))

    def hold_plan(self, text: str) -> None:
        """The Thinker's plan: said in a waiting line's place once the robot is quiet (only
        the newest is kept). "" drops a plan not yet said."""
        self._plan = text or None

    def drop_plan(self) -> None:
        """The job is over: a plan not said by now is old."""
        self._plan = None

    def moment(self) -> Moment:
        """The call as it is right now."""
        v = self._view
        return Moment(robot_speaking=v.robot_speaking(), caller_speaking=v.caller_speaking(), working=v.working(),
                      saving=v.saving(), guard_asking=v.guard_asking())

    async def tick(self) -> None:
        """One look at the call: maybe the plan, a waiting line or a silent-caller line."""
        now, moment = self._clock(), self.moment()
        if self._plan and not (moment.robot_speaking or moment.caller_speaking or moment.guard_asking):
            plan, self._plan = self._plan, None
            self.waiting.played_instead(now)
            await self._say_text(plan)
            return
        await self.waiting.tick(now, moment)
        await self.silent.tick(now, moment)

    async def run(self) -> None:
        """Look every tick until the call ends (the call box cancels it)."""
        while not self.silent.ended:
            await asyncio.sleep(self._tick_s)
            await self.tick()

    async def _play_waiting(self, name: str) -> None:
        """A waiting line is due."""
        await self._say_line("waiting", name)
