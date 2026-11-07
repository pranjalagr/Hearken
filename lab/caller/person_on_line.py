"""A person plays the robot, so they can judge the pretend caller by ear (Part 3, C14).

The web phone and the pretend caller both only know how to dial a robot. This sits
where a robot would and joins whoever dials it: each side's voice goes to the other
side's speaker. The person joins first, so the caller doesn't talk to an empty line.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from common.wire.protocol import PIECE_MS, End
from common.wire.robot_line import CallerAudio, LineClosed, RobotLine
from lab.synthetic_voice import is_speech

# Silence isn't passed on: each side's speaker only plays what it's sent, and the
# caller's ears hear the gaps by themselves. A little is kept after each word, so
# the ends of words aren't clipped.
KEEP_AFTER_SPEECH_MS = 300


class PersonBridge:
    def __init__(self, on_person_joined: Callable[[], None] = lambda: None) -> None:
        self._on_person_joined = on_person_joined
        self._lines: list[RobotLine] = []
        self.ended = asyncio.Event()
        self.end_reason = ""

    async def answer(self, line: RobotLine) -> None:
        if len(self._lines) >= 2:
            await line.end("this line is busy")
            return
        self._lines.append(line)
        side = len(self._lines) - 1
        if side == 0:
            self._on_person_joined()
        kept_ms = 0
        async for event in line.events():
            if isinstance(event, CallerAudio):
                kept_ms = KEEP_AFTER_SPEECH_MS if is_speech(event.piece) else max(0, kept_ms - PIECE_MS)
                other = self._other(side)
                if kept_ms and other is not None:
                    await other.send_audio(event.piece)
            elif isinstance(event, End):
                await self._finish(side, event.reason)
                return
            elif isinstance(event, LineClosed):
                await self._finish(side, "line closed")
                return

    def _other(self, side: int) -> RobotLine | None:
        return self._lines[1 - side] if len(self._lines) == 2 else None

    async def _finish(self, side: int, reason: str) -> None:
        self.end_reason = self.end_reason or f"{'person' if side == 0 else 'caller'}: {reason}"
        other = self._other(side)
        if other is not None:
            await other.end(reason)
        self.ended.set()
