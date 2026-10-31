"""The dummy robot: fixed lines in a fixed order, the same every time.

C14: a brand-new caller tested against a brand-new robot can't tell you who
broke a call. This puppet has no AI and never varies, so when a call with it
goes wrong, the caller is to blame. Its script is built to test the caller's
skills: long sentences (does it say "haan haan"?), a pause inside a sentence
(does it wrongly jump in?), "1 dabaiye" (does it press the key?), and a
goodbye (does it hang up cleanly?).

It calls the real backend at fixed points, so the backend's bells ring for
the director exactly as in a real race test, and it runs inside a call box
like every robot. It also hears the caller's speech, so it can report
whether the caller ever talked over it.

`EchoRobot` is a second, simpler puppet: it plays your voice back 2 seconds
later, to check the web phone by ear.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agents.common.call_box import CallBox, CallBoxSettings, CallResult
from common.wire.protocol import PIECE_MS, Cleared, End, Key, Played
from common.wire.robot_line import CallerAudio, LineClosed, RobotLine
from engine.safety.guard_door import GuardDoor
from engine.service.backend import Backend
from lab.synthetic_voice import is_speech, synthetic_line

# C9: the caller counts the robot as "done" after 0.8 s of quiet; the puppet
# uses the same rule to decide the caller has finished a turn.
TURN_DONE_MS = 800


@dataclass(frozen=True)
class Say:
    text: str


@dataclass(frozen=True)
class SayWithPause:
    """One sentence with a pause inside it, shorter than the caller's 0.8 s
    "done" rule: a good caller waits instead of jumping in."""

    first: str
    pause_s: float
    second: str


@dataclass(frozen=True)
class Listen:
    """Wait for the caller's turn: some speech, then 0.8 s of quiet."""

    timeout_s: float = 15.0


@dataclass(frozen=True)
class AskKey:
    """Say a line ("haan ke liye 1 dabaiye"), then wait for a keypad key."""

    text: str
    timeout_s: float = 10.0


@dataclass(frozen=True)
class UseTool:
    """Call the backend. With approve=True the puppet plays the Guard: it
    writes a yes-ticket for the last prepared token, then confirms it."""

    tool: str
    inputs: dict[str, Any] = field(default_factory=dict)
    approve: bool = False


@dataclass(frozen=True)
class HangUp:
    reason: str


Step = Say | SayWithPause | Listen | AskKey | UseTool | HangUp


@dataclass(frozen=True)
class RobotTurn:
    text: str
    start_ms: int  # on the caller's clock: when the caller started hearing it
    end_ms: int  # when the caller finished hearing it (its "played" receipt)


@dataclass
class DummyLog:
    turns: list[RobotTurn] = field(default_factory=list)
    caller_speech_ms: list[int] = field(default_factory=list)  # every caller piece with speech
    keys: list[Key] = field(default_factory=list)
    tool_replies: list[tuple[str, str]] = field(default_factory=list)  # (tool, status)
    cleared: list[Cleared] = field(default_factory=list)
    timeouts: list[str] = field(default_factory=list)
    pauses: list[tuple[int, int]] = field(default_factory=list)  # pauses inside a sentence, caller clock

    def talked_over(self) -> list[tuple[str, int]]:
        """Caller speech heard while one of the puppet's turns was playing:
        (the turn's text, caller_ms). C14 wants none outside planned tricks."""
        return [
            (turn.text, ms)
            for turn in self.turns
            for ms in self.caller_speech_ms
            if turn.start_ms <= ms < turn.end_ms
        ]

    def jumped_in(self) -> list[int]:
        """Caller speech during a pause inside a sentence (caller_ms): the
        caller took a breath for the end of the robot's turn."""
        return [ms for start, end in self.pauses for ms in self.caller_speech_ms if start <= ms < end]


class DummyRobot:
    def __init__(
        self,
        script: list[Step],
        backend: Backend | None = None,
        guard: GuardDoor | None = None,
        voice: Callable[[str], list[bytes]] = synthetic_line,
        box_settings: CallBoxSettings = CallBoxSettings(),
    ) -> None:
        self._script = script
        self._backend = backend
        self._guard = guard
        self._voice = voice
        self._box_settings = box_settings
        self.log = DummyLog()
        self.result: CallResult | None = None
        self._played: dict[str, int] = {}
        self._played_event = asyncio.Event()
        self._keys: asyncio.Queue[Key] = asyncio.Queue()
        self._turn_done = asyncio.Event()
        self._listening_since_ms: int | None = None
        self._heard_speech = False
        self._quiet_ms = 0
        self._marks = 0
        self._last_token: str | None = None
        # Set before the puppet closes the line itself, so that the closing it
        # caused is not mistaken for the line dropping.
        self._hanging_up: str | None = None

    async def __call__(self, line: RobotLine) -> None:
        box = CallBox(self._box_settings)
        self.result = await box.run(lambda b: self._main(line, b))

    async def _main(self, line: RobotLine, box: CallBox) -> None:
        box.start("ears", self._read(line, box))
        for step in self._script:
            await self._do(step, line, box)
        await self._hang_up("script done", line, box)

    async def _hang_up(self, reason: str, line: RobotLine, box: CallBox) -> None:
        self._hanging_up = f"robot:{reason}"
        await line.end(reason)
        box.end(self._hanging_up)

    # ---- the caller's side, as the puppet hears it ---------------------------------

    async def _read(self, line: RobotLine, box: CallBox) -> None:
        async for event in line.events():
            if isinstance(event, CallerAudio):
                box.heard_audio()
                self._hear(event)
            elif isinstance(event, Played):
                self._played[event.name] = event.caller_ms
                self._played_event.set()
            elif isinstance(event, Key):
                self.log.keys.append(event)
                self._keys.put_nowait(event)
            elif isinstance(event, Cleared):
                self.log.cleared.append(event)
            elif isinstance(event, End):
                box.end(f"caller:{event.reason}")
            elif isinstance(event, LineClosed):
                box.end(self._hanging_up or "line closed")

    def _hear(self, audio: CallerAudio) -> None:
        speech = is_speech(audio.piece)
        if speech:
            self.log.caller_speech_ms.append(audio.caller_ms)
        if self._listening_since_ms is None or audio.caller_ms < self._listening_since_ms:
            return
        if speech:
            self._heard_speech = True
            self._quiet_ms = 0
        elif self._heard_speech:
            self._quiet_ms += PIECE_MS
            if self._quiet_ms >= TURN_DONE_MS:
                self._turn_done.set()

    # ---- the script ---------------------------------------------------------------

    async def _do(self, step: Step, line: RobotLine, box: CallBox) -> None:
        if isinstance(step, Say):
            await self._say(step.text, line)
        elif isinstance(step, SayWithPause):
            first_end = await self._wait_played(await self._send(step.first, line), step.first)
            await asyncio.sleep(step.pause_s)
            second_end = await self._wait_played(await self._send(step.second, line), step.second)
            if first_end is not None and second_end is not None:
                self.log.pauses.append((first_end, self.log.turns[-1].start_ms))
        elif isinstance(step, Listen):
            await self._listen(line, step.timeout_s)
        elif isinstance(step, AskKey):
            await self._say(step.text, line)
            try:
                await asyncio.wait_for(self._keys.get(), step.timeout_s)
            except TimeoutError:
                self.log.timeouts.append(f"no key after: {step.text}")
        elif isinstance(step, UseTool):
            await self._use_tool(step)
        elif isinstance(step, HangUp):
            await self._hang_up(step.reason, line, box)
            await asyncio.Event().wait()  # the box is closing; wait to be cancelled

    async def _say(self, text: str, line: RobotLine) -> None:
        sent = await self._send(text, line)
        await self._wait_played(sent, text)

    async def _send(self, text: str, line: RobotLine) -> tuple[str, int]:
        """Send a line as fast as it comes (decision 22), then a mark after it."""
        pieces = self._voice(text)
        for piece in pieces:
            await line.send_audio(piece)
        self._marks += 1
        mark = f"say-{self._marks}"
        await line.send_mark(mark)
        return mark, len(pieces)

    async def _wait_played(self, sent: tuple[str, int], text: str) -> int | None:
        mark, count = sent
        limit = count * PIECE_MS / 1000 + 5.0  # its own length, plus a generous margin
        try:
            async with asyncio.timeout(limit):
                while mark not in self._played:
                    self._played_event.clear()
                    await self._played_event.wait()
        except TimeoutError:
            self.log.timeouts.append(f"never played: {text}")
            return None
        end_ms = self._played[mark]
        # Played back to back, so the caller started hearing it `count` pieces earlier.
        self.log.turns.append(RobotTurn(text, end_ms - count * PIECE_MS, end_ms))
        return end_ms

    async def _listen(self, line: RobotLine, timeout_s: float) -> None:
        self._listening_since_ms = line.pieces_received * PIECE_MS
        self._heard_speech = False
        self._quiet_ms = 0
        self._turn_done.clear()
        try:
            await asyncio.wait_for(self._turn_done.wait(), timeout_s)
        except TimeoutError:
            self.log.timeouts.append("the caller never took a turn")
        finally:
            self._listening_since_ms = None

    async def _use_tool(self, step: UseTool) -> None:
        if self._backend is None:
            raise RuntimeError("this script calls the backend, but the puppet has none")
        inputs = dict(step.inputs)
        if step.approve:
            # The puppet plays the Guard: a ticket for exactly the last prepared token.
            inputs = {"token": self._last_token, "ticket": await self._guard.issue_ticket(self._last_token)}
        reply = await self._backend.call_tool(step.tool, inputs)
        if reply.data and "token" in reply.data and step.tool.startswith("prepare_"):
            self._last_token = reply.data["token"]
        self.log.tool_replies.append((step.tool, reply.status))


class EchoRobot:
    """Plays the caller's own voice back after `delay_s`: hearing yourself is
    the quickest way to check the web phone's microphone, speaker and timing."""

    def __init__(self, delay_s: float = 2.0, box_settings: CallBoxSettings = CallBoxSettings()) -> None:
        self._delay_pieces = round(delay_s * 1000 / PIECE_MS)
        self._box_settings = box_settings
        self.result: CallResult | None = None

    async def __call__(self, line: RobotLine) -> None:
        box = CallBox(self._box_settings)
        self.result = await box.run(lambda b: self._main(line, b))

    async def _main(self, line: RobotLine, box: CallBox) -> None:
        for piece in synthetic_line("echo test ready"):
            await line.send_audio(piece)
        held: deque[bytes] = deque()
        async for event in line.events():
            if isinstance(event, CallerAudio):
                box.heard_audio()
                held.append(event.piece)
                if len(held) > self._delay_pieces:
                    await line.send_audio(held.popleft())
            elif isinstance(event, End):
                box.end(f"caller:{event.reason}")
                return
            elif isinstance(event, LineClosed):
                box.end("line closed")
                return
