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

In the test lab (Part 5) it is also the puppet that plays each task's perfect
and careless robot, as its own program over MCP (cli/puppet_robot.py), so
the whole lab can be checked with known right answers and no AI at all. It
writes the same diary as every robot (L5).

`EchoRobot` is a second, simpler puppet: it plays your voice back 2 seconds
later, to check the web phone by ear.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agents.common.call_box import CallBox, CallBoxSettings, CallResult
from agents.common.diary import NO_DIARY, Diary
from agents.common.tools import Tools
from common.wire.protocol import PIECE_MS, Cleared, End, Key, Played
from common.wire.robot_line import CallerAudio, LineClosed, RobotLine
from engine.safety.guard_door import GuardDoor
from lab.synthetic_voice import is_speech, synthetic_line

# C9: the caller counts the robot as "done" after 0.8 s of quiet; the puppet
# uses the same rule to decide the caller has finished a turn.
TURN_DONE_MS = 800
# How much caller speech, heard while a stoppable line plays, makes the
# puppet stop talking: long enough to ignore a click, short enough to stop
# well inside L9's 1 s.
CUT_IN_MS = 200


@dataclass(frozen=True)
class Say:
    text: str
    # True: stop talking (clear the caller's speaker) if the caller talks over
    # this line, like a robot that yields to "ruko". The scorer's "stopped"
    # case (L9) needs a robot that really stops.
    yields: bool = False


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
    writes a yes-ticket for the last prepared token, then confirms it.
    With shielded=True the call is a change already sent (decision 14): it
    finishes even if the call ends while it runs."""

    tool: str
    inputs: dict[str, Any] = field(default_factory=dict)
    approve: bool = False
    shielded: bool = False


@dataclass(frozen=True)
class RetryLast:
    """Send the last tool call again with exactly the same inputs (the same
    token and ticket), as a careful robot does when a reply is lost."""


@dataclass(frozen=True)
class HangUp:
    reason: str


Step = Say | SayWithPause | Listen | AskKey | UseTool | RetryLast | HangUp


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
    stopped: list[str] = field(default_factory=list)  # lines it stopped because the caller cut in

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
        tools: Tools | None = None,
        guard: GuardDoor | None = None,
        voice: Callable[[str], list[bytes]] = synthetic_line,
        box_settings: CallBoxSettings = CallBoxSettings(),
        diary: Diary = NO_DIARY,
    ) -> None:
        self._script = script
        self._tools = tools
        self._guard = guard
        self._voice = voice
        self._box_settings = box_settings
        self._diary = diary
        self.log = DummyLog()
        self._last_call: tuple[str, dict[str, Any]] | None = None
        # Set while a yielding line plays: caller speech then stops the line.
        self._yielding = False
        self._cut_in_ms = 0
        self._cut_in = asyncio.Event()
        self._turn_heard_from: int | None = None
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
        if self._yielding:
            self._cut_in_ms = self._cut_in_ms + PIECE_MS if speech else 0
            if self._cut_in_ms >= CUT_IN_MS:
                self._cut_in.set()
        if self._listening_since_ms is None or audio.caller_ms < self._listening_since_ms:
            return
        if speech:
            if not self._heard_speech:
                self._turn_heard_from = audio.caller_ms
            self._heard_speech = True
            self._quiet_ms = 0
        elif self._heard_speech:
            self._quiet_ms += PIECE_MS
            if self._quiet_ms >= TURN_DONE_MS and not self._turn_done.is_set():
                # The puppet has no ears to read words: it notes only when the caller spoke.
                self._diary.heard(None, from_ms=self._turn_heard_from, to_ms=audio.caller_ms - self._quiet_ms + PIECE_MS)
                self._turn_done.set()

    # ---- the script ---------------------------------------------------------------

    async def _do(self, step: Step, line: RobotLine, box: CallBox) -> None:
        if isinstance(step, Say):
            if step.yields:
                await self._say_yielding(step.text, line)
            else:
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
            await self._use_tool(step, box)
        elif isinstance(step, RetryLast):
            if self._last_call is None:
                raise RuntimeError("RetryLast with no tool call before it")
            tool, inputs = self._last_call
            await self._call(tool, inputs, box, shielded=True, retry=True)
        elif isinstance(step, HangUp):
            await self._hang_up(step.reason, line, box)
            await asyncio.Event().wait()  # the box is closing; wait to be cancelled

    async def _say(self, text: str, line: RobotLine) -> None:
        sent = await self._send(text, line)
        await self._wait_played(sent, text)

    async def _say_yielding(self, text: str, line: RobotLine) -> None:
        """Say a line, but stop it at once if the caller talks over it."""
        self._cut_in.clear()
        self._cut_in_ms = 0
        self._yielding = True
        try:
            sent = await self._send(text, line)
            played = asyncio.ensure_future(self._wait_played(sent, text))
            cut_in = asyncio.ensure_future(self._cut_in.wait())
            done, _ = await asyncio.wait({played, cut_in}, return_when=asyncio.FIRST_COMPLETED)
            if played in done:
                cut_in.cancel()
                return
            played.cancel()
            await line.clear()
            self.log.stopped.append(text)
            self._diary.decision("stopped talking: the caller cut in", line=text)
        finally:
            self._yielding = False

    async def _send(self, text: str, line: RobotLine) -> tuple[str, int]:
        """Send a line as fast as it comes (decision 22), then a mark after it."""
        self._diary.said(text)
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

    async def _use_tool(self, step: UseTool, box: CallBox) -> None:
        if self._tools is None:
            raise RuntimeError("this script calls the backend, but the puppet has none")
        inputs = dict(step.inputs)
        if step.approve:
            # The puppet plays the Guard: a ticket for exactly the last prepared token.
            inputs = {"token": self._last_token, "ticket": await self._guard.issue_ticket(self._last_token)}
            self._diary.decision("approved: wrote a yes-ticket", token=self._last_token)
        await self._call(step.tool, inputs, box, shielded=step.shielded)

    async def _call(self, tool: str, inputs: dict[str, Any], box: CallBox, shielded: bool, retry: bool = False) -> None:
        self._last_call = (tool, inputs)
        started = time.perf_counter()
        work = self._tools.call(tool, inputs)
        # A change already sent finishes even if the call ends now (decision 14).
        reply = await (box.change(tool, work) if shielded else work)
        self._diary.tool(tool, inputs, getattr(reply, "status", "unknown"), (time.perf_counter() - started) * 1000,
                         retry=retry)
        if reply is None or not hasattr(reply, "status"):
            # The box gave up waiting for a shielded change: its result is unknown.
            self._diary.error(f"{tool}: result unknown")
            self.log.tool_replies.append((tool, "unknown"))
            return
        if reply.data and "token" in reply.data and tool.startswith("prepare_"):
            self._last_token = reply.data["token"]
        self.log.tool_replies.append((tool, reply.status))


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
