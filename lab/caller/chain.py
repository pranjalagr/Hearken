"""The real pretend caller (Part 3, C1): ears -> brain -> voice, behind the director.

It sits exactly where the scripted caller does, and writes the same log, so the
director, the scorer and the judge don't know which one is on the line.

Timing (C9): the brain starts as soon as it has the robot's words, so the reply is
ready when it's needed, but the caller speaks only once the robot is done. It never
talks over the robot, apart from the director's planned tricks. If more robot words
arrive after it started thinking, that thought is thrown away and it thinks again.
A reply that never comes ends the call as a lab problem (`_stuck`), never a quiet call.

`heard` runs inside the 20 ms beat, so everything slow (the brain, the voice, the
ears' connection) runs in its own task and only leaves its result here.
"""

from __future__ import annotations

import asyncio
import difflib
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from common.wire.protocol import PIECE_MS
from engine.tasks.task_file import Task
from lab.caller.brain.brain import SAFE_LINE, Brain, Thought, Turn
from lab.caller.director.director import Dropped, SaidLine
from lab.caller.director.plan import DirectorSettings
from lab.caller.ears.live import LiveEars
from lab.caller.mouth.fixed_lines import FixedLines
from lab.caller.mouth.number_words import speak_numbers_letters
from lab.caller.mouth.phone_mic import Mic, PhoneMic
from lab.caller.mouth.voice import Voice
from lab.caller.scripted import ScriptMic
from lab.caller.settings import CallerSettings
from lab.ears_test.services import Service
from lab.synthetic_voice import is_speech

Event = Callable[[str, dict[str, Any]], None]

GIVE_UP_LINE = "Chhodiye, rehne dijiye. Bye."
COULD_NOT_HEAR = "(You heard sound on the line but couldn't make out any words.)"
# The director's lines change what the caller is doing, so a reply prepared before
# them no longer fits. A "haan haan" while listening changes nothing.
_INTERRUPTS_THOUGHT = ("race", "interrupt")


class CallerStuck(RuntimeError):
    """The caller owes the robot a reply and has none: the lab's failure, not the robot's."""


@dataclass
class ChainParts:
    """What a chain caller is made of, passed in so the lab can use fakes (no paid
    request in a test) and so a part can be swapped by a setting."""

    brain: Brain
    voice: Voice
    ears_service: Service
    make_noise: Callable[[], Any] | None = None  # makes a NoiseBed; None is a quiet line
    describe: dict[str, Any] = field(default_factory=dict)
    closers: list[Callable[[], Any]] = field(default_factory=list)  # connections to close after the call


@dataclass
class _Prepared:
    thought: Thought
    pieces: list[bytes]
    spoken: str


class Line(Protocol):
    async def press_key(self, digit: str) -> None: ...

    async def hang_up(self, reason: str = "bye") -> None: ...


class ChainCaller:
    def __init__(self, parts: ChainParts, task: Task, on_event: Event, settings: CallerSettings,
                 director_settings: DirectorSettings = DirectorSettings()) -> None:
        self._parts = parts
        self._task = task
        self._spec = task.caller
        self._on_event = on_event
        self._s = settings
        self._ds = director_settings
        self.mic = ScriptMic(on_event)
        self.recordings = FixedLines(parts.voice, self._spec.number_style)
        self._ears = LiveEars(parts.ears_service, self._on_words)
        self._noise = None
        self._language = "english" if self._spec.language == "mostly_english" else "hinglish"
        self.director = None
        self._line: Line | None = None
        self.error: BaseException | None = None
        self._tasks: set[asyncio.Task] = set()
        self._ears_task: asyncio.Task | None = None
        self._thinking: asyncio.Task | None = None
        self._thinking_from_ms = 0
        self._sound_after_thought = False  # robot sound since the current thought began
        self._owed_ms = 0  # quiet time spent owing the robot a reply

        self._now_ms = 0
        self._robot_spoke = False  # since the caller's last turn
        self._robot_last_ms: int | None = None
        self._history: list[Turn] = []
        self._pending: list[str] = []
        self._dirty = False
        self._prepared: _Prepared | None = None
        self._opening_said = False
        self._last_robot_turns: list[str] = []
        self._finished = False  # no more thinking: said goodbye, or the line is gone
        self._hang_up_at_ms: int | None = None
        self._hung_up = False
        self._give_up_after_ms = int(settings.give_up_s[self._spec.personality] * 1000)

    # ---- the lab's side ------------------------------------------------------------

    def wrap_mic(self, mic: Mic) -> Mic:
        return PhoneMic(mic, self._noise)

    def attach(self, line: Line, director: Any) -> None:
        self._line = line
        self.director = director

    async def prepare(self) -> None:
        """Before the call is placed: every line the director could ask for, and the
        opening line. The ears start here too, because connecting can take seconds."""
        lines = [self._spec.opening_line, self._ds.backchannel_line, self._ds.interrupt_line,
                 self._ds.silence_end_line, GIVE_UP_LINE]
        if self._task.race is not None:
            lines.append(self._task.race.line)
        await self.recordings.make(lines)
        if self._parts.make_noise is not None:
            # Reading and filtering a whole recording is slow work: off the loop.
            self._noise = await asyncio.to_thread(self._parts.make_noise)
        self._ears_task = asyncio.create_task(self._run_ears())

    def background(self) -> list[asyncio.Task]:
        return [self._ears_task] if self._ears_task is not None else []

    async def close(self) -> None:
        self._finished = True
        self._ears.end()
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        if self._ears_task is not None:
            # Give the service a moment to close cleanly, so its bill stops.
            await asyncio.wait({self._ears_task}, timeout=2.0)
            self._ears_task.cancel()
            await asyncio.gather(self._ears_task, return_exceptions=True)
        for closer in self._parts.closers:
            await closer()
        self._on_event("caller.cost", {"service": "ears", "seconds": round(self._ears.seconds, 1)})

    def about(self) -> dict[str, Any]:
        return dict(self._parts.describe)

    def director_event(self, happened: Any) -> None:
        """The director spoke for the caller, or dropped the call."""
        if isinstance(happened, Dropped):
            self._finished = True
            self._drop_thinking()
            return
        if not isinstance(happened, SaidLine):
            return
        self._history.append(Turn("me", happened.line))
        if happened.then_wants:
            self._parts.brain.wants = happened.then_wants  # C10: what the caller wants now
        if happened.kind in _INTERRUPTS_THOUGHT:
            self._drop_thinking()
            self._owed_ms = 0  # it has just spoken: the next move is the robot's

    # ---- the ear: every 20 ms --------------------------------------------------------

    def heard(self, piece: bytes | None, caller_ms: int) -> None:
        self._now_ms = caller_ms
        self._ears.feed(piece)
        if piece is not None and is_speech(piece, self._s.robot_sound_level):
            self._robot_spoke = True
            self._robot_last_ms = caller_ms
            if self._thinking is not None or self._prepared is not None:
                # The robot wasn't done, or it was only noise. The reply waits: new words
                # start a new thought, and with none it's said once late words can't come
                # any more (_speak_after_ms). Cancelled at once instead, a person's pauses
                # and room noise threw away 7 requests in one call, until Groq's limit
                # stopped the caller (5 Nov 2026).
                self._sound_after_thought = True
        if self._finished and self._hang_up_at_ms is None:
            return
        self._step(caller_ms)

    def _step(self, ms: int) -> None:
        quiet_ms = None if self._robot_last_ms is None else ms - self._robot_last_ms - PIECE_MS
        if self._hang_up_at_ms is not None:
            if not self.mic.speaking and ms >= self._hang_up_at_ms:
                self._hang_up("goal done")
            return
        if ms >= self._give_up_after_ms:
            self._say_give_up()
            return
        if not self._opening_said:
            no_greeting = self._robot_last_ms is None and ms >= self._s.opening_wait_s * 1000
            if no_greeting or (quiet_ms is not None and quiet_ms >= self._gap_ms()):
                self._say_opening()
            return
        if quiet_ms is None:
            return
        if self._stuck(quiet_ms):
            self._fail(CallerStuck(f"no reply after {self._s.reply_stuck_s:g} s of quiet"))
            return
        if self._dirty and quiet_ms >= self._s.think_after_quiet_s * 1000:
            self._start_thinking()
        elif (
            self._robot_spoke and not self._pending and self._thinking is None and self._prepared is None
            and quiet_ms >= (self._s.robot_done_s + self._s.heard_nothing_after_s) * 1000
        ):
            # Sound, but no words came: a real caller says so. Once per robot turn: the
            # note stays pending until it's answered.
            self._on_event("caller.heard_nothing", {"caller_ms": ms})
            self._pending.append(COULD_NOT_HEAR)
            self._dirty = True
        if self._prepared is not None and quiet_ms >= self._speak_after_ms() and not self.mic.speaking:
            self._speak(self._prepared)

    def _gap_ms(self) -> float:
        return (self._s.robot_done_s + self._s.extra_gap_s[self._spec.personality]) * 1000

    def _speak_after_ms(self) -> float:
        """A reply thought of before the robot's last sound waits until that sound's words,
        if any, have had time to reach the ears: said sooner, it once answered half a
        sentence (4 Nov 2026)."""
        if self._sound_after_thought:
            return max(self._gap_ms(), self._s.late_words_s * 1000)
        return self._gap_ms()

    def _stuck(self, quiet_ms: int) -> bool:
        """A caller gone quiet because its brain is slow looks just like a healthy call:
        the robot moves on without an answer, and gets the blame for what follows. On
        4 Nov 2026 a slow brain did this to 14 of 15 calls, all labelled fine. So quiet time
        spent owing a reply is counted, and a robot's "hello?" doesn't restart the count:
        that throws the thought away without answering anything. Words left over after a
        race line owe nothing: the caller has spoken, and waits for the robot's answer."""
        replying = self._dirty or self._thinking is not None or self._prepared is not None
        if replying and quiet_ms >= 0 and not self.mic.speaking:
            self._owed_ms += PIECE_MS
        return self._owed_ms >= self._s.reply_stuck_s * 1000

    # ---- hearing ---------------------------------------------------------------------

    async def _run_ears(self) -> None:
        try:
            await self._ears.run()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._fail(exc)

    def _on_words(self, text: str) -> None:
        self._on_event("caller.heard", {"text": text, "caller_ms": self._now_ms})
        if self.director is not None:
            self.director.notify_robot_said(text)  # for the after_done drop (C12)
        if not self._robot_spoke:
            # Late words from a turn already answered: no robot sound since the caller's
            # last turn. Treated as new, they made it speak twice in a row (5 Nov 2026).
            self._history.append(Turn("robot", text))
            return
        self._pending.append(text)
        self._dirty = True

    # ---- thinking --------------------------------------------------------------------

    def _start_thinking(self) -> None:
        self._dirty = False
        self._drop_thinking()
        if self._finished or self._same_question_too_often():
            if not self._finished:
                self._say_give_up()
            return
        history = [*self._history, Turn("robot", " ".join(self._pending))]
        self._thinking_from_ms = self._now_ms
        self._sound_after_thought = False
        self._thinking = asyncio.create_task(self._think(history))
        self._tasks.add(self._thinking)
        self._thinking.add_done_callback(self._tasks.discard)

    def _drop_thinking(self) -> None:
        if self._thinking is not None or self._prepared is not None:
            # Logged, so lost replies and their wasted requests can be counted: dropped
            # silently, a brain too slow to ever answer left no trace in the call.
            self._on_event("caller.thought_dropped", {"ran_ms": self._now_ms - self._thinking_from_ms, "caller_ms": self._now_ms})
        if self._thinking is not None:
            self._thinking.cancel()
            self._thinking = None
        self._prepared = None

    async def _think(self, history: list[Turn]) -> None:
        loop = asyncio.get_running_loop()
        started = loop.time()
        try:
            thought = await self._parts.brain.think(history)
            thought_at = loop.time()
            spoken = self._spoken(thought)
            pieces = await self._parts.voice.pieces(spoken) if spoken else []
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._fail(exc)
            return
        self._on_event("caller.thought", {"brain_ms": round((thought_at - started) * 1000),
                                          "voice_ms": round((loop.time() - thought_at) * 1000)})
        # Code caught these, so the robot never faces them, unless all that's left to say
        # is the safe line: that's a mistake the robot does hear, and C15's bar counts it.
        event = "caller.mistake" if thought.line == SAFE_LINE.line else "caller.caught"
        for problem in thought.problems:
            self._on_event(event, {"kind": "bad answer from the brain", "detail": problem, "caller_ms": self._now_ms})
        self._thinking = None
        self._prepared = _Prepared(thought, pieces, spoken)

    def _spoken(self, thought: Thought) -> str:
        if not thought.line:
            return ""
        # English lines stay in English letters: the voice reads those as English.
        text = thought.line if thought.language == "english" else thought.hindi_letters
        return speak_numbers_letters(text, self._spec.number_style)

    def _same_question_too_often(self) -> bool:
        """C12: the same question 3 times means the robot is stuck: give up."""
        turn = _plain(" ".join(self._pending))
        same = sum(difflib.SequenceMatcher(None, turn, past).ratio() > 0.85 for past in self._last_robot_turns)
        return bool(turn) and same + 1 >= self._s.give_up_same_question

    # ---- speaking --------------------------------------------------------------------

    def _say_opening(self) -> None:
        self._opening_said = True
        line = self._spec.opening_line
        self._commit_robot_words()
        self._say(line, self.recordings.spoken(line), self.recordings.pieces(line), None, self._language)

    def _speak(self, prepared: _Prepared) -> None:
        self._prepared = None
        thought = prepared.thought
        self._commit_robot_words()
        if thought.press_key and self._line is not None:
            self._on_event("caller.key", {"digit": thought.press_key, "caller_ms": self._now_ms})
            self._side(self._line.press_key(thought.press_key))
        if thought.line:
            self._say(thought.line, prepared.spoken, prepared.pieces, thought.agrees, thought.language)
        if thought.done:
            self._goodbye()

    def _say(self, text: str, spoken: str, pieces: list[bytes], agrees: tuple[str, int] | None, language: str) -> None:
        self.mic.say_line(text, pieces, agrees, language, {"spoken": spoken})
        self._history.append(Turn("me", text))
        if self.director is not None:
            self.director.notify_caller_said(text)

    def _commit_robot_words(self) -> None:
        """This turn answers every word heard. None can be newer than the reply: they'd
        have started a new thought first, because thinking starts sooner after quiet
        (think_after_quiet_s) than a reply is said (robot_done_s)."""
        if self._pending:
            turn = " ".join(self._pending)
            self._history.append(Turn("robot", turn))
            self._last_robot_turns.append(_plain(turn))
            self._pending = []
        self._dirty = False
        self._robot_spoke = False
        self._owed_ms = 0  # answered: new words start a new count

    def _say_give_up(self) -> None:
        self._on_event("caller.gave_up", {"caller_ms": self._now_ms})
        self._drop_thinking()
        self._say(GIVE_UP_LINE, self.recordings.spoken(GIVE_UP_LINE), self.recordings.pieces(GIVE_UP_LINE), None, self._language)
        self._goodbye()

    def _goodbye(self) -> None:
        """One goodbye only (C12): wait for the robot's bye, then hang up."""
        self._finished = True
        self._drop_thinking()
        self._hang_up_at_ms = self._now_ms + int(self._s.goodbye_wait_s * 1000)

    def _hang_up(self, reason: str) -> None:
        if self._hung_up or self._line is None:
            return
        self._hung_up = True
        self._side(self._line.hang_up(reason))

    def _side(self, work: Any) -> None:
        """Called from inside the beat, which can't wait."""
        task = asyncio.create_task(work)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _fail(self, exc: BaseException) -> None:
        """A part of the caller broke: end the call, and the lab reruns it. It says
        nothing about the robot, so it must never count against one (L4)."""
        if self.error is None:
            self.error = exc
        self._finished = True
        self._hang_up("the caller broke")


def _plain(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())
