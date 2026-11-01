"""The director: the part of the pretend caller that times exact moments.

The brain decides WHAT the caller says; the director decides the few things
that must happen at an exact moment, with pre-made lines in the caller's own
voice (Part 3):
- race lines at backend bells (C10), with a hit-or-missed verdict;
- "haan haan" during long robot turns, one planned interruption, one planned
  silence (C9, tricks 2, 3 and 7);
- hanging up or going silent at a chosen moment (C12, drop tests).

It runs inside the caller line's 20 ms beat: its microphone sits in front of
the brain's, and its ear sees every tick of the robot's audio. So it acts on
the caller's own clock, at most one piece (20 ms) after a moment arrives.

Timing rule: during tick t (caller time 20 t) this tick's piece has already
been sent, so whatever the director decides takes effect from the next piece,
at 20 (t + 1). Every time it reports is that moment, when it really happened.
"""

from __future__ import annotations

import asyncio
import random
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Protocol

from common.wire.caller_line import CallerLine, Ear, NullEar
from common.wire.playback import Playback
from common.wire.protocol import PIECE_MS, SILENT_PIECE
from engine.lab_controls.bells import ARRIVED, PREPARED, SAVED, SAVING, Bell
from lab.caller.director.plan import DirectorPlan, DirectorSettings


class BrainMic(Protocol):
    """The brain's mouth, as the director needs to see it."""

    @property
    def speaking(self) -> bool: ...

    @property
    def queued_pieces(self) -> int: ...

    def next_piece(self, tick: int) -> bytes: ...

    def stop(self) -> None: ...


class Recordings(Protocol):
    def pieces(self, line: str) -> list[bytes]:
        """A line pre-made in the caller's voice, as wire pieces."""


class Controls(Protocol):
    def hang_up(self, kind: str) -> None:
        """"clean": the red button. "silent": the line just dies."""


@dataclass(frozen=True)
class SaidLine:
    """The director spoke for the caller; the brain must know what it "said"."""

    kind: str  # race, backchannel, interrupt, silence_end
    line: str
    caller_ms: int
    then_wants: str | None = None  # race lines only: what the caller wants now


@dataclass(frozen=True)
class Dropped:
    moment: str
    kind: str
    caller_ms: int


@dataclass
class RaceReport:
    moment: str
    action: str
    armed_ms: int | None = None  # when the moment's bell was heard
    line_ms: int | None = None  # when the line's first piece went out
    hit: bool | None = None  # None until known
    why: str = ""


@dataclass
class DirectorReport:
    race: RaceReport | None
    said: list[SaidLine] = field(default_factory=list)
    dropped: Dropped | None = None


class Director:
    def __init__(
        self,
        plan: DirectorPlan,
        recordings: Recordings,
        brain_mic: BrainMic,
        playback: Playback,
        controls: Controls,
        on_event=None,
        settings: DirectorSettings = DirectorSettings(),
    ) -> None:
        self._plan = plan
        self._recordings = recordings
        self._brain = brain_mic
        self._playback = playback
        self._controls = controls
        self._on_event = on_event or (lambda event: None)
        self._s = settings
        # One random stream per call, from the task's seed, so A, B and C
        # get the same "random" moments (C13).
        self._rng = random.Random(plan.seed)
        race = plan.race
        self._report = DirectorReport(race=RaceReport(race.moment, race.action) if race else None)
        self.mic = _DirectorMic(self)

        self._inbox: deque[Bell] = deque()
        self._now_ms = 0
        # What the director is saying right now.
        self._queue: deque[bytes] = deque()
        self._queue_kind = ""
        self._queue_line = ""
        self._queue_then_wants: str | None = None
        self._queue_first_sent = False
        # The caller's own speech (director lines or the brain's).
        self._caller_speech_ms = 0  # length of the current utterance so far
        self._last_caller_speech_ms: int | None = None
        # The robot's turns, as heard on the caller's speaker.
        self._last_robot_ms: int | None = None
        self._turns = 0
        self._turn_start_ms = 0
        self._turn_played_ms = 0
        self._backchannel_at_ms: int | None = None
        self._backchannel_done = False
        self._interrupt_done = False
        self._silence = "waiting"  # waiting -> holding -> done
        self._hold_until_ms = 0
        # Race state.
        self._race_fire_at_ms: int | None = None
        self._race_question_deadline_ms: int | None = None
        self._race_saved_deadline_ms: int | None = None
        # Drop state.
        self._dropped = False
        self._drop_after_speech = False
        self._drop_after_race = False  # a while_action_runs drop waiting for the race line
        self._drop_at_ms: int | None = None
        if plan.drop is not None and plan.drop.moment == "random":
            low, high = settings.drop_random_window_s
            self._drop_at_ms = _to_tick(self._rng.uniform(low, high))

    # ---- wiring --------------------------------------------------------------------

    @property
    def report(self) -> DirectorReport:
        race = self._report.race
        if race is not None and race.hit is None and race.moment in ("right_after_sending", "while_running"):
            # No "saved" bell came at all: the action was stopped. If the line
            # was said, the moment was hit; if not, it was never fired.
            if race.line_ms is not None:
                race.hit, race.why = True, "said before any save"
            elif race.armed_ms is not None or self._race_fire_at_ms is not None:
                race.hit, race.why = False, "the line never went out"
        return self._report

    def ear(self, inner: Ear) -> Ear:
        """Wrap the caller's real ear: every tick goes to both."""
        return _DirectorEar(self, inner)

    def ring(self, bell: Bell) -> None:
        """A bell from the backend; acted on at the next tick."""
        self._inbox.append(bell)

    def notify_caller_said(self, text: str) -> None:
        """The brain says what its current line says (for the after_haan drop)."""
        drop = self._plan.drop
        if drop is not None and drop.moment == "after_haan" and _has_words(text, self._s.yes_words):
            self._drop_after_speech = True

    def notify_robot_said(self, text: str) -> None:
        """The caller's ears say what the robot just said (for the after_done drop)."""
        drop = self._plan.drop
        if drop is not None and drop.moment == "after_done" and _has_words(text, self._s.done_words):
            self._drop("after_done")

    # ---- every tick ----------------------------------------------------------------

    def _tick(self, robot_piece: bytes | None, start_ms: int) -> None:
        if self._dropped:
            return
        self._now_ms = start_ms
        self._follow_robot(robot_piece is not None, start_ms)
        while self._inbox:
            self._on_bell(self._inbox.popleft(), start_ms)
            if self._dropped:
                return
        self._race_tick(start_ms)
        self._tricks_tick(start_ms)
        self._drop_tick(start_ms)

    def _follow_robot(self, playing: bool, ms: int) -> None:
        if not playing:
            return
        done_ms = _to_tick(self._s.robot_done_s)
        if self._last_robot_ms is None or ms - (self._last_robot_ms + PIECE_MS) >= done_ms:
            self._turns += 1
            self._turn_start_ms = ms
            self._turn_played_ms = 0
            self._backchannel_done = False
            self._backchannel_at_ms = None
            if "haan_haan" in self._plan.tricks:
                self._backchannel_at_ms = _to_tick(self._rng.uniform(*self._s.backchannel_at_s))
        self._turn_played_ms += PIECE_MS
        self._last_robot_ms = ms

    @property
    def _robot_playing(self) -> bool:
        return self._last_robot_ms == self._now_ms

    @property
    def _robot_quiet_ms(self) -> int:
        if self._last_robot_ms is None:
            return self._now_ms
        return self._now_ms - (self._last_robot_ms + PIECE_MS)

    @property
    def _buffered_ms(self) -> int:
        return self._playback.queued_pieces * PIECE_MS

    @property
    def _caller_speaking(self) -> bool:
        return bool(self._queue) or self._brain.speaking

    # ---- race lines (C10) ---------------------------------------------------------

    def _on_bell(self, bell: Bell, ms: int) -> None:
        drop = self._plan.drop
        if drop is not None and drop.moment == "while_action_runs" and bell.bell == SAVING:
            if self._race_comes_first(bell):
                # Part 5 L1, task 15: "ruko!" while saving, THEN the line dies.
                # Dropping now would cut the race line before it could play.
                self._drop_after_race = True
            else:
                self._drop("while_action_runs")
                return
        race, report = self._plan.race, self._report.race
        if race is None or bell.action != race.action or report.hit is not None:
            return
        if race.moment == "while_asking" and bell.bell == PREPARED and report.armed_ms is None:
            report.armed_ms = ms
            self._race_question_deadline_ms = ms + _to_tick(self._s.race_question_wait_s)
        elif race.moment == "right_after_sending" and bell.bell == ARRIVED and report.armed_ms is None:
            report.armed_ms = ms
            self._fire_race()
        elif race.moment == "while_running" and bell.bell == SAVING and report.armed_ms is None:
            report.armed_ms = ms
            self._race_fire_at_ms = ms + _to_tick(self._s.race_running_after_s)
        elif race.moment == "saved_not_heard" and bell.bell == SAVED and report.armed_ms is None:
            report.armed_ms = ms
            self._race_saved_deadline_ms = ms + _to_tick(self._plan.saved_gap_s)
            self._fire_race()
        elif race.moment in ("right_after_sending", "while_running") and bell.bell == SAVED:
            # The save landed. Said before it: hit. Not yet said: the moment is gone.
            if report.line_ms is not None:
                report.hit, report.why = True, "said before the save"
            else:
                report.hit, report.why = False, "the save came first"
                self._race_fire_at_ms = None
                if self._queue_kind == "race":
                    self._queue.clear()

    def _race_comes_first(self, bell: Bell) -> bool:
        """A while_running race on the same action is planned and not yet said."""
        race, report = self._plan.race, self._report.race
        return (
            race is not None
            and race.moment == "while_running"
            and race.action == bell.action
            and report.line_ms is None
            and report.hit is None
        )

    def _race_tick(self, ms: int) -> None:
        race, report = self._plan.race, self._report.race
        if race is None or report.hit is not None:
            return
        if self._race_fire_at_ms is not None and ms >= self._race_fire_at_ms:
            self._race_fire_at_ms = None
            self._fire_race()
        if race.moment == "while_asking" and report.armed_ms is not None and report.line_ms is None:
            in_question = self._robot_playing and self._turn_start_ms >= report.armed_ms
            if in_question and self._turn_played_ms >= _to_tick(self._s.race_asking_at_s) and not self._queue:
                self._fire_race()
            elif ms >= self._race_question_deadline_ms and self._queue_kind != "race":
                report.hit, report.why = False, "no yes-question of 1 s or more came"

    def _fire_race(self) -> None:
        race = self._plan.race
        self._play("race", race.line, then_wants=race.then_wants)

    def _race_line_started(self, ms: int) -> None:
        report, race = self._report.race, self._plan.race
        report.line_ms = ms
        if race.moment == "while_asking":
            report.hit = self._playback.is_playing or self._robot_playing
            report.why = "said while the question played" if report.hit else "the question had ended"
        elif race.moment == "saved_not_heard":
            report.hit = ms <= self._race_saved_deadline_ms
            report.why = "said before the reply came" if report.hit else "the reply had come"

    # ---- director tricks (C9) -----------------------------------------------------

    def _tricks_tick(self, ms: int) -> None:
        tricks = self._plan.tricks
        long_enough = self._turn_played_ms + self._buffered_ms
        if (
            "haan_haan" in tricks
            and self._robot_playing
            and not self._backchannel_done
            and self._backchannel_at_ms is not None
            and self._turn_played_ms >= self._backchannel_at_ms
            and long_enough >= _to_tick(self._s.backchannel_min_turn_s)
            and not self._caller_speaking
        ):
            self._backchannel_done = True
            self._play("backchannel", self._s.backchannel_line, stop_brain=False)
        if (
            "interrupts" in tricks
            and not self._interrupt_done
            and self._robot_playing
            and self._turn_played_ms >= _to_tick(self._s.interrupt_at_s)
            and long_enough >= _to_tick(self._s.interrupt_min_turn_s)
            and not self._queue
        ):
            self._interrupt_done = True
            self._play("interrupt", self._s.interrupt_line)
        if "goes_silent" in tricks:
            done_ms = _to_tick(self._s.robot_done_s)
            if (
                self._silence == "waiting"
                and self._turns >= self._s.silence_after_robot_turn
                and not self._robot_playing
                and self._robot_quiet_ms >= done_ms
            ):
                self._silence = "holding"
                self._hold_until_ms = ms + _to_tick(self._s.silence_s)
                self._brain.stop()
            elif self._silence == "holding" and ms >= self._hold_until_ms:
                self._silence = "done"
                self._play("silence_end", self._s.silence_end_line)

    @property
    def _holding(self) -> bool:
        return self._silence == "holding"

    # ---- drop tests (C12) ---------------------------------------------------------

    def _drop_tick(self, ms: int) -> None:
        drop = self._plan.drop
        if drop is None:
            return
        moment = drop.moment
        if self._drop_after_race:
            report = self._report.race
            race_line_done = report.line_ms is not None and self._queue_kind == "race" and not self._queue
            # Missed (the save came first): nothing left to wait for, drop now.
            if race_line_done or report.hit is False:
                self._drop(moment)
            return
        # A drop decided now takes effect on the next piece, PIECE_MS later
        # (see the timing rule above); conditions on time count that in.
        if moment == "random" and ms + PIECE_MS >= self._drop_at_ms:
            self._drop(moment)
        elif moment == "mid_robot_sentence":
            at = _to_tick(self._s.drop_mid_robot_s)
            if self._robot_playing and self._turn_played_ms >= at and self._buffered_ms >= at:
                self._drop(moment)
        elif moment == "mid_caller_sentence":
            at = _to_tick(self._s.drop_mid_sentence_s)
            left = (len(self._queue) if self._queue else self._brain.queued_pieces) * PIECE_MS
            if self._caller_speaking and self._caller_speech_ms >= at and left >= at:
                self._drop(moment)
        elif moment == "long_silence":
            quiet = _to_tick(self._s.drop_long_silence_s)
            at = ms + PIECE_MS  # when the drop would take effect
            robot_quiet = self._robot_quiet_ms + PIECE_MS
            caller_quiet = self._last_caller_speech_ms is None or at - (self._last_caller_speech_ms + PIECE_MS) >= quiet
            if self._turns >= 1 and not self._caller_speaking and robot_quiet >= quiet and caller_quiet:
                self._drop(moment)
        elif moment == "after_haan" and self._drop_after_speech and not self._caller_speaking:
            self._drop(moment)

    def _drop(self, moment: str) -> None:
        if self._dropped:
            return
        self._dropped = True
        kind = self._plan.drop.kind
        event = Dropped(moment, kind, self._now_ms + PIECE_MS)  # takes effect on the next piece
        self._report.dropped = event
        self._controls.hang_up(kind)
        self._on_event(event)

    # ---- speaking ------------------------------------------------------------------

    def _play(self, kind: str, line: str, then_wants: str | None = None, stop_brain: bool = True) -> None:
        if stop_brain:
            # A real person stops mid-sentence to shout "ruko!".
            self._brain.stop()
        self._queue = deque(self._recordings.pieces(line))
        self._queue_kind, self._queue_line, self._queue_then_wants = kind, line, then_wants
        self._queue_first_sent = False

    def _next_piece(self, tick: int) -> bytes:
        ms = tick * PIECE_MS
        if self._queue:
            piece = self._queue.popleft()
            if not self._queue_first_sent:
                self._queue_first_sent = True
                said = SaidLine(self._queue_kind, self._queue_line, ms, self._queue_then_wants)
                self._report.said.append(said)
                if self._queue_kind == "race":
                    self._race_line_started(ms)
                self._on_event(said)
            self._count_caller_speech(ms, True)
            return piece
        if self._holding:
            self._count_caller_speech(ms, False)
            return SILENT_PIECE
        speaking = self._brain.speaking
        piece = self._brain.next_piece(tick)
        self._count_caller_speech(ms, speaking)
        return piece

    def _count_caller_speech(self, ms: int, speaking: bool) -> None:
        if speaking:
            self._caller_speech_ms += PIECE_MS
            self._last_caller_speech_ms = ms
        else:
            self._caller_speech_ms = 0


class _DirectorMic:
    def __init__(self, director: Director) -> None:
        self._director = director

    def next_piece(self, tick: int) -> bytes:
        return self._director._next_piece(tick)


class _DirectorEar:
    def __init__(self, director: Director, inner: Ear) -> None:
        self._director = director
        self._inner = inner

    def heard(self, piece: bytes | None, caller_ms: int) -> None:
        self._inner.heard(piece, caller_ms)
        self._director._tick(piece, caller_ms)


class LineControls:
    """Controls for a real CallerLine. Bound after the line is made, because
    the line is made with the director's microphone and ear."""

    def __init__(self) -> None:
        self._line = None
        self._tasks: set[asyncio.Task] = set()

    def bind(self, line) -> None:
        self._line = line

    def hang_up(self, kind: str) -> None:
        if kind == "silent":
            self._line.go_silent()
            return
        # Called from inside the line's beat, which can't wait: hang up on the side.
        task = asyncio.create_task(self._line.hang_up("drop test"))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)


def caller_with_director(
    url: str,
    call_id: str,
    plan: DirectorPlan,
    brain_mic: BrainMic,
    recordings: Recordings,
    ear: Ear | None = None,
    on_event=None,
    settings: DirectorSettings = DirectorSettings(),
    tap=None,
    on_cleared=None,
) -> tuple[CallerLine, Director]:
    """Wire a caller line with the director in front of the brain's mouth and
    behind the caller's ear. The one place this wiring is done.

    `tap(tick, piece)` sees every piece the caller really sends, director
    lines included: the lab's recorder (Part 5, L5/L6) keeps the caller's
    side of the call from it."""
    playback = Playback()
    controls = LineControls()
    director = Director(plan, recordings, brain_mic, playback, controls, on_event, settings)
    mic = director.mic if tap is None else _TappedMic(director.mic, tap)
    line = CallerLine(url, call_id, mic, director.ear(ear or NullEar()), playback=playback, on_cleared=on_cleared)
    controls.bind(line)
    return line, director


class _TappedMic:
    def __init__(self, inner, tap) -> None:
        self._inner = inner
        self._tap = tap

    def next_piece(self, tick: int) -> bytes:
        piece = self._inner.next_piece(tick)
        self._tap(tick, piece)
        return piece


async def follow_bells(director: Director, bells: asyncio.Queue) -> None:
    """Moves one call's bells from the bell board to its director."""
    while True:
        heard = await bells.get()
        director.ring(heard.bell)


def _to_tick(seconds: float) -> int:
    """Seconds -> caller milliseconds, rounded to whole 20 ms pieces."""
    return round(seconds * 1000 / PIECE_MS) * PIECE_MS


def _has_words(text: str, words: tuple[str, ...]) -> bool:
    lowered = " ".join(text.lower().split())
    return any(re.search(rf"(?<![a-z]){re.escape(word)}(?![a-z])", lowered) for word in words)
