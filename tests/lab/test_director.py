"""The director, tick by tick, with a simulated robot (Part 3, C9-C12).

Each test runs the caller's 20 ms beat by hand: the director's microphone
gives the piece the caller sends, then the robot's audio plays and the
director's ear sees it. Times are exact: tick t starts at caller_ms 20 t.
"""

import random

import pytest

from common.wire.caller_line import QueueMic, SilentMic
from common.wire.playback import Playback
from common.wire.protocol import PIECE_BYTES, SILENT_PIECE
from engine.lab_controls.bells import ARRIVED, PREPARED, SAVED, SAVING, Bell
from engine.tasks.caller_spec import DropSpec, RaceSpec
from lab.caller.director.director import Director, Dropped, SaidLine
from lab.caller.director.plan import DirectorPlan, DirectorSettings

SETTINGS = DirectorSettings()
LINE_PIECES = 10  # every recorded line lasts 0.2 s in these tests


def line_piece(line: str, n: int) -> bytes:
    return (line[:20].encode().ljust(20, b".") + bytes([n])).ljust(PIECE_BYTES, b"\x01")


class Lines:
    def pieces(self, line):
        return [line_piece(line, n) for n in range(LINE_PIECES)]


class Controls:
    def __init__(self):
        self.calls = []

    def hang_up(self, kind):
        self.calls.append(kind)


class Call:
    """One simulated call: the director, a speaker, and the robot's audio."""

    def __init__(self, race=None, drop=None, tricks=(), seed=7, saved_gap_s=0.0, brain=None):
        self.plan = DirectorPlan(race=race, drop=drop, tricks=tuple(tricks), seed=seed, saved_gap_s=saved_gap_s)
        self.playback = Playback()
        self.brain = brain or QueueMic()
        self.controls = Controls()
        self.events = []
        self.director = Director(self.plan, Lines(), self.brain, self.playback, self.controls, self.events.append)
        self.ear = self.director.ear(_NullEar())
        self.sent: dict[int, bytes] = {}
        self.tick = 0

    def robot_says(self, seconds: float):
        for _ in range(round(seconds * 50)):
            self.playback.add_audio(b"\x05" * PIECE_BYTES)

    def run(self, ticks: int, bells_at: dict | None = None):
        bells_at = bells_at or {}
        for _ in range(ticks):
            t = self.tick
            self.sent[t] = self.director.mic.next_piece(t)
            for bell in bells_at.get(t, []):
                self.director.ring(bell)
            played, _ = self.playback.tick(t * 20)
            self.ear.heard(played, t * 20)
            self.tick += 1

    def ticks_saying(self, line: str) -> list[int]:
        return [t for t, piece in sorted(self.sent.items()) if piece[:20] == line[:20].encode().ljust(20, b".")]


class _NullEar:
    def heard(self, piece, caller_ms):
        pass


def bell(name: str, action: str = "cancel_order") -> Bell:
    return Bell(name, action, "c", 0.0, 0)


RUKO = "ruko, cancel mat karo!"


def race(moment: str) -> RaceSpec:
    return RaceSpec(moment, "cancel_order", RUKO, "keep the order")


# ---- race moments (C10) -------------------------------------------------------------


def test_while_running_fires_0_3_s_after_saving_and_hits_before_the_save():
    call = Call(race=race("while_running"))
    call.run(120, bells_at={50: [bell(ARRIVED), bell(SAVING)], 90: [bell(SAVED)]})
    report = call.director.report.race
    assert report.armed_ms == 1000 and report.line_ms == 1320  # fired at +300 ms, sent on the next piece
    assert report.hit is True and report.why == "said before the save"
    assert call.ticks_saying(RUKO) == list(range(66, 76))
    assert SaidLine("race", RUKO, 1320, "keep the order") in call.events


def test_while_running_is_missed_when_the_save_comes_first():
    call = Call(race=race("while_running"))
    call.run(100, bells_at={50: [bell(SAVING)], 55: [bell(SAVED)]})
    report = call.director.report.race
    assert report.hit is False and report.why == "the save came first"
    assert call.ticks_saying(RUKO) == []  # nothing was said after the moment had passed


def test_while_running_hits_when_the_action_is_stopped_and_never_saved():
    """The success case of a race: "ruko" landed, the robot stopped, no save ever came."""
    call = Call(race=race("while_running"))
    call.run(120, bells_at={50: [bell(SAVING)]})
    assert call.director.report.race.hit is True


def test_right_after_sending_fires_on_the_next_piece():
    call = Call(race=race("right_after_sending"))
    call.run(80, bells_at={20: [bell(ARRIVED), bell(SAVING)], 60: [bell(SAVED)]})
    report = call.director.report.race
    assert report.line_ms == 420 and report.hit is True


def test_right_after_sending_is_missed_if_the_save_arrives_with_it():
    call = Call(race=race("right_after_sending"))
    call.run(40, bells_at={20: [bell(ARRIVED), bell(SAVING), bell(SAVED)]})
    assert call.director.report.race.hit is False and call.ticks_saying(RUKO) == []


def test_while_asking_fires_1_s_into_the_question_while_it_plays():
    call = Call(race=race("while_asking"))
    call.run(20, bells_at={10: [bell(PREPARED)]})
    call.robot_says(4.0)  # the yes-question starts at tick 20
    call.run(100)
    report = call.director.report.race
    assert report.armed_ms == 200 and report.line_ms == 1400  # the question began at 400 ms: exactly 1 s in
    assert report.hit is True and report.why == "said while the question played"


def test_while_asking_ignores_a_turn_that_began_before_the_bell():
    call = Call(race=race("while_asking"))
    call.robot_says(3.0)  # a filler already playing when the bell rings
    call.run(200, bells_at={10: [bell(PREPARED)]})
    assert call.director.report.race.line_ms is None


def test_while_asking_is_missed_when_no_question_comes():
    call = Call(race=race("while_asking"))
    call.run(600, bells_at={10: [bell(PREPARED)]})  # 12 s, no robot audio at all
    report = call.director.report.race
    assert report.hit is False and report.why == "no yes-question of 1 s or more came"


def test_saved_not_heard_fires_at_once_inside_the_lost_reply_gap():
    call = Call(race=race("saved_not_heard"), saved_gap_s=2.0)
    call.run(60, bells_at={30: [bell(SAVED)]})
    report = call.director.report.race
    assert report.line_ms == 620 and report.hit is True


def test_bells_for_another_action_are_ignored():
    call = Call(race=race("while_running"))
    call.run(100, bells_at={50: [bell(SAVING, "refund_order")], 90: [bell(SAVED, "refund_order")]})
    assert call.director.report.race.armed_ms is None


# ---- director tricks (C9) -----------------------------------------------------------


def test_haan_haan_once_per_long_robot_turn_between_1_and_2_s_in():
    call = Call(tricks=["haan_haan"])
    call.robot_says(4.0)
    call.run(300)
    said = call.ticks_saying("haan haan")
    assert len(said) == LINE_PIECES  # exactly once
    expected_at = round(random.Random(7).uniform(1.0, 2.0) * 50) * 20  # the same seeded draw
    assert 1000 <= expected_at <= 2000
    assert said[0] * 20 == expected_at  # the turn began at 0 ms


def test_no_haan_haan_in_a_short_turn_or_over_the_callers_own_speech():
    short = Call(tricks=["haan_haan"])
    short.robot_says(2.0)
    short.run(200)
    assert short.ticks_saying("haan haan") == []

    busy = Call(tricks=["haan_haan"])
    busy.brain.say([b"\x09" * PIECE_BYTES] * 300)  # the caller is talking the whole time
    busy.robot_says(4.0)
    busy.run(200)
    assert busy.ticks_saying("haan haan") == []


def test_one_planned_interruption_stops_the_brain_mid_sentence():
    call = Call(tricks=["interrupts"])
    call.robot_says(3.0)
    call.run(30)
    call.brain.say([b"\x09" * PIECE_BYTES] * 500)  # the brain was about to talk
    call.run(200)
    said = call.ticks_saying("ek minute, suniye...")
    assert said and said[0] == 50  # exactly 1 s into the robot's turn (it began at 0 ms)
    assert call.brain.queued_pieces == 0  # its sentence was dropped, as a person would
    call.robot_says(3.0)
    call.run(200)
    assert len(call.ticks_saying("ek minute, suniye...")) == LINE_PIECES  # only once per call


def test_goes_silent_after_the_robots_second_turn_then_asks_if_anyone_is_there():
    call = Call(tricks=["goes_silent"])
    call.robot_says(1.0)  # turn 1: ticks 0-49
    call.run(100)
    call.robot_says(1.0)  # turn 2: ticks 100-149
    call.run(60)  # 1.2 s: turn 2 is over after 0.8 s of quiet (tick 190)
    call.brain.say([b"\x09" * PIECE_BYTES] * 50)  # the brain wants to answer
    call.run(600)
    sent_before_line = [call.sent[t] for t in range(191, 691)]
    assert all(piece == SILENT_PIECE for piece in sent_before_line)  # 10 s of silence
    said = call.ticks_saying("hello? sun rahe ho?")
    assert said[0] == 691  # right after the 10 s hold


# ---- drops (C12) --------------------------------------------------------------------


def test_a_random_drop_comes_at_the_seeded_moment():
    call = Call(drop=DropSpec("random", "silent"), seed=11)
    call.run(3200)
    expected = round(random.Random(11).uniform(5.0, 60.0) * 50) * 20
    assert call.controls.calls == ["silent"]
    assert call.director.report.dropped == Dropped("random", "silent", expected)


def test_drop_mid_robot_sentence():
    call = Call(drop=DropSpec("mid_robot_sentence", "clean"))
    call.robot_says(2.0)
    call.run(100)
    assert call.controls.calls == ["clean"] and call.director.report.dropped.caller_ms == 500  # 0.5 s in


def test_drop_mid_caller_sentence():
    call = Call(drop=DropSpec("mid_caller_sentence", "silent"))
    call.brain.say([b"\x09" * PIECE_BYTES] * 50)  # a 1 s line
    call.run(60)
    assert call.director.report.dropped.caller_ms == 300  # 0.3 s into it


def test_drop_in_a_long_silence_after_the_robot_spoke():
    call = Call(drop=DropSpec("long_silence", "clean"))
    call.robot_says(0.5)
    call.run(200)
    assert call.director.report.dropped.caller_ms == 2500  # 2 s after the robot's last piece ended (500 ms)


def test_drop_right_after_the_caller_says_haan():
    call = Call(drop=DropSpec("after_haan", "clean"))
    call.brain.say([b"\x09" * PIECE_BYTES] * 25)
    call.director.notify_caller_said("Haan ji, kar do")
    call.run(60)
    assert call.director.report.dropped.caller_ms == 500  # the moment the line ended


def test_drop_while_the_action_runs_and_after_the_robot_says_done():
    running = Call(drop=DropSpec("while_action_runs", "silent"))
    running.run(40, bells_at={20: [bell(ARRIVED), bell(SAVING)]})
    assert running.director.report.dropped.caller_ms == 420  # the piece right after the bell's tick

    done = Call(drop=DropSpec("after_done", "clean"))
    done.run(10)
    done.director.notify_robot_said("Aapka order 4521 cancel ho gaya hai.")
    assert done.controls.calls == ["clean"]


def test_after_a_drop_the_director_does_nothing_more():
    # The drop comes at "saving"; the race would only fire at "saved", after it.
    call = Call(race=race("saved_not_heard"), drop=DropSpec("while_action_runs", "silent"))
    call.run(100, bells_at={20: [bell(SAVING)], 40: [bell(SAVED)]})
    assert call.ticks_saying(RUKO) == [] and call.controls.calls == ["silent"]


def test_ruko_while_saving_then_the_line_dies():
    """Part 5 L1, task 15: both planned for the same moment, the race line
    goes out first and the drop follows it; dropping at "saving" would cut
    the "ruko" before it could play."""
    call = Call(race=race("while_running"), drop=DropSpec("while_action_runs", "silent"))
    call.run(100, bells_at={20: [bell(SAVING)]})
    said = call.ticks_saying(RUKO)
    assert said and call.controls.calls == ["silent"]
    assert call.director.report.dropped.caller_ms == (said[-1] + 1) * 20  # right after its last piece
    assert call.director.report.race.hit is True


def test_ruko_missed_while_saving_drops_at_once():
    """The save came before the race line was due: nothing left to wait for."""
    call = Call(race=race("while_running"), drop=DropSpec("while_action_runs", "silent"))
    call.run(100, bells_at={20: [bell(SAVING), bell(SAVED)]})
    assert call.ticks_saying(RUKO) == [] and call.controls.calls == ["silent"]
    assert call.director.report.race.hit is False


def test_the_same_seed_gives_the_same_call():
    def stream(seed):
        call = Call(tricks=["haan_haan"], drop=DropSpec("random", "clean"), seed=seed)
        call.robot_says(5.0)
        call.run(3200)
        return call.sent, call.director.report.dropped

    assert stream(3) == stream(3)
    assert stream(3) != stream(4)


def test_a_director_with_no_plan_just_passes_the_brain_through():
    brain = QueueMic()
    brain.say([b"\x09" * PIECE_BYTES] * 3)
    call = Call(brain=brain)
    call.run(5)
    assert [call.sent[t] for t in range(5)] == [b"\x09" * PIECE_BYTES] * 3 + [SILENT_PIECE] * 2


def test_silent_mic_brain_is_never_speaking():
    assert SilentMic().next_piece(0) == SILENT_PIECE
