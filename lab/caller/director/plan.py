"""What the director will do in one call, and its timing settings."""

from __future__ import annotations

from dataclasses import dataclass

from engine.lab_controls.switches import AFTER_SAVE, REPLY_LOST, SLOW
from engine.tasks.caller_spec import DropSpec, RaceSpec
from engine.tasks.task_file import Task

# The tricks the director performs itself (C7); the brain does the others.
DIRECTOR_TRICKS = ("haan_haan", "interrupts", "goes_silent")


@dataclass(frozen=True)
class DirectorSettings:
    """Every timing number of the director, in one place (C18). Starting
    guesses from C9-C12, to be tuned from real calls."""

    # C9: the robot is "done" after this much quiet; shorter gaps are pauses inside a turn.
    robot_done_s: float = 0.8
    # C9 trick 2: "haan haan" 1-2 s into a robot turn longer than ~3 s.
    backchannel_min_turn_s: float = 3.0
    backchannel_at_s: tuple[float, float] = (1.0, 2.0)
    # C9 trick 3: one planned interruption, 1 s into the first robot turn of 2 s or more.
    interrupt_min_turn_s: float = 2.0
    interrupt_at_s: float = 1.0
    # C9 trick 7: after the robot's 2nd turn the caller goes quiet this long,
    # longer than the robot's 6 s "aap sun rahe hain?" (decision 13) and well
    # under its 21 s hang-up.
    silence_after_robot_turn: int = 2
    silence_s: float = 10.0
    # C10 race timing.
    race_asking_at_s: float = 1.0  # moment 1: ~1 s into the yes-question
    race_running_after_s: float = 0.3  # moment 3: inside the 0.8 s wait before the save
    race_question_wait_s: float = 10.0  # moment 1 is missed if no question of 1 s+ comes by then
    # C12 drop moments.
    drop_mid_sentence_s: float = 0.3  # this far into a sentence, with as much still to come
    drop_mid_robot_s: float = 0.5
    drop_long_silence_s: float = 2.0
    drop_random_window_s: tuple[float, float] = (5.0, 60.0)
    # Words that tell the director the robot said the job is done ("after_done").
    done_words: tuple[str, ...] = ("ho gaya", "ho chuka", "done")
    # Words that make the caller's line a yes ("after_haan").
    yes_words: tuple[str, ...] = ("haan", "ji haan", "yes", "theek hai", "kar do")

    # The director's own lines (in the caller's voice; C4).
    backchannel_line: str = "haan haan"
    interrupt_line: str = "ek minute, suniye..."
    silence_end_line: str = "hello? sun rahe ho?"


@dataclass(frozen=True)
class DirectorPlan:
    race: RaceSpec | None
    drop: DropSpec | None
    tricks: tuple[str, ...]  # only the director's tricks
    seed: int
    # For a saved_not_heard race: how long the reply stays lost or slow after
    # the save, which is the window the line must land in (C10).
    saved_gap_s: float = 0.0

    @staticmethod
    def from_task(task: Task, seed: int) -> DirectorPlan:
        caller_tricks = task.caller.tricks if task.caller else ()
        gap = 0.0
        if task.race is not None and task.race.moment == "saved_not_heard":
            gaps = [
                s.seconds
                for s in task.switches
                if s.action == task.race.action
                and (s.kind == REPLY_LOST or (s.kind == SLOW and s.delay_at == AFTER_SAVE))
            ]
            gap = max(gaps)  # the loader guarantees one exists
        return DirectorPlan(
            race=task.race,
            drop=task.drop,
            tricks=tuple(t for t in caller_tricks if t in DIRECTOR_TRICKS),
            seed=seed,
            saved_gap_s=gap,
        )
