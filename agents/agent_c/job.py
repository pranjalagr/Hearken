"""A job: one piece of back-office work the Talker asked for, and when it may close
(decision 7).

Agent A told callers "ho gaya" while a change was still in flight, or after the
caller had added something it never read (practice run 7, 15 Nov 2026). So a job
closes only when the Thinker's step pressed no tool AND five code checks pass:
1. that step read the newest caller turn;
2. no caller turn began after it started;
3. nothing waits for the caller: the last "need info"/"need yes" note has a caller
   answer after it;
4. no change is running;
5. the final reply was fully spoken (played to the end, not cut off).
The first time they pass the job is at "answer given": the caller's next turn still
goes to the Thinker, and only a second stop with the checks passing closes it.
"""

from __future__ import annotations

from dataclasses import dataclass

from agents.agent_c.notebook import CALLER, NEED_INFO, NEED_YES, YES_ANSWER, Notebook

OPEN = "open"
ANSWER_GIVEN = "answer given"
CLOSED = "closed"
ENDED = "ended"  # hand-over, a Thinker crash or hang-up: at once, no checks

CHECKS = ("read the newest turn", "no turn since", "nothing waits for the caller", "no change running",
          "reply fully spoken")


class Turns:
    """The caller's turns, numbered. `begin` comes when the caller starts speaking,
    `finish` with the finished words."""

    def __init__(self) -> None:
        self.begun = 0
        self.newest = 0  # the newest finished turn
        self.speaking = False

    def begin(self) -> None:
        if not self.speaking:
            self.begun += 1
            self.speaking = True

    def finish(self) -> int:
        if not self.speaking:
            self.begun += 1  # words with no "began" seen first (the ears were faster)
        self.speaking = False
        self.newest += 1
        return self.newest


@dataclass(frozen=True)
class StepMark:
    """What the world looked like when a Thinker step began."""

    read_turn: int  # the newest caller turn the step had read
    begun: int  # turns begun so far
    version: int


class Job:
    def __init__(self, number: int, request: str) -> None:
        self.number = number
        self.request = request
        self.version = 1
        self.state = OPEN
        self.changes_running = 0
        self.end_reason: str | None = None

    @property
    def live(self) -> bool:
        return self.state in (OPEN, ANSWER_GIVEN)

    def bump(self) -> int:
        """A new version (decision 16): the caller finished a turn, or said stop."""
        self.version += 1
        return self.version

    def change_sent(self) -> None:
        self.changes_running += 1

    def change_answered(self) -> None:
        """An answer came, or the box gave up on one ("result unknown")."""
        self.changes_running = max(0, self.changes_running - 1)

    def end(self, reason: str) -> None:
        if self.live:
            self.state = ENDED
            self.end_reason = reason


def failing_checks(job: Job, step: StepMark, turns: Turns, notebook: Notebook, reply_spoken: bool) -> list[str]:
    """The close checks that fail now, by name; empty when the job may move on."""
    failing = []
    if step.read_turn != turns.newest:
        failing.append(CHECKS[0])
    if turns.begun != step.begun or turns.speaking:
        failing.append(CHECKS[1])
    asked = notebook.last((NEED_INFO, NEED_YES))
    answered = notebook.last((CALLER, YES_ANSWER))
    if asked is not None and (answered is None or answered.number < asked.number):
        failing.append(CHECKS[2])
    if job.changes_running:
        failing.append(CHECKS[3])
    if not reply_spoken:
        failing.append(CHECKS[4])
    return failing
