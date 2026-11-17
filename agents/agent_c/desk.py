"""The job desk: opens jobs for the Talker, passes every caller turn to the Thinker
while a job is open, and moves a job on only when the close checks pass (decision 7).

Plain code: it holds the job's state, numbers the caller's turns and bumps the
version (decision 16). It starts the Thinker through a function it is given, so tests
run it without a call box or a brain.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from agents.agent_c.job import ANSWER_GIVEN, CLOSED, OPEN, Job, StepMark, Turns, failing_checks
from agents.agent_c.notebook import CALLER, FAILED, REQUEST, Note, Notebook
from agents.common.diary import Diary
from engine.service.replies import ERROR, ToolReply

STARTED = "started: the back office is on it; say you are checking"
PASSED_ON = "passed on: the back office is already on this call's request and has these words too"
THINKER_CRASHED = "thinker crashed"


class JobDesk:
    def __init__(self, notebook: Notebook, turns: Turns, start_thinker: Callable[[Job, JobDesk], None],
                 diary: Diary) -> None:
        self._notebook = notebook
        self._turns = turns
        self.start_thinker = start_thinker
        self._diary = diary
        self._jobs = 0
        self.job: Job | None = None
        self._waiting_reply: tuple[StepMark, int] | None = None  # (step, reply note) waiting to be spoken
        self.on_prepared: Callable[[Note], Awaitable[None]] | None = None  # the Guard, from build step 4
        self.on_hand_over: Callable[[str, Any], Awaitable[ToolReply]] | None = None  # the call's one hand-over
        # Decision 27's steps, run by code (agents/agent_c/order_ladder.py); None: not asked again.
        self.on_missed: Callable[[str], Awaitable[bool]] | None = None
        self.name_heard: Callable[[str], bool] = lambda name: True  # the name check; on in a real call
        self.caller_words: list[str] = []  # every finished caller turn, job or not, for the name check

    # ---- the Talker's side ------------------------------------------------------

    def open_job(self, request: str) -> str:
        """Only the Talker opens a job (ask_thinker). One job at a time: a second
        request while one is open goes to that job's Thinker."""
        if self.job is not None and self.job.live:
            self._notebook.add(REQUEST, request, self.job.version)
            return PASSED_ON
        self._jobs += 1
        self.job = Job(self._jobs, request)
        self._waiting_reply = None
        self._notebook.add(REQUEST, request, self.job.version)
        self._diary.decision("job opened", job=self.job.number, request=request)
        self.start_thinker(self.job, self)
        return STARTED

    def why_not_end(self) -> str | None:
        """Why the Talker may not hang up yet, or None. A goodbye while the back office
        still works would drop a caller whose change is saving or whose answer isn't in."""
        if self.job is None:
            return None
        if self.job.changes_running:
            return "a change is still saving; wait for its note"
        if self.job.state == OPEN:
            return "the back office is still working on the caller's request; wait for its note"
        return None

    # ---- the caller's side ------------------------------------------------------

    def caller_began(self) -> None:
        self._turns.begin()

    def caller_finished(self, words: str) -> None:
        """Every finished turn during a job goes to the Thinker, even "haan haan": no
        word list decides what matters (decision 7). It also bumps the version, so
        notes written before these words wait (decision 16)."""
        self._turns.finish()
        self.caller_words.append(words)
        job = self.job
        if job is not None and job.live:
            job.bump()
            self._notebook.add(CALLER, words, job.version)
        else:
            self._notebook.ring()  # a note held while the caller spoke may go now

    def robot_stopped(self) -> None:
        """The stop-or-keep model stopped the voice (decision 16): what was being said
        is now old, so notes wait for the Thinker's "still good" or its redone work."""
        if self.job is not None and self.job.live:
            self.job.bump()

    def reply_spoken(self, up_to_note: int) -> None:
        """The voice played the Talker's reply to the notes up to this one, to the end."""
        if self._waiting_reply is None or self._waiting_reply[1] > up_to_note:
            return
        mark = self._waiting_reply[0]
        self._waiting_reply = None
        self._try_to_move_on(mark, reply_spoken=True)

    # ---- the Thinker's side (JobHooks) ------------------------------------------

    def mark(self) -> StepMark:
        assert self.job is not None
        return StepMark(self._turns.newest, self._turns.begun, self.job.version)

    def stopped(self, mark: StepMark, reply: Note | None) -> None:
        """A Thinker step pressed no tool. With a reply to speak, check 5 waits for
        the voice; with none, there is nothing to wait for."""
        if reply is None:
            self._try_to_move_on(mark, reply_spoken=True)
            return
        failing = self._failing(mark, reply_spoken=True)  # checks 1-4 now, check 5 once it is spoken
        if failing:
            self._diary.decision("job stays open", job=self.job.number, failing=failing)
            return
        self._waiting_reply = (mark, reply.number)

    async def prepared(self, note: Note) -> None:
        if self.on_prepared is not None:
            await self.on_prepared(note)

    async def missed(self, problem: str) -> bool:
        return await self.on_missed(problem) if self.on_missed is not None else False

    def caller_said(self, name: str) -> bool:
        return self.name_heard(name)

    def typed_number(self, number: str) -> None:
        """The caller typed the order number on the keypad: it reaches the Thinker as the
        caller's words, which can't be misheard."""
        text = f"(typed on the keypad) order number {number}"
        self.caller_words.append(text)
        if self.job is not None and self.job.live:
            self.job.bump()
            self._notebook.add(CALLER, text, self.job.version)

    async def hand_over(self, reason: str, order_no: Any = None) -> ToolReply:
        if self.on_hand_over is None:
            self.end_job(reason)
            return ToolReply(ERROR, "No hand-over on this line.")
        return await self.on_hand_over(reason, order_no)

    # ---- ending -----------------------------------------------------------------

    def end_job(self, reason: str) -> None:
        """Hand-over, a Thinker crash, hang-up: the job ends at once, no checks."""
        if self.job is not None and self.job.live:
            self.job.end(reason)
            self._diary.decision("job ended", job=self.job.number, reason=reason)
            self._notebook.ring()  # wake the Thinker's wait so it sees the end

    def on_crash(self, task: str, error: BaseException) -> None:
        """The call box reports a crash of a crash-safe task (the Thinker)."""
        if self.job is not None and self.job.live:
            self._notebook.add(FAILED, "The back office stopped working on this request.", self.job.version)
        self.end_job(THINKER_CRASHED)

    # ---- inside -----------------------------------------------------------------

    def _failing(self, mark: StepMark, reply_spoken: bool) -> list[str]:
        assert self.job is not None
        return failing_checks(self.job, mark, self._turns, self._notebook, reply_spoken)

    def _try_to_move_on(self, mark: StepMark, reply_spoken: bool) -> None:
        job = self.job
        if job is None or not job.live:
            return
        failing = self._failing(mark, reply_spoken)
        if failing:
            self._diary.decision("job stays open", job=job.number, failing=failing)
            return
        job.state = ANSWER_GIVEN if job.state == OPEN else CLOSED
        self._diary.decision(f"job {job.state}", job=job.number)
        if job.state == CLOSED:
            self._notebook.ring()  # the Thinker's wait ends with the job
