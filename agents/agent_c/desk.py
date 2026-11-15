"""The job desk: opens jobs for the Talker, passes every caller turn to the Thinker
while a job is open, and moves a job on only when the close checks pass (decision 7).

Plain code: it holds the job's state, numbers the caller's turns and bumps the
version (decision 16). It starts the Thinker through a function it is given, so tests
run it without a call box or a brain.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from agents.agent_c.job import ANSWER_GIVEN, CLOSED, OPEN, Job, StepMark, Turns, failing_checks
from agents.agent_c.notebook import CALLER, FAILED, REQUEST, Note, Notebook
from agents.common.diary import Diary

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

    # ---- the caller's side ------------------------------------------------------

    def caller_began(self) -> None:
        self._turns.begin()

    def caller_finished(self, words: str) -> None:
        """Every finished turn during a job goes to the Thinker, even "haan haan": no
        word list decides what matters (decision 7). It also bumps the version, so
        notes written before these words wait (decision 16)."""
        self._turns.finish()
        job = self.job
        if job is not None and job.live:
            job.bump()
            self._notebook.add(CALLER, words, job.version)
        else:
            self._notebook.ring()  # a note held while the caller spoke may go now

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
