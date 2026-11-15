"""The Thinker's loop (decision 7): decide -> press tools -> read the answers -> decide
again, until a step presses no tool. Then wait for news (the caller's next words, a
change's result) and go again, until the job closes or ends.

It runs as a crash-safe task in the call box: a crash ends the job, not the call,
and the Talker apologises and offers a human. Every wait has a time limit.
"""

from __future__ import annotations

import json
import time
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any, Protocol

from agents.agent_c.job import Job, StepMark
from agents.agent_c.live_retry import try_live
from agents.agent_c.notebook import CALLER, FAILED, NEED_YES, REQUEST, RESULT, YES_ANSWER, Note, Notebook
from agents.agent_c.thinker.brain import Step, ThinkerBrain, ToolCall
from agents.agent_c.thinker.copy_notes import note_for_reply, note_for_stop
from agents.common.diary import Diary
from agents.common.tools import Tools
from engine.service.replies import ERROR, ToolReply

# What the Thinker reads from the notebook, and how each line starts for it.
NEWS = {REQUEST: "Request", CALLER: "Caller", YES_ANSWER: "Caller's answer to the yes-question", RESULT: "Result"}
THINKER_FAILED = "thinker failed"
GOING_ROUND = "going round in circles"


@dataclass(frozen=True)
class ThinkerLimits:
    step_s: float  # one Thinker step (decision 30: ~8 s)
    step_retries: int  # then a "failed" note and a hand-over
    lookup_s: float  # one lookup or prepare tool (decision 30: 5 s)
    idle_wait_s: float  # how long one wait for news lasts before looking again
    lookup_retries: int = 0  # decision 19: a lookup that failed or timed out is tried once more
    same_call_limit: int = 0  # decision 19: this many identical tool, inputs and answer in a row -> hand over (0: off)

    @classmethod
    def from_settings(cls, section: dict[str, Any]) -> ThinkerLimits:
        return cls(float(section["step_s"]), int(section["step_retries"]), float(section["lookup_s"]),
                   float(section["idle_wait_s"]), int(section["lookup_retries"]), int(section["same_call_limit"]))


class JobHooks(Protocol):
    """The Thinker's side of the job desk."""

    def mark(self) -> StepMark: ...

    def stopped(self, mark: StepMark, reply: Note | None) -> None: ...

    async def prepared(self, note: Note) -> None:
        """A change was prepared: the Guard takes it from here."""
        ...

    async def hand_over(self, reason: str, order_no: Any = None) -> ToolReply:
        """The call's one hand-over (agents/agent_c/hand_over.py)."""
        ...


class Thinker:
    def __init__(self, job: Job, notebook: Notebook, brain: ThinkerBrain, tools: Tools, allowed: Collection[str],
                 hooks: JobHooks, limits: ThinkerLimits, diary: Diary, hand_over_tool: str = "") -> None:
        self._job = job
        self._notebook = notebook
        self._brain = brain
        self._tools = tools
        self._allowed = set(allowed)
        self._hooks = hooks
        self._limits = limits
        self._diary = diary
        self._read = 0  # how far into the notebook the Thinker has read
        self._hand_over_tool = hand_over_tool  # pressed through code, so a call hands over only once
        self._same: tuple[str, int] = ("", 0)  # the last tool call with its answer, and how many times in a row

    async def run(self) -> None:
        results: list[tuple[ToolCall, dict[str, Any]]] = []
        while self._job.live:
            news = self._news()
            mark = self._hooks.mark()
            step = await self._step(news, results)
            if step is None:
                return
            if step.calls:
                results = [await self._press(call, mark.version) for call in step.calls]
                if self._going_round():
                    self._job.end(GOING_ROUND)  # first, so the job's end names the reason
                    await self._hooks.hand_over("The back office kept getting the same answer.")
                    return
                continue
            results = []
            stop = note_for_stop(step.text)
            reply = self._notebook.add(stop[0], stop[1], mark.version) if stop else None
            self._hooks.stopped(mark, reply)
            await self._wait_for_news()

    def _news(self) -> list[str]:
        notes = self._notebook.after(self._read)
        self._read = self._notebook.newest
        return [f"{NEWS[note.kind]}: {note.text}" for note in notes if note.kind in NEWS]

    def _has_news(self) -> bool:
        return any(note.kind in NEWS for note in self._notebook.after(self._read))

    async def _wait_for_news(self) -> None:
        while self._job.live and not self._has_news():
            await self._notebook.wait(self._limits.idle_wait_s)

    async def _step(self, news: list[str], results: list[tuple[ToolCall, dict[str, Any]]]) -> Step | None:
        started = time.perf_counter()
        try:
            step = await try_live(lambda: self._brain.step(news, results), self._limits.step_s, self._limits.step_retries,
                                  on_retry=lambda error: self._diary.error(f"thinker step retried: {error!r}"))
        except Exception as error:
            # Decision 30: one retry, then a "failed" note and a hand-over; never a guess.
            self._diary.error(f"thinker step failed: {error!r}")
            self._notebook.add(FAILED, "The back office could not finish this request.", self._job.version)
            self._job.end(THINKER_FAILED)
            await self._hooks.hand_over("The back office could not finish the request.")
            return None
        self._diary.step("thinker", (time.perf_counter() - started) * 1000, tools=[call.name for call in step.calls])
        return step

    async def _press(self, call: ToolCall, version: int) -> tuple[ToolCall, dict[str, Any]]:
        started = time.perf_counter()
        if call.name not in self._allowed:
            # Confirm is pressed by code after a real yes, never by the Thinker (decision 17).
            reply = ToolReply(ERROR, f"{call.name} is not one of your tools.")
        elif call.name == self._hand_over_tool:
            reply = await self._hooks.hand_over(str(call.inputs.get("reason") or "the caller needs a human"),
                                                call.inputs.get("order_no"))
        else:
            reply = await self._lookup(call)
        self._diary.tool(call.name, dict(call.inputs), reply.status, (time.perf_counter() - started) * 1000,
                         message=reply.message, by="thinker")
        kind, text, data = note_for_reply(call, reply)
        note = self._notebook.add(kind, text, version, data)
        if kind == NEED_YES:
            await self._hooks.prepared(note)
        answer = {"status": reply.status, "message": reply.message, "data": dict(reply.data)}
        seen = json.dumps([call.name, call.inputs, answer], sort_keys=True, ensure_ascii=False, default=str)
        self._same = (seen, self._same[1] + 1 if self._same[0] == seen else 1)
        return call, answer

    async def _lookup(self, call: ToolCall) -> ToolReply:
        """Lookups and prepares change nothing, so a failed or timed-out one is simply
        tried again, once (decision 19). A refusal is an answer: never retried."""
        reply = ToolReply(ERROR, "not tried")
        for attempt in range(1 + self._limits.lookup_retries):
            try:
                reply = await try_live(lambda: self._tools.call(call.name, dict(call.inputs)), self._limits.lookup_s, 0)
            except Exception as error:
                reply = ToolReply(ERROR, f"No answer from the service ({type(error).__name__}). The result is unknown.")
            if reply.status != ERROR:
                break
            if attempt < self._limits.lookup_retries:
                self._diary.error(f"{call.name} failed, trying once more: {reply.message}")
        return reply

    def _going_round(self) -> bool:
        """The same tool, inputs and answer, again and again: nothing new is being learned."""
        return 0 < self._limits.same_call_limit <= self._same[1]
