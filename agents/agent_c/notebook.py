"""The notebook: what the Thinker and the backend have found, for the Talker to say
(decision 7).

A plain in-memory list per call, with a bell rung on each new note. No queue with a
size limit and no locks: everything runs on one event loop, so a list is enough, and
every wait on the bell has a time limit. Code writes the notes, never an AI's memory
of them: every backend answer is copied here as it came.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

# The kinds of note. Who writes each is fixed: code copies backend answers (FACT,
# NEED_YES, RESULT); the Thinker's own words are ANSWER, NEED_INFO or STILL_GOOD.
REQUEST = "request"        # the Talker opened a job (its words)
CALLER = "caller"          # a finished caller turn, passed to the Thinker during a job
FACT = "fact"              # a backend answer, copied by code
NEED_INFO = "need_info"    # the Thinker needs something from the caller
NEED_YES = "need_yes"      # a prepared change waits for the caller's yes
RESULT = "result"          # what the backend answered to a change (or that it never did)
ANSWER = "answer"          # the Thinker's final reply for the caller
STILL_GOOD = "still_good"  # after new caller words: the notes already given still hold
FAILED = "failed"          # the Thinker could not finish: the job ends, offer a human
YES_ANSWER = "yes_answer"  # the Guard's reading of the caller's answer to a yes-question
ASKED = "asked"            # code asked the caller itself (decision 27's steps): already spoken
# Added 17 Nov 2026 for the two graphs (decisions 16, 27):
ORDER_NUMBER = "order_number"      # the Talker heard a full order number (its reader's digits)
CHECKED_NUMBER = "checked_number"  # the Thinker's AI checked that number; the only one used or said
PLAN = "plan"                      # the Thinker's plan, said instead of a waiting line
ASKS_USED_UP = "asks_used_up"      # the Talker asked again, digit by digit and by keypad: still not found
ANSWERED = "answered"              # the Thinker answered from the order (a lookup): the Talker asks "anything else?"

KINDS = (REQUEST, CALLER, FACT, NEED_INFO, NEED_YES, RESULT, ANSWER, STILL_GOOD, FAILED, YES_ANSWER, ASKED,
         ORDER_NUMBER, CHECKED_NUMBER, PLAN, ASKS_USED_UP, ANSWERED)
# What the caller should hear about, through the Talker.
SPEAKABLE = (FACT, NEED_INFO, RESULT, ANSWER, FAILED, CHECKED_NUMBER, PLAN)


class NotebookFull(RuntimeError):
    """More notes than any sane call needs: something is going round in circles."""


@dataclass(frozen=True)
class Note:
    number: int  # 1, 2, 3...: the order they were written in
    kind: str
    text: str
    version: int  # the job version the note was written for (decision 16)
    data: Mapping[str, Any] = field(default_factory=dict)


class Notebook:
    def __init__(self, max_notes: int) -> None:
        self._max = max_notes
        self._notes: list[Note] = []
        self._bell = asyncio.Event()

    @property
    def notes(self) -> tuple[Note, ...]:
        return tuple(self._notes)

    @property
    def newest(self) -> int:
        return len(self._notes)

    def add(self, kind: str, text: str, version: int, data: Mapping[str, Any] | None = None) -> Note:
        if kind not in KINDS:
            raise ValueError(f"unknown note kind {kind!r}")
        if len(self._notes) >= self._max:
            raise NotebookFull(f"over {self._max} notes in one call")
        note = Note(len(self._notes) + 1, kind, text, version, dict(data or {}))
        self._notes.append(note)
        self.ring()
        return note

    def ring(self) -> None:
        """Wake every waiter. A fresh event each time, so a waiter that comes late
        never sees an old ring."""
        bell, self._bell = self._bell, asyncio.Event()
        bell.set()

    def after(self, number: int) -> list[Note]:
        return self._notes[number:]

    def last(self, kinds: Iterable[str]) -> Note | None:
        wanted = set(kinds)
        return next((note for note in reversed(self._notes) if note.kind in wanted), None)

    async def wait(self, timeout_s: float) -> bool:
        """Wait for the next ring, at most timeout_s. True if it rang."""
        bell = self._bell
        try:
            await asyncio.wait_for(bell.wait(), timeout_s)
            return True
        except TimeoutError:
            return False
