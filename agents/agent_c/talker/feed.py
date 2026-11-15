"""The note feed: hands new notes to the Talker as "NOTES:" lines, at the right moment.

- Only notes for the job's current version are given (decision 16). A note written
  before the caller's newest words waits; the Thinker either says "still good"
  (then it is given) or redoes the work (then the newer notes replace it).
- Never while the caller is talking (decision 8): a note that comes then waits for
  the turn to end, so the Talker never speaks over the caller.
- It remembers which notes the Talker's last reply answered, so the job desk can
  tell when the final reply was fully spoken (close check 5).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from agents.agent_c.desk import JobDesk
from agents.agent_c.job import Turns
from agents.agent_c.notebook import FAILED, NEED_INFO, SPEAKABLE, STILL_GOOD, Note, Notebook


def notes_line(notes: list[Note]) -> str:
    parts = []
    for note in notes:
        if note.kind == NEED_INFO:
            parts.append(f"ask the caller: {note.text}")
        elif note.kind == FAILED:
            parts.append(f"failed: {note.text} Apologise and offer to connect a team member.")
        else:
            parts.append(note.text)
    return "NOTES: " + "; ".join(parts)


class NoteFeed:
    def __init__(self, notebook: Notebook, desk: JobDesk, turns: Turns, push: Callable[[str], Awaitable[None]],
                 wait_s: float) -> None:
        self._notebook = notebook
        self._desk = desk
        self._turns = turns
        self._push = push
        self._wait_s = wait_s  # how long one wait for a new note lasts before looking again
        self._read = 0
        self._held: list[Note] = []
        self._given = 0  # the newest note given to the Talker
        self._replied_after_giving = False

    async def run(self) -> None:
        while True:
            await self.give_new_notes()
            await self._notebook.wait(self._wait_s)

    async def give_new_notes(self) -> None:
        if self._turns.speaking:
            return  # the turn's end rings the notebook again
        job = self._desk.job
        version = job.version if job is not None else 0
        ready: list[Note] = []
        for note in self._notebook.after(self._read):
            if note.kind == STILL_GOOD:
                ready += [held for held in self._held if held.version < note.version]
                self._held = [held for held in self._held if held.version >= note.version]
            elif note.kind not in SPEAKABLE:
                continue
            elif note.kind == FAILED or note.version >= version:
                # Newer work replaces what was held from before the caller's words.
                self._held = [held for held in self._held if held.version >= note.version]
                ready.append(note)
            else:
                self._held.append(note)
        self._read = self._notebook.newest
        if ready:
            self._given = max(note.number for note in ready)
            self._replied_after_giving = False
            await self._push(notes_line(ready))

    # ---- what the speech watch reports ------------------------------------------

    def talker_replied(self) -> None:
        """The Talker finished writing a reply."""
        self._replied_after_giving = True

    def speech_ended(self, cut: bool) -> None:
        """The voice stopped: played to the end, or cut off by the caller."""
        if cut or not self._replied_after_giving or self._given == 0:
            return
        self._desk.reply_spoken(self._given)
