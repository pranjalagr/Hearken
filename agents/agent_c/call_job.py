"""The open job's version and its changes in flight, shared by the two graphs and the Guard
(decision 16, 17 Nov 2026).

The old job desk (desk.py) also decided when a job ends; in the two graphs the Thinker
graph does that itself. What is left is small:
- the version: every caller turn during a job makes the notes written before it old news,
  so the Talker never says something the caller has since changed (decision 16);
- the count of changes sent but not yet answered, which the Guard keeps (decision 7).
"""

from __future__ import annotations


class CallJob:
    """One call's job: its version number and how many changes are still being saved."""

    def __init__(self) -> None:
        self.version = 1
        self.changes_running = 0

    @property
    def job(self) -> CallJob:
        """The Guard reads the job through `desk.job`, as it did with the old desk."""
        return self

    def new_turn(self) -> None:
        """A caller turn finished during a job: notes written before it are now old."""
        self.version += 1

    def change_sent(self) -> None:
        """The Guard sent a confirm (a change is being saved)."""
        self.changes_running += 1

    def change_answered(self) -> None:
        """The confirm's answer came (or it was given up)."""
        self.changes_running = max(0, self.changes_running - 1)
