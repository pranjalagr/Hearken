"""How well a set of tasks covers what the lab must test (Part 5, L1).

Counted from the task files themselves (jobs, race moments, tricks, caller
language, breakdown switches), plus which tasks have "do nothing" as the
right ending, which only running a perfect robot can show. A cell below its
minimum is a gap: the set would let a robot pass while failing at that thing.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from engine.tasks.caller_spec import RACE_MOMENTS, TRICKS
from engine.tasks.task_file import Task


@dataclass(frozen=True)
class Minimums:
    """L1's starting minimums for the 15 practice tasks."""

    per_job: int = 3
    per_race_moment: int = 2
    per_trick: int = 1
    english_callers: int = 2
    do_nothing: int = 3
    per_breakdown: int = 1


@dataclass
class Coverage:
    jobs: Counter = field(default_factory=Counter)
    race_moments: Counter = field(default_factory=Counter)
    tricks: Counter = field(default_factory=Counter)
    languages: Counter = field(default_factory=Counter)
    breakdowns: Counter = field(default_factory=Counter)
    do_nothing: list[str] = field(default_factory=list)

    def gaps(self, actions: list[str], m: Minimums = Minimums()) -> list[str]:
        found = []
        for name, count, minimum, wanted in (
            ("job", self.jobs, m.per_job, actions),
            ("race moment", self.race_moments, m.per_race_moment, RACE_MOMENTS),
            ("trick", self.tricks, m.per_trick, TRICKS),
            ("breakdown", self.breakdowns, m.per_breakdown, ("reply_lost", "slow")),
        ):
            found += [f"{name} {key}: {count[key]} < {minimum}" for key in wanted if count[key] < minimum]
        if self.languages["mostly_english"] < m.english_callers:
            found.append(f"English callers: {self.languages['mostly_english']} < {m.english_callers}")
        if len(self.do_nothing) < m.do_nothing:
            found.append(f"do-nothing endings: {len(self.do_nothing)} < {m.do_nothing}")
        return found

    def grid(self) -> str:
        """A readable table, printed when the check fails."""
        rows = [("jobs", self.jobs), ("race moments", self.race_moments), ("tricks", self.tricks),
                ("languages", self.languages), ("breakdowns", self.breakdowns)]
        lines = [f"{name}: " + ", ".join(f"{key} {n}" for key, n in sorted(counter.items())) for name, counter in rows]
        lines.append(f"do nothing: {', '.join(self.do_nothing)}")
        return "\n".join(lines)


def count_coverage(tasks: list[Task], do_nothing_ids: set[str]) -> Coverage:
    coverage = Coverage()
    for task in tasks:
        coverage.jobs.update(task.jobs)
        if task.race is not None:
            coverage.race_moments[task.race.moment] += 1
        if task.caller is not None:
            coverage.tricks.update(task.caller.tricks)
            coverage.languages[task.caller.language] += 1
        coverage.breakdowns.update({switch.kind for switch in task.switches})
        if task.id in do_nothing_ids:
            coverage.do_nothing.append(task.id)
    return coverage
