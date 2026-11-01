"""The numbers in the results table (Part 5, L12).

- Every number gets a 95% range, found by reshuffling TASKS (not calls): a
  hard task fails all its runs together, so its runs aren't separate evidence.
- "C beats A" only if the range of "C minus A", compared task by task on the
  same tasks, is clearly above zero; otherwise "no clear difference".
- Two kinds of success: the success rate, and "solved every time" (every
  counted run of a task passed).
- Zero harm is written honestly: 0 in n calls means at most about 3/n (the
  rule of three for a 95% upper bound).
- Lab-problem calls are never counted (L4).
"""

from __future__ import annotations

import random
import statistics
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from lab.scorer.score import CallScore


@dataclass(frozen=True)
class Ranged:
    value: float | None
    low: float | None
    high: float | None

    def show(self, unit: str = "", scale: float = 1.0, digits: int = 0) -> str:
        if self.value is None:
            return "n/a"
        f = lambda x: f"{x * scale:.{digits}f}{unit}"  # noqa: E731
        return f"{f(self.value)} ({f(self.low)}–{f(self.high)})"


@dataclass(frozen=True)
class RobotResult:
    robot: str
    calls: int
    counted: int
    tasks: int
    success: Ranged
    solved_every_time: Ranged
    harms: int
    harm_upper_per_100: float | None  # when harms == 0: at most this, with 95% confidence
    reply_delay_median_ms: Ranged
    reply_delay_p90_ms: Ranged
    tool_silence_median_ms: Ranged
    stops_on_ruko: str
    wrong_stops_on_haan: str
    cut_offs_per_100_turns: float
    made_up_calls: int
    judged: int


def summarize(scores: Sequence[CallScore], rounds: int, seed: int) -> dict[str, RobotResult]:
    by_robot: dict[str, list[CallScore]] = defaultdict(list)
    for score in scores:
        by_robot[score.robot].append(score)
    return {robot: _robot_result(robot, calls, rounds, seed) for robot, calls in sorted(by_robot.items())}


def _robot_result(robot: str, calls: list[CallScore], rounds: int, seed: int) -> RobotResult:
    counted = [c for c in calls if c.counted]
    by_task = _by_task(counted)
    tasks = sorted(by_task)
    rate = lambda picked: _mean([float(c.success) for t in picked for c in by_task[t]])  # noqa: E731
    every = lambda picked: _mean([float(all(c.success for c in by_task[t])) for t in picked])  # noqa: E731
    delays = lambda picked: [d for t in picked for c in by_task[t] for d in c.timing["reply_delays_ms"]]  # noqa: E731
    silences = lambda picked: [d for t in picked for c in by_task[t] for d in c.timing["tool_silences_ms"]]  # noqa: E731
    harms = sum(len(c.harms) for c in counted)
    stops = [s for c in counted for s in c.timing["stops"]]
    wrong = [s for c in counted for s in c.timing["wrong_stops"]]
    turns = sum(c.timing["turns"] for c in counted)
    return RobotResult(
        robot=robot,
        calls=len(calls),
        counted=len(counted),
        tasks=len(tasks),
        success=reshuffled(tasks, rate, rounds, seed),
        solved_every_time=reshuffled(tasks, every, rounds, seed),
        harms=harms,
        harm_upper_per_100=(300 / len(counted)) if counted and harms == 0 else None,
        reply_delay_median_ms=reshuffled(tasks, lambda p: _median(delays(p)), rounds, seed),
        reply_delay_p90_ms=reshuffled(tasks, lambda p: _p90(delays(p)), rounds, seed),
        tool_silence_median_ms=reshuffled(tasks, lambda p: _median(silences(p)), rounds, seed),
        stops_on_ruko=_share(stops),
        wrong_stops_on_haan=_share(wrong),
        cut_offs_per_100_turns=round(100 * sum(c.timing["cut_offs"] for c in counted) / turns, 1) if turns else 0.0,
        made_up_calls=sum(1 for c in counted if c.made_up),
        judged=sum(1 for c in counted if c.judge is not None),
    )


def reshuffled(tasks: list[str], stat: Callable[[list[str]], float | None], rounds: int, seed: int) -> Ranged:
    """The value on all tasks, and the middle 95% of its values when the
    tasks are drawn again at random (some twice, some not at all), `rounds` times."""
    if not tasks:
        return Ranged(None, None, None)
    rng = random.Random(seed)
    value = stat(tasks)
    draws = [stat([rng.choice(tasks) for _ in tasks]) for _ in range(rounds)]
    draws = sorted(d for d in draws if d is not None)
    if value is None or not draws:
        return Ranged(value, None, None)
    return Ranged(value, draws[int(0.025 * (len(draws) - 1))], draws[int(0.975 * (len(draws) - 1))])


@dataclass(frozen=True)
class Comparison:
    better: str
    worse: str
    difference: Ranged  # success of `better` minus `worse`, task by task
    verdict: str


def compare(scores: Sequence[CallScore], a: str, b: str, rounds: int, seed: int) -> Comparison:
    """Robot a against robot b on the tasks both took, task by task (L12)."""
    tasks_a, tasks_b = _by_task([s for s in scores if s.robot == a and s.counted]), _by_task(
        [s for s in scores if s.robot == b and s.counted])
    shared = sorted(set(tasks_a) & set(tasks_b))
    gap = {t: _mean([float(c.success) for c in tasks_a[t]]) - _mean([float(c.success) for c in tasks_b[t]])
           for t in shared}
    diff = reshuffled(shared, lambda picked: _mean([gap[t] for t in picked]), rounds, seed)
    if diff.low is not None and diff.low > 0:
        verdict = f"{a} is clearly better"
    elif diff.high is not None and diff.high < 0:
        verdict = f"{b} is clearly better"
    else:
        verdict = "no clear difference"
    return Comparison(a, b, diff, verdict)


def results_markdown(results: dict[str, RobotResult], comparisons: Sequence[Comparison] = (),
                     header: dict[str, Any] | None = None) -> str:
    lines = ["# Results", ""]
    for key, value in (header or {}).items():
        lines.append(f"- **{key}:** {value}")
    lines += ["", "| | " + " | ".join(results) + " |", "|---|" + "---|" * len(results)]
    rows = [
        ("Calls counted (of all)", lambda r: f"{r.counted} of {r.calls}, {r.tasks} tasks"),
        ("Task success (95% range)", lambda r: r.success.show("%", 100)),
        ("Solved every time", lambda r: r.solved_every_time.show("%", 100)),
        ("Harmful actions", lambda r: f"{r.harms}" + (f" (at most ~{r.harm_upper_per_100:.0f} per 100)"
                                                      if r.harm_upper_per_100 is not None else "")),
        ("Calls with made-up numbers", lambda r: f"{r.made_up_calls}"),
        ("Reply delay, median", lambda r: r.reply_delay_median_ms.show(" s", 0.001, 2)),
        ("Reply delay, p90", lambda r: r.reply_delay_p90_ms.show(" s", 0.001, 2)),
        ("Silence during tools, median", lambda r: r.tool_silence_median_ms.show(" s", 0.001, 2)),
        ("Stops on ruko", lambda r: r.stops_on_ruko),
        ("Wrong stops on haan", lambda r: r.wrong_stops_on_haan),
        ("Robot cut-offs per 100 turns", lambda r: f"{r.cut_offs_per_100_turns}"),
        ("Judged by the judge (L11)", lambda r: f"{r.judged} of {r.counted}"),
    ]
    for name, cell in rows:
        lines.append(f"| {name} | " + " | ".join(cell(r) for r in results.values()) + " |")
    for c in comparisons:
        lines += ["", f"**{c.better} vs {c.worse}:** success difference {c.difference.show(' points', 100)} "
                      f"→ {c.verdict}."]
    return "\n".join(lines) + "\n"


def _by_task(calls: Sequence[CallScore]) -> dict[str, list[CallScore]]:
    grouped: dict[str, list[CallScore]] = defaultdict(list)
    for call in calls:
        grouped[call.task_id].append(call)
    return grouped


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _p90(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1))))]


def _share(events: list[dict[str, Any]]) -> str:
    if not events:
        return "none happened"
    stopped = sum(1 for e in events if e["stopped"])
    return f"{stopped} of {len(events)} ({100 * stopped / len(events):.0f}%)"
