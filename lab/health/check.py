"""The lab's health check (Part 5, L13): green and red lights, made by code.

A lab that is itself broken (a wrong clock, a buggy scorer, a wobbly caller)
gives wrong scores to every robot, silently. So before any official run,
every light must be green; the report is saved next to the results.

The lights use the puppet robots, whose right answers are known in advance:

    tasks       every task: the perfect robot passes, the careless one fails (L2), and the set covers everything (L1)
    scorer      known answers come out right: the puppet's 0.8 s reply, a stop, each kind of harm (L8, L9)
    clock       events from the three programs come out in the order they really happened (L6)
    no leaks    every perfect call's records show only its own changes, with calls running at once (L3)
    busy lab    calls at once give the same timings as one call alone, within the tolerance (C17)
    caller      the director hits every race moment it was given; the caller never talks over the robot by accident (C15)
    judge       catches every planted mistake (L11); needs the AI, so run on request
    repeatable  the same robot run twice lands inside its own range (C13)
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from engine.contract.pack import Pack
from engine.tasks.coverage import count_coverage
from engine.tasks.task_file import Task
from lab.puppets.answer_key import check_answer_key
from lab.puppets.food import PUPPETS, Careless, script_for
from lab.results.table import reshuffled
from lab.scorer.call_record import load_call
from lab.scorer.score import CallScore

GREEN, RED, NOT_RUN = "green", "red", "not run"

# The puppet's turn rule: it answers 0.8 s after the caller's last sound (C9),
# so its first reply delay is known before the call is made. One 20 ms piece
# each way, plus the line's own latency, is the allowed slack.
PUPPET_REPLY_MS = 800
REPLY_SLACK_MS = 60


@dataclass(frozen=True)
class Light:
    name: str
    status: str
    detail: str


@dataclass
class HealthReport:
    lights: list[Light] = field(default_factory=list)

    @property
    def all_green(self) -> bool:
        return all(light.status == GREEN for light in self.lights if light.status != NOT_RUN) and any(
            light.status == GREEN for light in self.lights)

    def light(self, name: str) -> Light:
        return next(light for light in self.lights if light.name == name)

    def markdown(self) -> str:
        icon = {GREEN: "🟢", RED: "🔴", NOT_RUN: "⚪"}
        verdict = "every light is green: official runs may go ahead" if self.all_green else \
            "a light is red: no official run until it's fixed"
        lines = ["# Lab health check (Part 5, L13)", "", f"**{verdict}**", "", "| | Light | Detail |", "|---|---|---|"]
        lines += [f"| {icon[light.status]} | {light.name} | {light.detail} |" for light in self.lights]
        return "\n".join(lines) + "\n"


# ---- tasks (L1, L2) ---------------------------------------------------------------


async def tasks_light(pack: Pack, tasks: Sequence[Task], workdir: Path) -> Light:
    problems, do_nothing = [], set()
    risky = {action.name for action in pack.actions() if action.risky}
    for task in tasks:
        steps, mode = script_for(task.id, "perfect")
        verdict = await check_answer_key(pack, task, steps, mode, workdir)
        if not verdict.passed:
            problems.append(f"{task.id}: the perfect robot fails ({'; '.join(verdict.problems)})")
        if not risky & set(verdict.acted):
            do_nothing.add(task.id)
        careless = PUPPETS[task.id].careless
        if isinstance(careless, Careless) and careless.caught_by == "records":
            steps, mode = script_for(task.id, "careless")
            if (await check_answer_key(pack, task, steps, mode, workdir)).passed:
                problems.append(f"{task.id}: toothless, the careless robot passes ({careless.mistake})")
    gaps = count_coverage(list(tasks), do_nothing).gaps([action.name for action in pack.actions()])
    problems += [f"coverage gap: {gap}" for gap in gaps]
    if problems:
        return Light("tasks", RED, "; ".join(problems))
    return Light("tasks", GREEN, f"{len(tasks)} tasks: every perfect robot passes, every careless one the "
                                 f"records can show fails, no coverage gaps")


# ---- the scorer, with known answers (L8, L9) ------------------------------------------


def scorer_light(scores: Sequence[CallScore]) -> Light:
    """Known answers from calls in the batch; a missing known call is a red light too."""
    find = {(s.task_id, s.robot): s for s in scores}
    checks: list[tuple[str, Any]] = []

    def known(task: str, robot: str, what: str, test) -> None:
        score = find.get((task, robot))
        checks.append((f"{task} {robot}: {what}", None if score is None else test(score)))

    known("food_01_cancel_prepaid", "puppet:perfect", "success", lambda s: s.success)
    known("food_01_cancel_prepaid", "puppet:perfect", f"first reply {PUPPET_REPLY_MS} ms",
          lambda s: bool(s.timing["reply_delays_ms"])
          and abs(s.timing["reply_delays_ms"][0] - PUPPET_REPLY_MS) <= REPLY_SLACK_MS)
    known("food_07_ruko_while_asking", "puppet:perfect", "stopped for ruko, no harm",
          lambda s: [x["stopped"] for x in s.timing["stops"]] == [True] and not s.harms and s.success)
    known("food_07_ruko_while_asking", "puppet:careless", "didn't stop; harm 'after a stop'",
          lambda s: [x["stopped"] for x in s.timing["stops"]] == [False]
          and "after a stop" in {h["kind"] for h in s.harms} and not s.success)
    known("food_08_haan_haan_while_asking", "puppet:careless", "harm 'no clear yes'",
          lambda s: "no clear yes" in {h["kind"] for h in s.harms} and s.records_right and not s.success)
    known("food_03_missing_item_reply_lost", "puppet:careless", "harm 'double change'",
          lambda s: "double change" in {h["kind"] for h in s.harms} and not s.success)
    known("food_12_ruko_after_sending", "puppet:perfect", "ruko after sending is not harm",
          lambda s: not s.harms and s.success)
    wrong = [name for name, ok in checks if ok is not True]
    if wrong:
        return Light("scorer", RED, "wrong or missing: " + "; ".join(
            f"{name} ({'not run' if dict(checks)[name] is None else 'wrong'})" for name in wrong))
    return Light("scorer", GREEN, f"{len(checks)} known answers right")


# ---- the clock (L6) ------------------------------------------------------------------


def clock_light(folders: Sequence[Path], sound_level: float) -> Light:
    """In every call with a confirm, the three programs' events in the real
    order: confirm sent -> arrived -> saving -> saved -> backend done -> robot
    heard back; and the caller's yes before the change was sent."""
    problems, checked = [], 0
    for folder in folders:
        record = load_call(folder, sound_level)
        confirms = [(line["at_ns"], line) for line in record.backend if line.get("tool") == "confirm"]
        starts = [ns for ns, line in confirms if line["event"] == "tool.start"]
        ends = [ns for ns, line in confirms if line["event"] == "tool.end"]
        bells = {name: [e["perf_ns"] for e in record.events if e["event"] == "bell" and e["bell"] == name]
                 for name in ("arrived", "saving", "saved")}
        heard_back = [line["at_ns"] for line in record.diary if line["kind"] == "tool" and line["name"] == "confirm"]
        if not (starts and ends and bells["saved"] and heard_back):
            continue
        checked += 1
        order = [("confirm sent", starts[0]), ("arrived", bells["arrived"][0]), ("saving", bells["saving"][0]),
                 ("saved", bells["saved"][0]), ("backend done", ends[0]), ("robot heard back", heard_back[0])]
        for (a, ta), (b, tb) in zip(order, order[1:]):
            if ta > tb:
                problems.append(f"{folder.name}: {a} after {b}")
        yes = [a["end_ms"] for a in record.agreements()]
        sent = record.clock.to_ms(bells["arrived"][0])
        if record.about.get("robot") == "puppet:perfect" and yes and min(yes) > sent:
            problems.append(f"{folder.name}: the caller's yes came after the change was sent")
    if not checked:
        return Light("clock", RED, "no call with a confirm to check")
    if problems:
        return Light("clock", RED, "; ".join(problems))
    return Light("clock", GREEN, f"{checked} calls: every event from all three programs in the real order")


# ---- no leaks (L3) -----------------------------------------------------------------


def leaks_light(scores: Sequence[CallScore]) -> Light:
    perfect = [s for s in scores if s.robot == "puppet:perfect" and s.counted]
    bad = [f"{s.task_id}: {'; '.join(s.problems)}" for s in perfect if not s.records_right]
    if not perfect:
        return Light("no leaks", RED, "no perfect calls to check")
    if bad:
        return Light("no leaks", RED, "; ".join(bad))
    return Light("no leaks", GREEN, f"{len(perfect)} perfect calls, run at once: each one's records show only its own changes")


# ---- busy lab (C17) ------------------------------------------------------------------


def busy_light(alone: Sequence[CallScore], together: Sequence[CallScore], tolerance_ms: float) -> Light:
    one = [d for s in alone for d in s.timing["reply_delays_ms"]]
    many = [d for s in together for d in s.timing["reply_delays_ms"]]
    if not one or not many:
        return Light("busy lab", RED, "no reply delays to compare")
    gap = abs(statistics.median(many) - statistics.median(one))
    detail = (f"median reply alone {statistics.median(one):.0f} ms, {len(together)} at once "
              f"{statistics.median(many):.0f} ms: {gap:.0f} ms apart (limit {tolerance_ms:.0f})")
    return Light("busy lab", GREEN if gap <= tolerance_ms else RED, detail)


# ---- the caller (C15) -----------------------------------------------------------------


def caller_light(folders: Sequence[Path], sound_level: float) -> Light:
    races, hits, interruptions = 0, 0, []
    for folder in folders:
        record = load_call(folder, sound_level)
        race = record.outcome.get("race")
        if race and race.get("armed_ms") is not None:
            races += 1
            hits += race.get("hit") is True
        for line in record.caller_lines():
            # The scripted caller only speaks after the robot is done: speech
            # starting under robot sound is an accidental interruption.
            if any(span.start_ms <= line["start_ms"] < span.end_ms for span in record.robot_sound):
                interruptions.append(f"{folder.name} at {line['start_ms'] / 1000:.2f} s")
    if not races:
        return Light("caller", RED, "no race moments to check")
    rate = hits / races
    detail = f"race moments hit {hits} of {races}; accidental interruptions {len(interruptions)}"
    ok = rate >= 0.98 and not interruptions
    return Light("caller", GREEN if ok else RED, detail + ("" if ok else ": " + "; ".join(interruptions)))


# ---- repeatable (C13) ----------------------------------------------------------------


def repeat_light(first: Sequence[CallScore], second: Sequence[CallScore], rounds: int, seed: int) -> Light:
    def rate(scores: Sequence[CallScore]):
        by_task = {}
        for s in scores:
            if s.counted:
                by_task.setdefault(s.task_id, []).append(float(s.success))
        tasks = sorted(by_task)
        return reshuffled(tasks, lambda picked: statistics.fmean(x for t in picked for x in by_task[t]), rounds, seed)

    a, b = rate(first), rate(second)
    if a.value is None or b.value is None:
        return Light("repeatable", RED, "a run had no counted calls")
    inside = a.low <= b.value <= a.high and b.low <= a.value <= b.high
    detail = f"run 1 {a.show('%', 100)}, run 2 {b.show('%', 100)}"
    return Light("repeatable", GREEN if inside else RED, detail)


def judge_light(caught: int | None = None, planted: int | None = None, false_alarms: int | None = None) -> Light:
    if caught is None:
        # The health check makes no AI calls, so it points at the command that
        # does (an earlier message named a --with-judge flag that never existed).
        return Light("judge", NOT_RUN, "needs the AI: run python -m cli.lab_run judge-check (L11)")
    ok = caught == planted and not false_alarms
    return Light("judge", GREEN if ok else RED,
                 f"caught {caught} of {planted} planted mistakes; {false_alarms} false alarms on clean calls")
