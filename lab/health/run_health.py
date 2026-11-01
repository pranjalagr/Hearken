"""Run the lab's health check (Part 5, L13) and save its report.

quick: before every batch, ~1 minute, puppets only (no AI cost): the tasks'
       answer keys, and the calls the scorer's known answers need, which
       also check the clock, leaks and the caller.
full:  before any official run: every task's perfect and careless robot,
       the busy-lab comparison, and a second run for "repeatable".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from engine.contract.pack import Pack
from engine.tasks.task_file import Task
from lab.health.check import (
    HealthReport,
    busy_light,
    caller_light,
    clock_light,
    judge_light,
    leaks_light,
    repeat_light,
    scorer_light,
    tasks_light,
)
from lab.puppets.food import PUPPETS, Careless, script_for
from lab.results.table import results_markdown, summarize
from lab.runner.batch import make_run_dir, run_batch, write_run_json
from lab.runner.run_call import LAB_PROBLEM, CallPlan, PuppetRobot
from lab.settings import LabSettings

PERFECT, CARELESS = PuppetRobot("perfect"), PuppetRobot("careless")
# The calls the scorer's known answers need (see check.scorer_light).
QUICK = [
    ("food_01_cancel_prepaid", PERFECT), ("food_07_ruko_while_asking", PERFECT),
    ("food_07_ruko_while_asking", CARELESS), ("food_08_haan_haan_while_asking", CARELESS),
    ("food_03_missing_item_reply_lost", CARELESS), ("food_12_ruko_after_sending", PERFECT),
]


@dataclass
class HealthRun:
    report: HealthReport
    run_dir: Path


def puppet_plan(task: Task, robot: PuppetRobot, prefix: str = "") -> CallPlan:
    _steps, mode = script_for(task.id, robot.variant)
    short = "".join(task.id.split("_")[:2])
    return CallPlan(f"{prefix}{short}-{robot.variant}", task, robot, mode, seed=int(task.id.split("_")[1]))


async def run_health(pack: Pack, tasks: list[Task], settings: LabSettings, root: Path, quick: bool) -> HealthRun:
    run_dir = make_run_dir(root, "health-quick" if quick else "health")
    by_id = {task.id: task for task in tasks}
    report = HealthReport()
    report.lights.append(await tasks_light(pack, tasks, run_dir / "answer_keys"))
    if quick:
        plans = [puppet_plan(by_id[task_id], robot) for task_id, robot in QUICK]
    else:
        plans = [puppet_plan(task, PERFECT) for task in tasks]
        plans += [puppet_plan(task, CARELESS) for task in tasks if isinstance(PUPPETS[task.id].careless, Careless)]
    write_run_json(run_dir, plans, [PERFECT, CARELESS], settings, {"kind": "health check", "quick": quick})
    batch = await run_batch(plans, pack, run_dir, settings)
    folders = [Path(o.folder) for o in batch.outcomes if o.label != LAB_PROBLEM]
    report.lights += [
        scorer_light(batch.scores),
        clock_light(folders, settings.sound_level),
        leaks_light(batch.scores),
        caller_light(folders, settings.sound_level),
    ]
    if not quick:
        first = by_id["food_01_cancel_prepaid"]
        alone = await run_batch([puppet_plan(first, PERFECT, "alone-")], pack, run_dir, settings.with_changes(calls_at_once=1))
        together = await run_batch(
            [puppet_plan(first, PERFECT, f"busy{n}-") for n in range(settings.calls_at_once)], pack, run_dir, settings)
        report.lights.append(busy_light(alone.scores, together.scores, settings.busy_lab_tolerance_ms))
        second = await run_batch([puppet_plan(task, PERFECT, "again-") for task in tasks], pack, run_dir, settings)
        report.lights.append(repeat_light([s for s in batch.scores if s.robot == PERFECT.name], second.scores,
                                          settings.reshuffle_rounds, settings.reshuffle_seed))
    report.lights.append(judge_light())
    if batch.stopped:
        report.lights.append(_stopped_light(batch.stopped))
    (run_dir / "health.md").write_text(report.markdown(), encoding="utf-8")
    results = summarize(batch.scores, settings.reshuffle_rounds, settings.reshuffle_seed)
    (run_dir / "results.md").write_text(
        results_markdown(results, header={"Run": run_dir.name, "Calls": len(batch.outcomes),
                                          "Reruns": batch.reruns, "Seconds": f"{batch.seconds:.0f}"}),
        encoding="utf-8")
    return HealthRun(report, run_dir)


def _stopped_light(why: str):
    from lab.health.check import RED, Light

    return Light("batch", RED, f"the batch stopped: {why}")
