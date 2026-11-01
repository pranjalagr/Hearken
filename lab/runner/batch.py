"""A batch of lab calls: several at once, lab problems rerun, and a stop
when the lab itself keeps breaking (Part 5, L3, L4, L14).

Each call goes in its own folder under the run; a run folder alone is enough
to re-mark its calls, replay any of them, and name the code and settings
that made it (L14).
"""

from __future__ import annotations

import asyncio
import gc
import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from engine.contract.pack import Pack
from engine.storage.call_factory import CallFactory
from engine.tasks.task_file import Task
from lab.bell_board import BellBoard
from lab.caller.director.plan import DirectorSettings
from lab.runner.run_call import LAB_PROBLEM, PROJECT_ROOT, CallOutcome, CallPlan, RobotSpec, run_call
from lab.scorer.score import SCORER_VERSION, CallScore, score_call
from lab.settings import LabSettings


@dataclass
class BatchResult:
    outcomes: list[CallOutcome] = field(default_factory=list)  # every attempt, reruns included
    scores: list[CallScore] = field(default_factory=list)  # one per call that ended (its last attempt)
    stopped: str = ""  # why the batch stopped early ("" if it didn't)
    seconds: float = 0.0

    @property
    def reruns(self) -> int:
        return sum(1 for o in self.outcomes if o.attempt > 1)

    @property
    def lab_problem_share(self) -> float:
        return sum(1 for o in self.outcomes if o.label == LAB_PROBLEM) / max(1, len(self.outcomes))


def plan_calls(tasks: list[Task], robots: list[RobotSpec], repeats: int, mode_for, seed: int = 1) -> list[CallPlan]:
    """Every robot on every task, `repeats` times, mixed in order (C13,
    decision 31: busy moments hit every robot alike). The same seed per
    task and repeat for every robot (C13: the same exam)."""
    plans = []
    # Repeat, then task, then robot: every robot takes each task back to back,
    # never one robot's whole set in a block.
    for repeat in range(repeats):
        for n, task in enumerate(tasks):
            call_seed = seed * 1000 + repeat * 100 + n
            short = "".join(task.id.split("_")[:2])  # food_01_... -> food01
            for robot in robots:
                call_id = f"{short}-{robot.name.replace(':', '-')}-r{repeat + 1}"
                plans.append(CallPlan(call_id, task, robot, mode_for(task, robot), call_seed))
    return plans


async def run_batch(plans: list[CallPlan], pack: Pack, run_dir: Path, settings: LabSettings) -> BatchResult:
    started = time.perf_counter()
    result = BatchResult()
    board = BellBoard()
    bells_url = await board.start()
    gate = asyncio.Semaphore(settings.calls_at_once)  # the traffic light (C17)
    broken = asyncio.Event()
    # One factory for the batch, its template built before any call starts.
    factory = CallFactory(pack, run_dir / "copies" / f"batch-{int(time.time() * 1000)}")
    await factory.build_template()
    ended: list[tuple[CallOutcome, CallPlan]] = []
    # The tests' freeze check found Python's garbage collector pausing for up to
    # 50 ms (batch 1); what exists now is set aside, so collections stay small
    # while 20 ms beats are running.
    gc.freeze()

    async def one(plan: CallPlan) -> None:
        async with gate:
            attempt = plan
            for number in range(1, settings.max_reruns + 2):
                if broken.is_set():
                    return
                try:
                    outcome = await run_call(attempt, pack, run_dir, board, bells_url, settings, factory)
                except Exception as exc:  # the lab's own code broke: never the robot's fault (L4)
                    outcome = CallOutcome(attempt.call_id, plan.task.id, plan.robot.name, plan.mode, plan.seed,
                                          attempt.attempt, LAB_PROBLEM, f"the lab crashed: {exc!r}")
                result.outcomes.append(outcome)
                if outcome.label != LAB_PROBLEM:
                    ended.append((outcome, plan))
                    return
                attempt = replace(plan, call_id=f"{plan.call_id}-again{number}", attempt=number + 1)
            # It kept breaking for lab reasons: the lab is broken, not the robot (L4).
            result.stopped = f"{plan.call_id} kept breaking: {outcome.why}"
            broken.set()

    try:
        await asyncio.gather(*(one(plan) for plan in plans))
    finally:
        await board.close()
        gc.unfreeze()
    # Marked only once no call is live: scoring reads audio and databases, and
    # even from another thread it competes with the 20 ms beats (Python's lock).
    for outcome, plan in ended:
        try:
            result.scores.append(await score_call(Path(outcome.folder), plan.task, pack, settings))
        except Exception as exc:  # the scorer broke: the lab's fault, not the robot's
            outcome.label, outcome.why = LAB_PROBLEM, f"the scorer crashed: {exc!r}"
            (Path(outcome.folder) / "outcome.json").write_text(json.dumps(asdict(outcome), indent=1, default=str),
                                                               encoding="utf-8")
            result.stopped = result.stopped or f"the scorer crashed on {outcome.call_id}: {exc!r}"
    result.seconds = time.perf_counter() - started
    return result


def make_run_dir(root: Path, label: str) -> Path:
    folder = root / f"{time.strftime('%Y-%m-%d_%H%M%S')}_{label}"
    folder.mkdir(parents=True, exist_ok=False)
    return folder


def write_run_json(run_dir: Path, plans: list[CallPlan], robots: list[RobotSpec], settings: LabSettings,
                   extra: dict[str, Any] | None = None) -> None:
    """What was run, with copies of every setting and prompt (L14), so changing
    one tomorrow can't change what this run meant."""
    about = {
        "robots": [robot.name for robot in robots],
        "tasks": sorted({plan.task.id for plan in plans}),
        "calls": len(plans),
        "code": _code_version(),
        "machine": {"os": platform.platform(), "python": platform.python_version(), "cpus": _cpus()},
        "models": {robot.name: getattr(robot, "models", {}) for robot in robots},
        "prompts": {robot.name: getattr(robot, "prompts", {}) for robot in robots},
        "lab_settings": settings.as_dict(),
        "director_settings": asdict(DirectorSettings()),
        "scorer_version": SCORER_VERSION,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        **(extra or {}),
    }
    (run_dir / "run.json").write_text(json.dumps(about, indent=1, ensure_ascii=False, default=str), encoding="utf-8")


def _code_version() -> dict[str, Any]:
    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    # "dirty" means the numbers came from code that isn't in any commit yet.
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def _cpus() -> int:
    import os

    return os.cpu_count() or 0
