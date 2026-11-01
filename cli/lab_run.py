"""The test lab (Part 5): run calls, check the lab's health, re-mark old runs.

    python -m cli.lab_run run --robots puppet:perfect,puppet:careless [--repeats 1] [--at-once 5] [--label x]
    python -m cli.lab_run health [--quick]
    python -m cli.lab_run rescore runs/lab/<run folder>
    python -m cli.lab_run judge-check            (the judge on planted mistakes, twice; needs GEMINI_API_KEY)
    python -m cli.lab_run judge runs/lab/<run folder>   (the judge marks a run's calls)

Every run gets its own folder under runs/lab/ (not in git): run.json (what
was run, with copies of every setting), calls/<call>/ (the black box and its
score.json), results.md, and health.md for a health check.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from common.retry import http_status
from engine.tasks.task_file import load_task, load_tasks
from lab.health.run_health import run_health
from lab.puppets.food import PUPPETS, Careless, script_for
from lab.results.table import compare, results_markdown, summarize
from lab.runner.batch import make_run_dir, plan_calls, run_batch, write_run_json
from lab.runner.run_call import PuppetRobot
from lab.scorer.score import score_call
from lab.settings import load_settings
from packs.food.pack import FoodPack

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "lab"
PRACTICE = ROOT / "packs" / "food" / "tasks" / "practice"
ROBOTS = {"puppet:perfect": PuppetRobot("perfect"), "puppet:careless": PuppetRobot("careless")}


async def run(args: argparse.Namespace) -> Path:
    settings = load_settings()
    if args.at_once:
        settings = settings.with_changes(calls_at_once=args.at_once)
    robots = [ROBOTS[name] for name in args.robots.split(",")]
    tasks = load_tasks(PRACTICE)

    def mode_for(task, robot):
        return script_for(task.id, robot.variant)[1]

    plans = [p for p in plan_calls(tasks, robots, args.repeats, mode_for)
             if p.robot.variant == "perfect" or isinstance(PUPPETS[p.task.id].careless, Careless)]
    run_dir = make_run_dir(RUNS, args.label)
    write_run_json(run_dir, plans, robots, settings, {"kind": "run", "repeats": args.repeats})
    print(f"{len(plans)} calls, {settings.calls_at_once} at once -> {run_dir}")
    batch = await run_batch(plans, FoodPack(), run_dir, settings)
    _write_results(run_dir, batch.scores, settings, {"Calls": len(batch.outcomes), "Reruns": batch.reruns,
                                                     "Stopped early": batch.stopped or "no",
                                                     "Seconds": f"{batch.seconds:.0f}"})
    return run_dir


def _write_results(run_dir: Path, scores, settings, header) -> None:
    results = summarize(scores, settings.reshuffle_rounds, settings.reshuffle_seed)
    names = list(results)
    comparisons = [compare(scores, names[0], names[1], settings.reshuffle_rounds, settings.reshuffle_seed)] \
        if len(names) == 2 else []
    (run_dir / "results.md").write_text(results_markdown(results, comparisons, {"Run": run_dir.name, **header}),
                                        encoding="utf-8")


async def rescore(run_dir: Path) -> None:
    """Re-mark every call of an old run from its folder alone (L9, L14)."""
    settings = load_settings()
    scores = []
    for folder in sorted((run_dir / "calls").iterdir()):
        about = json.loads((folder / "caller.json").read_text(encoding="utf-8"))
        task = load_task(PRACTICE / f"{about['task_id']}.yaml")
        if json.loads((folder / "outcome.json").read_text(encoding="utf-8")).get("label") == "lab_problem":
            continue
        scores.append(await score_call(folder, task, FoodPack(), settings))
    _write_results(run_dir, scores, settings, {"Re-marked calls": len(scores)})


def _gemini(timeout_s: float):
    from common.env import load_env, require
    from common.google import google_client

    load_env()
    return google_client(require("GEMINI_API_KEY"), timeout_s)


async def judge_check() -> tuple[dict, Path]:
    """L11: the judge on the planted calls, run twice for steadiness."""
    from lab.health.check import judge_light
    from lab.judge.check import changed_share, judge_report, run_planted
    from lab.judge.judge import gemini_judge

    client = _gemini(load_settings().judge_timeout_s)
    started = time.perf_counter()

    def show(run: int):
        def on_case(number: int, total: int, case_id: str) -> None:
            print(f"run {run}, case {number}/{total}: {case_id} ({time.perf_counter() - started:.0f} s in)", flush=True)
        return on_case

    def retrying(error: BaseException, wait: float) -> None:
        said = http_status(error) or type(error).__name__
        print(f"  Google said {said}; asking again in {wait:.0f} s", flush=True)

    try:
        ask = gemini_judge(client, on_retry=retrying)
        graded, first = await run_planted(ask, on_case=show(1))
        _again, second = await run_planted(ask, on_case=show(2))
    finally:
        await client.aio.aclose()
    report = judge_report(graded, changed_share(first, second))
    folder = make_run_dir(RUNS, "judge-check")
    light = judge_light(graded.caught, graded.planted, len(graded.false_alarms))
    report["light"] = {"status": light.status, "detail": light.detail}
    (folder / "judge.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    return report, folder


async def judge_a_run(run_dir: Path) -> None:
    from lab.judge.apply import judge_run
    from lab.judge.judge import gemini_judge

    settings = load_settings()
    client = _gemini(settings.judge_timeout_s)
    try:
        scores = await judge_run(run_dir, gemini_judge(client), settings)
    finally:
        await client.aio.aclose()
    _write_results(run_dir, scores, load_settings(), {"Judged calls": len(scores)})


def main() -> None:
    parser = argparse.ArgumentParser(description="The test lab.")
    commands = parser.add_subparsers(dest="command", required=True)
    run_cmd = commands.add_parser("run", help="run calls")
    run_cmd.add_argument("--robots", default="puppet:perfect")
    run_cmd.add_argument("--repeats", type=int, default=1)
    run_cmd.add_argument("--at-once", type=int, default=0)
    run_cmd.add_argument("--label", default="run")
    health = commands.add_parser("health", help="check the lab itself")
    health.add_argument("--quick", action="store_true")
    again = commands.add_parser("rescore", help="re-mark an old run")
    again.add_argument("folder")
    commands.add_parser("judge-check", help="check the judge on planted mistakes")
    judged = commands.add_parser("judge", help="the judge marks a run's calls")
    judged.add_argument("folder")
    args = parser.parse_args()
    for name in getattr(args, "robots", "").split(","):
        if name and name not in ROBOTS:
            parser.error(f"unknown robot {name!r}; choose from {sorted(ROBOTS)}")
    if args.command == "run":
        folder = asyncio.run(run(args))
        print(f"results: {folder / 'results.md'}")
    elif args.command == "health":
        outcome = asyncio.run(run_health(FoodPack(), load_tasks(PRACTICE), load_settings(), RUNS, args.quick))
        print(outcome.report.markdown())
        print(f"report: {outcome.run_dir / 'health.md'}")
        sys.exit(0 if outcome.report.all_green else 1)
    elif args.command == "judge-check":
        report, folder = asyncio.run(judge_check())
        print(json.dumps(report, indent=1, ensure_ascii=False))
        print(f"report: {folder / 'judge.json'}")
        sys.exit(0 if report["light"]["status"] == "green" else 1)
    elif args.command == "judge":
        asyncio.run(judge_a_run(Path(args.folder)))
        print(f"results: {Path(args.folder) / 'results.md'}")
    else:
        asyncio.run(rescore(Path(args.folder)))
        print(f"results: {Path(args.folder) / 'results.md'}")


if __name__ == "__main__":
    main()
