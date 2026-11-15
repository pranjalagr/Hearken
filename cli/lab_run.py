"""The test lab (Part 5): run calls, check the lab's health, re-mark old runs.

    python -m cli.lab_run run --robots puppet:perfect,puppet:careless [--repeats 1] [--at-once 5] [--label x]
    python -m cli.lab_run health [--quick]
    python -m cli.lab_run rescore runs/lab/<run folder>
    python -m cli.lab_run judge-check [--judge gemini]   (the judge on planted mistakes, twice)
    python -m cli.lab_run judge runs/lab/<run folder> [--judge gemini]   (the judge marks a run's calls)

The judge needs GROQ_API_KEY (groq, the default) or GEMINI_API_KEY (gemini) in .env.

Every run gets its own folder under runs/lab/ (not in git): run.json (what
was run, with copies of every setting), calls/<call>/ (the black box and its
score.json), results.md, and health.md for a health check.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from pathlib import Path

from common.env import groq_key, load_env, require
from common.freeze_watch import FreezeWatch, Stall
from common.groq import groq_client
from common.retry import http_status
from engine.storage.call_settings import PLAIN, SAFE
from engine.tasks.task_file import load_task, load_tasks
from lab.caller.parts import RealCaller, missing_keys
from lab.caller.report_card import AI_FORM, report_on_run
from lab.caller.settings import load_caller_settings
from lab.health.run_health import run_health
from lab.judge.judge import Ask, groq_judge, paced
from lab.puppets.food import PUPPETS, Careless, script_for
from lab.results.table import compare, results_markdown, summarize
from lab.runner.batch import make_run_dir, plan_calls, run_batch, write_run_json
from lab.runner.run_call import AgentRobot, PuppetRobot
from lab.scorer.score import score_call
from lab.settings import load_settings
from packs.food.pack import FoodPack

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "lab"
PRACTICE = ROOT / "packs" / "food" / "tasks" / "practice"
# The caller's report card reads each call once, after the call, so speed doesn't
# matter. Each Groq model has its own daily tokens: the caller's model ran out on
# 4 Nov 2026, and this one takes nothing from it.
REPORT_MODEL = "openai/gpt-oss-120b"
AGENTS = Path(__file__).resolve().parents[1] / "agents"


def _agent_a(name: str, mode: str, background: bool = False) -> AgentRobot:
    import yaml

    # Read straight from its settings file, so run.json keeps the models without
    # importing Pipecat into every lab command.
    data = yaml.safe_load((AGENTS / "agent_a" / "settings.yaml").read_text(encoding="utf-8"))
    models = {"brain": data["brain"]["model"], "ears": data["ears"]["model"],
              "voice": f"{data['voice']['model']}/{data['voice']['speaker']}", "background_tools": str(background)}
    return AgentRobot(name, "cli.agent_a", mode, ticket_check=False,
                      extra_args=("--background-tools",) if background else (), models=models)


ROBOTS = {
    "puppet:perfect": PuppetRobot("perfect"),
    "puppet:careless": PuppetRobot("careless"),
    # Decision 3: in the results table A runs in safe mode with the ticket check off.
    "agent:a": _agent_a("agent:a", SAFE),
    # Part 7's extra run: Pipecat's one-line background-tool switch, in plain mode.
    "agent:a-background": _agent_a("agent:a-background", PLAIN, background=True),
}


def write_stalls(path: Path, stalls: list[Stall]) -> None:
    """The lab's own stalls over 100 ms: "ours" names the code that ran; "the OS" means
    the loop was only waiting, so the laptop was busy or paging memory."""
    lines = ["# Event-loop stalls over 100 ms (decision 14)", "", f"{len(stalls)} stalls.", ""]
    for stall in stalls:
        cause = "ours" if stall.ours else "the OS (busy or paging laptop)"
        lines.append(f"- {stall.wall} (+{stall.at_s:.1f} s): {stall.late_ms:.0f} ms late, cause: {cause}")
        lines += [f"  {frame}" for frame in stall.where]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def run(args: argparse.Namespace) -> Path:
    settings = load_settings()
    if args.at_once:
        settings = settings.with_changes(calls_at_once=args.at_once)
    robots = [ROBOTS[name] for name in args.robots.split(",")]
    agents = [robot for robot in robots if isinstance(robot, AgentRobot)]
    if agents and args.caller != "chain":
        # An AI robot hears real speech only: the scripted caller just beeps.
        raise SystemExit("AI robots need the real caller: add --caller chain")
    if agents and not args.at_once:
        settings = settings.with_changes(calls_at_once=settings.ai_calls_at_once)
    tasks = load_tasks(PRACTICE)
    if args.tasks:
        wanted = args.tasks.split(",")
        tasks = [task for task in tasks if any(task.id.startswith(prefix) for prefix in wanted)]
    caller_info: dict = {}
    if args.caller == "chain":
        # Real ears, brain and voice (Part 3): the puppet needs real speech for them to hear.
        caller_settings = load_caller_settings()
        if args.brain:
            # C3's comparison: "gemini:gemini-3.5-flash-lite", for one run.
            company, model = args.brain.split(":", 1)
            caller_settings = replace(caller_settings, brain_company=company, brain_model=model)
        missing = missing_keys(caller_settings)
        if missing:
            raise SystemExit(f"missing in pakka/.env: {', '.join(missing)}")
        if agents:
            from lab.runner.key_check import refused_keys

            # One tiny request per key before any call: a refused or full key must not cost
            # a whole batch (15 Nov 2026). The caller's brain is checked too.
            groq_brain = caller_settings.brain_model if caller_settings.brain_company == "groq" else None
            refused = refused_keys(agents[0].models.get("brain", ""), caller_model=groq_brain)
            if refused:
                raise SystemExit("a service refused a key, so no calls were started:\n  " + "\n  ".join(refused))
        robots = [replace(robot, voice="sarvam") if isinstance(robot, PuppetRobot) else robot for robot in robots]
        caller_info = {"caller": "chain", "caller_settings": caller_settings.as_dict()}

    def mode_for(task, robot):
        if isinstance(robot, AgentRobot):
            return robot.mode
        return script_for(task.id, robot.variant)[1]

    def wanted(plan) -> bool:
        if isinstance(plan.robot, AgentRobot):
            # Some races exist only in safe mode (a task's `modes`), so a plain-mode run skips them.
            return plan.robot.mode in plan.task.modes
        return plan.robot.variant == "perfect" or isinstance(PUPPETS[plan.task.id].careless, Careless)

    plans = [replace(p, ticket_check=p.robot.ticket_check) if isinstance(p.robot, AgentRobot) else p
             for p in plan_calls(tasks, robots, args.repeats, mode_for) if wanted(p)]
    if args.caller == "chain":
        # Puppets get a clean line; AI robots face each task's background noise (C5).
        maker = RealCaller(caller_settings, use_noise=bool(agents))
        await maker.warm_up()
        plans = [replace(plan, caller=maker) for plan in plans]
    run_dir = make_run_dir(RUNS, args.label)
    write_run_json(run_dir, plans, robots, settings, {"kind": "run", "repeats": args.repeats, **caller_info})
    print(f"{len(plans)} calls, {settings.calls_at_once} at once -> {run_dir}")
    try:
        # Every stall of this program's event loop, with its cause, goes into the run's
        # folder: on 15 Nov 2026 long calls kept "falling behind" and only a guess was left.
        watch = FreezeWatch()
        watch.start()
        try:
            batch = await run_batch(plans, FoodPack(), run_dir, settings)
        finally:
            await watch.stop()
            write_stalls(run_dir / "stalls.md", watch.stalls)
    finally:
        if args.caller == "chain":
            await maker.close()
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


@dataclass
class Judge:
    """A ready judge: how to ask it, its model (kept with every mark), and how to close its client."""

    ask: Ask
    model: str
    close: Callable[[], Awaitable[None]]


def _gemini_judge(timeout_s: float, on_retry, on_usage) -> Judge:
    from common.google import google_client
    from lab.judge.judge import GEMINI_MODEL, gemini_judge

    client = google_client(require("GEMINI_API_KEY"), timeout_s)
    return Judge(gemini_judge(client, on_retry=on_retry), GEMINI_MODEL, client.aio.aclose)


def _groq_judge(timeout_s: float, on_retry, on_usage) -> Judge:
    from common.groq import groq_client
    from lab.judge.judge import GROQ_MODEL, groq_judge

    client = groq_client(groq_key(), timeout_s)
    return Judge(groq_judge(client, on_retry=on_retry, on_usage=on_usage), GROQ_MODEL, client.aclose)


JUDGES = {"gemini": _gemini_judge, "groq": _groq_judge}


def _make_judge(name: str) -> Judge:
    """The chosen judge, with every retry and token count printed as it happens."""
    load_env()

    def retrying(error: BaseException, wait: float) -> None:
        said = http_status(error) or type(error).__name__
        print(f"  {name} said {said}; asking again in {wait:.0f} s", flush=True)

    def used(usage: dict) -> None:
        print(f"  used {usage.get('total_tokens', '?')} tokens", flush=True)

    settings = load_settings()
    judge = JUDGES[name](settings.judge_timeout_s, retrying, used)
    return dataclasses.replace(judge, ask=paced(judge.ask, settings.judge_gap_s))


async def judge_check(name: str) -> tuple[dict, Path]:
    """L11: the judge on the planted calls, run twice for steadiness."""
    from lab.health.check import judge_light
    from lab.judge.check import changed_share, judge_report, run_planted

    judge = _make_judge(name)
    started = time.perf_counter()

    def show(run: int):
        def on_case(number: int, total: int, case_id: str) -> None:
            print(f"run {run}, case {number}/{total}: {case_id} ({time.perf_counter() - started:.0f} s in)", flush=True)
        return on_case

    try:
        graded, first = await run_planted(judge.ask, on_case=show(1))
        _again, second = await run_planted(judge.ask, on_case=show(2))
    finally:
        await judge.close()
    report = {"model": judge.model, **judge_report(graded, changed_share(first, second))}
    folder = make_run_dir(RUNS, f"judge-check-{name}")
    light = judge_light(graded.caught, graded.planted, len(graded.false_alarms), len(graded.refused))
    report["light"] = {"status": light.status, "detail": light.detail}
    (folder / "judge.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    return report, folder


async def judge_a_run(run_dir: Path, name: str) -> None:
    from lab.judge.apply import judge_run

    settings = load_settings()
    judge = _make_judge(name)
    try:
        scores = await judge_run(run_dir, judge.ask, settings, judge.model)
    finally:
        await judge.close()
    _write_results(run_dir, scores, settings, {"Judged calls": len(scores), "Judge": judge.model})


async def caller_report(run_dir: Path, with_ai: bool):
    """C15. The AI reads each call once, on the free tier, paced like the judge."""
    settings = load_settings()
    caller_settings = load_caller_settings()
    if not with_ai:
        return await report_on_run(run_dir, caller_settings, settings.sound_level)
    load_env()
    client = groq_client(groq_key(), settings.judge_timeout_s)
    try:
        ask = paced(groq_judge(client, REPORT_MODEL, "low", form=AI_FORM), settings.judge_gap_s)
        return await report_on_run(run_dir, caller_settings, settings.sound_level, ask)
    finally:
        await client.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description="The test lab.")
    commands = parser.add_subparsers(dest="command", required=True)
    run_cmd = commands.add_parser("run", help="run calls")
    run_cmd.add_argument("--robots", default="puppet:perfect")
    run_cmd.add_argument("--repeats", type=int, default=1)
    run_cmd.add_argument("--at-once", type=int, default=0)
    run_cmd.add_argument("--label", default="run")
    run_cmd.add_argument("--caller", choices=["scripted", "chain"], default="scripted")
    run_cmd.add_argument("--tasks", default="", help="only these, e.g. food_01,food_07")
    run_cmd.add_argument("--brain", default="", help="the chain caller's brain for this run, e.g. gemini:gemini-3.5-flash-lite")
    card = commands.add_parser("caller-report", help="mark the pretend caller on a run (C15)")
    card.add_argument("folder")
    card.add_argument("--no-ai", action="store_true", help="only the checks code can do")
    health = commands.add_parser("health", help="check the lab itself")
    health.add_argument("--quick", action="store_true")
    again = commands.add_parser("rescore", help="re-mark an old run")
    again.add_argument("folder")
    checked = commands.add_parser("judge-check", help="check the judge on planted mistakes")
    judged = commands.add_parser("judge", help="the judge marks a run's calls")
    judged.add_argument("folder")
    for command in (checked, judged):
        # Groq's gpt-oss-120b: caught 6 of 6 planted mistakes with no false alarm and
        # no change between two runs (1 Nov 2026); Gemini 3.8 Flash's free tier
        # (20 requests a day) can't run one judge check.
        command.add_argument("--judge", choices=sorted(JUDGES), default="groq")
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
        report, folder = asyncio.run(judge_check(args.judge))
        print(json.dumps(report, indent=1, ensure_ascii=False))
        print(f"report: {folder / 'judge.json'}")
        sys.exit(0 if report["light"]["status"] == "green" else 1)
    elif args.command == "judge":
        asyncio.run(judge_a_run(Path(args.folder), args.judge))
        print(f"results: {Path(args.folder) / 'results.md'}")
    elif args.command == "caller-report":
        report = asyncio.run(caller_report(Path(args.folder), not args.no_ai))
        print(report.markdown())
    else:
        asyncio.run(rescore(Path(args.folder)))
        print(f"results: {Path(args.folder) / 'results.md'}")


if __name__ == "__main__":
    main()
