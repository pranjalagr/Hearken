"""Talk to the pretend caller yourself, as if you were the support robot (Part 3, C14).

    python -m cli.talk_to_caller --task food_01

Then open http://127.0.0.1:8080 in Chrome or Edge and press Call. The pretend caller
phones you with that task's goal and personality. The question for you: does it feel
like a real customer? Each call is saved under runs/lab/.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from cli.phone_page import make_server
from engine.tasks.task_file import load_tasks
from lab.bell_board import BellBoard
from lab.caller.parts import RealCaller, missing_keys
from lab.caller.settings import load_caller_settings
from lab.runner.batch import make_run_dir
from lab.runner.run_call import CallPlan, run_call
from lab.settings import load_settings
from packs.food.pack import FoodPack

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "lab"
PRACTICE = ROOT / "packs" / "food" / "tasks" / "practice"
ROBOT_PORT = 8765  # the web phone's default robot address
PAGE_PORT = 8080
# The "robot" says ready only once the person presses Call, which can take a while.
JOIN_WAIT_S = 300.0


@dataclass(frozen=True)
class PersonRobot:
    name: str = "person"

    def command(self, db, folder, bells_url, settings, task) -> list[str]:
        return [sys.executable, "-m", "cli.person_robot", "--port", str(ROBOT_PORT)]


def show_call(folder: Path) -> None:
    """What was said, including what the caller heard from you."""
    rows = []
    for raw in (folder / "events.jsonl").read_text(encoding="utf-8").splitlines():
        event = json.loads(raw)
        if event["event"] == "caller.line":
            rows.append((event.get("start_ms") or 0, f"caller:  {event['text']}"))
        elif event["event"] == "caller.heard":
            rows.append((event["caller_ms"], f"it heard you: {event['text']}"))
        elif event["event"] == "director.said":
            rows.append((event["caller_ms"], f"caller (director): {event['line']}"))
    for at, text in sorted(rows, key=lambda row: row[0]):
        print(f"{at / 1000:6.1f}  {text}")


async def main(task_id: str) -> None:
    tasks = [task for task in load_tasks(PRACTICE) if task.id.startswith(task_id)]
    if len(tasks) != 1:
        raise SystemExit(f"{task_id!r} matches {len(tasks)} tasks; try one like food_01")
    settings = load_caller_settings()
    missing = missing_keys(settings)
    if missing:
        raise SystemExit(f"missing in pakka/.env: {', '.join(missing)}")
    task = tasks[0]
    maker = RealCaller(settings)
    await maker.warm_up()
    page = make_server(PAGE_PORT)
    threading.Thread(target=page.serve_forever, daemon=True).start()
    board = BellBoard()
    bells_url = await board.start()
    run_dir = make_run_dir(RUNS, f"talk-{task_id}")
    lab = load_settings().with_changes(robot_ready_timeout_s=JOIN_WAIT_S)
    print(f"\nYou are the support agent. Open http://127.0.0.1:{PAGE_PORT} and press Call.")
    print(f"Task {task.id}: the caller is {task.caller.name}, {task.caller.personality}.\n", flush=True)
    plan = CallPlan(f"talk-{task_id}", task, PersonRobot(), task.modes[0], seed=1, caller=maker)
    try:
        outcome = await run_call(plan, FoodPack(), run_dir, board, bells_url, lab)
    finally:
        page.shutdown()
        await board.close()
        await maker.close()
    print(f"\nThe call ended ({outcome.end_reason}). What was said:\n")
    show_call(Path(outcome.folder))
    print(f"\nSaved in {outcome.folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Talk to the pretend caller yourself.")
    parser.add_argument("--task", default="food_01", help="for example food_01 or food_12")
    asyncio.run(main(parser.parse_args().task))
