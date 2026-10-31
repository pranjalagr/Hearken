"""Make a fresh call copy from a task file and print where it is.

    python -m cli.make_call --task packs/food/tasks/food_01_cancel_prepaid.yaml --mode safe

Then start its tool server with:  python -m cli.serve_call --db <printed path>
"""

from __future__ import annotations

import argparse
import asyncio
import uuid
from pathlib import Path

from engine.storage.call_factory import CallFactory
from engine.tasks.task_file import load_task
from engine.tasks.task_setup import start_call
from packs.registry import get_pack


async def main(task_path: str, mode: str, ticket_check: bool, workdir: str) -> Path:
    task = load_task(Path(task_path))
    pack = get_pack(task.pack)
    factory = CallFactory(pack, Path(workdir))
    call_id = f"{task.id}-{mode}-{uuid.uuid4().hex[:6]}"
    return await start_call(factory, pack, task, call_id, mode, ticket_check)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make a fresh call copy from a task file.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--mode", choices=["plain", "safe"], default="safe")
    parser.add_argument("--ticket-check", choices=["on", "off"], default=None,
                        help="default: on in safe mode, off in plain mode")
    parser.add_argument("--workdir", default="runs")
    args = parser.parse_args()
    ticket_check = (args.mode == "safe") if args.ticket_check is None else args.ticket_check == "on"
    print(asyncio.run(main(args.task, args.mode, ticket_check, args.workdir)))
