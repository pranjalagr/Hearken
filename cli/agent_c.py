"""Run Agent C as its own program, for one call (Part 5, L3).

    python -m cli.agent_c --db <call copy> [--port 0] [--bells URL] [--diary PATH]
                          [--backend-events PATH] [--no-audio-s 30] [--max-call-s 900]
                          [--change-wait-s 10] [--max-brain-requests 40]

Started by the lab exactly like Agent A (same arguments, less Agent A's
background-tool switch, which has no meaning here), and like the puppets: given its own database copy and a
phone port, nothing else. It opens its tools itself over MCP, so they can only
touch that one call. Prints "ready ws://127.0.0.1:PORT/call" when it can take
the call, "ended <reason>" after it. A crash exits with an error code (L4).
Needs GEMINI_API_KEY and SARVAM_API_KEY in pakka/.env.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from loguru import logger

from agents.agent_c.robot import AgentC, Keys
from agents.agent_c.settings import load_settings
from agents.common.call_box import CallBoxSettings
from agents.common.diary import NO_DIARY, Diary
from agents.common.tools import mcp_tools
from common.env import gemini_key, load_env, sarvam_key
from common.stopwatch import JsonlSink
from common.wire.robot_line import RobotLine, bound_port, serve_robot
from engine.storage.call_settings import read_call_settings
from engine.storage.database import sqlite_session
from packs.registry import get_pack


async def main(args: argparse.Namespace) -> str:
    load_env()
    keys = Keys(gemini=gemini_key(), sarvam=sarvam_key())  # read here, never printed
    settings = load_settings()
    db_path = Path(args.db)
    sink = JsonlSink(Path(args.diary)) if args.diary else None
    async with sqlite_session(db_path) as db:
        call_settings = await read_call_settings(db)
    call_id = call_settings.call_id
    words = get_pack(call_settings.pack).spoken_words()  # statuses and money said as the pack says them
    async with mcp_tools(db_path, args.bells, Path(args.backend_events) if args.backend_events else None) as tools:
        diary = Diary(call_id, sink) if sink else NO_DIARY
        box = CallBoxSettings(no_audio_s=args.no_audio_s, max_call_s=args.max_call_s, change_wait_s=args.change_wait_s)
        robot = AgentC(settings, keys, tools, box_settings=box, diary=diary, max_brain_requests=args.max_brain_requests,
                       words=words)
        answered = asyncio.Event()

        async def answer(line: RobotLine) -> None:
            try:
                await robot(line)
            finally:
                answered.set()

        server = await serve_robot(answer, "127.0.0.1", args.port)
        try:
            print(f"ready ws://127.0.0.1:{bound_port(server)}/call", flush=True)
            await answered.wait()
        finally:
            server.close()
            await server.wait_closed()
            if sink is not None:
                sink.close()
    result = robot.result
    if result is not None and result.error is not None:
        raise RuntimeError(f"robot crashed: {result.error!r}")
    return result.reason if result is not None else "never answered"


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agent C, the team robot, for one call.")
    parser.add_argument("--db", required=True)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--bells", default=None)
    parser.add_argument("--diary", default=None)
    parser.add_argument("--backend-events", default=None)
    parser.add_argument("--no-audio-s", type=float, default=30.0)
    parser.add_argument("--max-call-s", type=float, default=900.0)
    parser.add_argument("--change-wait-s", type=float, default=10.0)
    parser.add_argument("--max-brain-requests", type=int, default=40, help="hang up after this many (lab/settings.yaml)")
    return parser.parse_args(argv)


if __name__ == "__main__":
    # Pipecat logs every frame at DEBUG; the robot's own diary is the record (L5),
    # so its stderr keeps only warnings and errors.
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    print(f"ended {asyncio.run(main(parse()))}", flush=True)
