"""Run the puppet robot as its own program, for one call (Part 5, L3).

    python -m cli.puppet_robot --db <call copy> --task <task id> --variant perfect|careless
                               [--port 0] [--bells URL] [--diary PATH] [--backend-events PATH]
                               [--no-audio-s 30] [--max-call-s 900] [--change-wait-s 10]

The lab starts every robot this way: given its own database copy and a phone
port, and nothing else. The robot opens its tools itself over MCP (serve_call
on that copy, through pipes), so the tools can only touch that one call.

When it can take the call it prints one line, "ready ws://127.0.0.1:PORT/call",
and after the call it prints "ended <reason>" and exits. A crash exits with
an error code, which the lab counts as the robot's failure (L4).
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from agents.common.call_box import CallBoxSettings
from agents.common.diary import NO_DIARY, Diary
from agents.common.tools import mcp_tools
from common.env import load_env, require
from common.stopwatch import JsonlSink
from common.wire.robot_line import RobotLine, bound_port, serve_robot
from engine.lab_controls.clock import WallClock
from engine.safety.guard_door import GuardDoor
from engine.storage.call_settings import read_call_settings
from engine.storage.database import sqlite_session
from lab.dummy_robot.real_voice import SavedRobotVoice, lines_of
from lab.dummy_robot.robot import DummyRobot
from lab.puppets.food import script_for


async def main(args: argparse.Namespace) -> str:
    steps, _mode = script_for(args.task, args.variant)
    db_path = Path(args.db)
    sink = JsonlSink(Path(args.diary)) if args.diary else None
    extras = {}
    if args.voice == "sarvam":
        # Real speech, for a caller with real ears (Part 3, C14). Made before "ready":
        # the puppet has to say each line the moment its turn comes.
        load_env()
        voice = SavedRobotVoice(require("SARVAM_API_KEY"))
        await asyncio.to_thread(voice.make, lines_of(steps))
        extras = {"voice": voice}
    async with mcp_tools(db_path, args.bells, Path(args.backend_events) if args.backend_events else None) as tools:
        # The Guard's private door, on its own connection (Part 2): never a tool.
        async with sqlite_session(db_path) as guard_db:
            settings = await read_call_settings(guard_db)
            guard = GuardDoor(guard_db, WallClock(settings.started_at_wall))
            diary = Diary(settings.call_id, sink) if sink else NO_DIARY
            box = CallBoxSettings(no_audio_s=args.no_audio_s, max_call_s=args.max_call_s, change_wait_s=args.change_wait_s)
            robot = DummyRobot(steps, tools, guard, box_settings=box, diary=diary, **extras)
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
        # The robot itself broke: say so loudly, and exit with an error.
        raise RuntimeError(f"robot crashed: {result.error!r}")
    return result.reason if result is not None else "never answered"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The puppet robot, for one lab call.")
    parser.add_argument("--db", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--variant", choices=["perfect", "careless"], required=True)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--bells", default=None)
    parser.add_argument("--diary", default=None)
    parser.add_argument("--backend-events", default=None)
    parser.add_argument("--no-audio-s", type=float, default=30.0)
    parser.add_argument("--max-call-s", type=float, default=15 * 60.0)
    parser.add_argument("--change-wait-s", type=float, default=10.0)
    parser.add_argument("--voice", choices=["synthetic", "sarvam"], default="synthetic")
    reason = asyncio.run(main(parser.parse_args()))
    print(f"ended {reason}", flush=True)
