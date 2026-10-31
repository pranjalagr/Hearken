"""Answer calls with the echo robot: it plays your voice back 2 s later.

    python -m cli.dummy_robot [--port 8765] [--delay 2.0]

The quickest way to check the web phone by ear: call
ws://127.0.0.1:8765/call, say something, and hear it come back.
"""

from __future__ import annotations

import argparse
import asyncio

from common.wire.robot_line import RobotLine, bound_port, serve_robot
from lab.dummy_robot.robot import EchoRobot


async def main(port: int, delay_s: float) -> None:
    async def answer(line: RobotLine) -> None:
        # A fresh robot per call, so two calls never share state.
        await EchoRobot(delay_s=delay_s)(line)

    server = await serve_robot(answer, "127.0.0.1", port)
    print(f"echo robot: ws://127.0.0.1:{bound_port(server)}/call", flush=True)
    await server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Answer calls with the echo robot.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--delay", type=float, default=2.0, help="seconds before your voice comes back")
    args = parser.parse_args()
    try:
        asyncio.run(main(args.port, args.delay))
    except KeyboardInterrupt:
        pass
