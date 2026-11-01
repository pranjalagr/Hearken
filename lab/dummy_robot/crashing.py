"""A robot that crashes in the middle of a call, on purpose (Part 5, L4).

    python -m lab.dummy_robot.crashing

It says "ready", takes the call, sends half a second of sound, then its
program dies at once with an error code, as a real robot's crash would look
from outside. The lab must count that against the robot, never rerun it as a
lab problem.
"""

from __future__ import annotations

import asyncio
import os

from common.wire.robot_line import RobotLine, bound_port, serve_robot
from lab.synthetic_voice import synthetic_line

CRASH_CODE = 3


async def main() -> None:
    async def answer(line: RobotLine) -> None:
        for piece in synthetic_line("Namaste, ek")[:25]:
            await line.send_audio(piece)
        await asyncio.sleep(0.5)
        os._exit(CRASH_CODE)  # no goodbye, no cleanup: a crash

    server = await serve_robot(answer)
    print(f"ready ws://127.0.0.1:{bound_port(server)}/call", flush=True)
    await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
