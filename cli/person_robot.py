"""The line between a person's web phone and the pretend caller (Part 3, C14).

Started by cli.talk_to_caller, the way the lab starts any robot. It says "ready" only
once the person has joined, so the caller never dials an empty line.
"""

from __future__ import annotations

import argparse
import asyncio

from common.wire.robot_line import bound_port, serve_robot
from lab.caller.person_on_line import PersonBridge


async def main(port: int) -> str:
    server_port: list[int] = []
    bridge = PersonBridge(lambda: print(f"ready ws://127.0.0.1:{server_port[0]}/call", flush=True))
    server = await serve_robot(bridge.answer, "127.0.0.1", port)
    server_port.append(bound_port(server))
    try:
        await bridge.ended.wait()
    finally:
        server.close()
        await server.wait_closed()
    return bridge.end_reason


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Join a person's web phone to the pretend caller.")
    parser.add_argument("--port", type=int, default=8765)
    print(f"ended {asyncio.run(main(parser.parse_args().port))}", flush=True)
