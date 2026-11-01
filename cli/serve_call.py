"""Run the MCP tool server for ONE call.

    python -m cli.serve_call --db runs/calls/call-1.db [--bells ws://127.0.0.1:PORT/bells]
                             [--events runs/<run>/calls/<call>/backend.jsonl]

The database file named here decides which call's copy every tool touches
(decision 11). There is no tool input that can point anywhere else.

--bells connects to the lab's bell board, so the director hears the exact
moments of each action (Part 3, C10). Robots never see it: it is not a tool.

--events writes when each tool call starts and ends, on the one shared clock
(Part 5, L6 and L9). Also lab-only: the robot never reads it.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from common.stopwatch import JsonlSink, Stopwatch
from engine.lab_controls.bells import NO_BELLS, BellLine, SocketBells
from engine.lab_controls.clock import WallClock
from engine.service.backend import Backend
from engine.service.mcp_server import serve_stdio
from engine.storage.call_settings import read_call_settings
from engine.storage.database import sqlite_session
from packs.registry import get_pack


async def main(db_path: str, bells_url: str | None = None, events_path: str | None = None) -> None:
    async with sqlite_session(db_path) as db:
        settings = await read_call_settings(db)
        bells: BellLine = NO_BELLS
        if bells_url:
            bells = await SocketBells.connect(bells_url, settings.call_id)
        sink = JsonlSink(Path(events_path)) if events_path else None
        try:
            backend = Backend(
                db,
                get_pack(settings.pack),
                WallClock(settings.started_at_wall),
                settings,
                bells=bells,
                stopwatch=Stopwatch(settings.call_id, sink) if sink else None,
            )
            await serve_stdio(backend)
        finally:
            if isinstance(bells, SocketBells):
                await bells.aclose()
            if sink is not None:
                sink.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve one call's tools over MCP (stdio).")
    parser.add_argument("--db", required=True, help="the call's database copy")
    parser.add_argument("--bells", default=None, help="the lab's bell board URL (lab runs only)")
    parser.add_argument("--events", default=None, help="where to write tool start/end times (lab runs only)")
    args = parser.parse_args()
    asyncio.run(main(args.db, args.bells, args.events))
