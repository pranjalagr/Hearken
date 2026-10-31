"""Run the MCP tool server for ONE call.

    python -m cli.serve_call --db runs/calls/call-1.db

The database file named here decides which call's copy every tool touches
(decision 11). There is no tool input that can point anywhere else.
"""

from __future__ import annotations

import argparse
import asyncio

from engine.lab_controls.clock import WallClock
from engine.service.backend import Backend
from engine.service.mcp_server import serve_stdio
from engine.storage.call_settings import read_call_settings
from engine.storage.database import sqlite_session
from packs.registry import get_pack


async def main(db_path: str) -> None:
    async with sqlite_session(db_path) as db:
        settings = await read_call_settings(db)
        backend = Backend(db, get_pack(settings.pack), WallClock(settings.started_at_wall), settings)
        await serve_stdio(backend)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve one call's tools over MCP (stdio).")
    parser.add_argument("--db", required=True, help="the call's database copy")
    asyncio.run(main(parser.parse_args().db))
