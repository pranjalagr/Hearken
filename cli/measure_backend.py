"""Measure the Part 1 numbers to optimize, so the backend is never the slow part.

    python -m cli.measure_backend

Prints: time to make a call copy, tool call time (median and p90), and the
size of the tool list the AI reads on every turn.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from engine.lab_controls.clock import WallClock
from engine.service.backend import Backend
from engine.service.tool_catalog import build_catalog
from engine.storage.call_factory import CallFactory
from engine.storage.call_settings import CallSettings, read_call_settings
from engine.storage.database import sqlite_session
from packs.food.pack import FoodPack

COPIES = 50
CALLS = 300


def percentile(values: list[float], share: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(share * len(ordered)))]


async def main() -> None:
    pack = FoodPack()
    with tempfile.TemporaryDirectory() as workdir:
        factory = CallFactory(pack, Path(workdir))
        await factory.build_template()

        copy_ms = []
        for n in range(COPIES):
            start = time.perf_counter()
            path = await factory.new_call(CallSettings(call_id=f"m{n}", pack=pack.name))
            copy_ms.append((time.perf_counter() - start) * 1000)

        tool_ms = []
        async with sqlite_session(path) as db:
            settings = await read_call_settings(db)
            backend = Backend(db, pack, WallClock(settings.started_at_wall), settings)
            for _ in range(CALLS):
                start = time.perf_counter()
                await backend.call_tool("find_order", {"order_no": 4521, "customer_name": "Rahul"})
                tool_ms.append((time.perf_counter() - start) * 1000)
            reply = await backend.call_tool("find_order", {"order_no": 4521, "customer_name": "Rahul"})

        for mode, ticket_check in (("plain", False), ("safe", True)):
            tools = build_catalog(pack.actions(), CallSettings("x", pack.name, mode, ticket_check))
            text = json.dumps([asdict(tool) for tool in tools], separators=(",", ":"))
            # About 4 characters per token is a common rough rule for English text.
            print(f"tool list, {mode} mode: {len(tools)} tools, {len(text)} chars (~{len(text) // 4} tokens)")

    print(f"call copy:   median {statistics.median(copy_ms):.1f} ms, p90 {percentile(copy_ms, 0.9):.1f} ms")
    print(f"find_order:  median {statistics.median(tool_ms):.2f} ms, p90 {percentile(tool_ms, 0.9):.2f} ms")
    print(f"find_order reply: {len(reply.to_json())} chars (~{len(reply.to_json()) // 4} tokens)")


if __name__ == "__main__":
    asyncio.run(main())
