"""How a robot reaches the shop's tools (Part 5, L3).

In a lab run the robot is its own program and reaches its tools over MCP:
it starts `serve_call --db <its copy>` itself, through pipes, so the tools
can only ever touch that one call's copy. Tests that need no second program
use the backend directly. Both give the same reply shape, so robot code never
knows which one it has.
"""

from __future__ import annotations

import contextlib
import json
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Protocol

from engine.service.backend import Backend
from engine.service.replies import ERROR, ToolReply

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Tools(Protocol):
    async def call(self, tool: str, inputs: dict[str, Any]) -> ToolReply: ...


class BackendTools:
    """The backend in the same program: for tests."""

    def __init__(self, backend: Backend) -> None:
        self._backend = backend

    async def call(self, tool: str, inputs: dict[str, Any]) -> ToolReply:
        return await self._backend.call_tool(tool, inputs)


class McpTools:
    """The real tool server, over MCP, as every lab robot uses it."""

    def __init__(self, session: Any) -> None:
        self._session = session

    async def call(self, tool: str, inputs: dict[str, Any]) -> ToolReply:
        result = await self._session.call_tool(tool, inputs)
        try:
            body = json.loads(result.content[0].text)
        except (IndexError, AttributeError, ValueError):
            # Not our server's reply shape: say so, never guess a status.
            return ToolReply(ERROR, f"unreadable tool reply for {tool}")
        return ToolReply(body.get("status", ERROR), body.get("message", ""), body.get("data") or {})


@contextlib.asynccontextmanager
async def mcp_tools(db_path: Path, bells_url: str | None = None, events_path: Path | None = None) -> AsyncIterator[McpTools]:
    """Start this call's tool server and connect to it; stops it on exit."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    args = ["-m", "cli.serve_call", "--db", str(db_path)]
    if bells_url:
        args += ["--bells", bells_url]
    if events_path:
        args += ["--events", str(events_path)]
    server = StdioServerParameters(command=sys.executable, args=args, cwd=str(PROJECT_ROOT))
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield McpTools(session)
