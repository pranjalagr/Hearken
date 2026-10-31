"""Serves one call's backend over MCP (stdio), so every agent version (A, B
and C) plugs into exactly the same tools.

One server process per call. The database it was started with decides which
call's copy every tool touches (decision 11): no tool input can point at
another call's data.
"""

from __future__ import annotations

import mcp_types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from engine.service.backend import Backend


def build_server(backend: Backend) -> Server:
    async def list_tools(ctx, params) -> types.ListToolsResult:
        return types.ListToolsResult(
            tools=[
                types.Tool(name=spec.name, description=spec.description, input_schema=spec.input_schema)
                for spec in backend.tools
            ]
        )

    async def call_tool(ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
        reply = await backend.call_tool(params.name, params.arguments or {})
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=reply.to_json())],
            is_error=reply.is_error,
        )

    return Server("pakka", on_list_tools=list_tools, on_call_tool=call_tool)


async def serve_stdio(backend: Backend) -> None:
    """Serve until the agent closes the connection."""
    server = build_server(backend)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())
