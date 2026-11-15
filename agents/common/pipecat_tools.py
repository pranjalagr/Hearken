"""The shop's tools, as a Pipecat brain sees and calls them (Agents A and B).

The brain gets exactly the tools the call's mode offers (decision 5), straight
from the tool server, so no agent keeps a list of its own that could drift from
the backend's. Every call goes to the backend unchanged and is written in the
diary with its status and time (L5).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from pipecat.adapters.llm.schemas.function_schema import FunctionSchema
from pipecat.adapters.llm.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import FunctionCallParams, LLMService

from agents.common.diary import Diary
from agents.common.tools import Tools
from engine.service.tool_catalog import ToolSpec

# Written in the diary when Pipecat cancels a running tool call because the caller
# cut in. The backend may still finish it: that gap is what the race tests measure.
CANCELLED = "cancelled_by_interruption"


def tools_schema(specs: list[ToolSpec], robot_tools: tuple[FunctionSchema, ...] = ()) -> ToolsSchema:
    """Our JSON schemas as Pipecat functions. Only properties and required are kept:
    Gemini's function format has no "additionalProperties". `robot_tools` are the robot's
    own, never the backend's (such as hanging up)."""
    return ToolsSchema(standard_tools=[
        FunctionSchema(name=spec.name, description=spec.description,
                       properties=dict(spec.input_schema.get("properties", {})),
                       required=list(spec.input_schema.get("required", [])))
        for spec in specs
    ] + list(robot_tools))


def connect_tools(llm: LLMService, tools: Tools, diary: Diary, *, background: bool) -> None:
    """Every tool the brain calls goes to the backend. `background` is Pipecat's own
    switch (cancel_on_interruption=False): the brain talks on while a tool runs, and the
    answer is added when it comes. Off, the brain waits, and a caller cutting in cancels
    the wait (Pipecat's default)."""

    async def handle(params: FunctionCallParams) -> None:
        name, inputs = params.function_name, dict(params.arguments or {})
        started = time.perf_counter()
        try:
            reply = await tools.call(name, inputs)
        except asyncio.CancelledError:
            diary.tool(name, inputs, CANCELLED, _ms_since(started))
            raise
        diary.tool(name, inputs, reply.status, _ms_since(started), message=reply.message)
        result: dict[str, Any] = {"status": reply.status, "message": reply.message, "data": reply.data}
        await params.result_callback(result)

    llm.register_function(None, handle, cancel_on_interruption=not background)


def _ms_since(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)
