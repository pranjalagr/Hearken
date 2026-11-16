"""The Talker's one tool: ask_thinker (decision 7, "only the Talker opens a job").

The Talker presses no backend button. When the caller wants something done or
looked up, it calls ask_thinker; code opens a job and the Thinker does the work
in the background. The tool answers at once, so the Talker can say "ek second"
in the same breath instead of waiting on the backend: Agent A's 3.4 s median
reply (15 Nov 2026) was mostly the brain waiting on its tools.
"""

from __future__ import annotations

from typing import Protocol

from pipecat.adapters.llm.schemas.function_schema import FunctionSchema
from pipecat.services.llm_service import FunctionCallParams, LLMService

from agents.common.diary import Diary

ASK_THINKER = "ask_thinker"
ASK_THINKER_SCHEMA = FunctionSchema(
    name=ASK_THINKER,
    description=("Pass the caller's request to the back office, which looks up and changes orders. "
                 "It answers later, as NOTES. Call it once per request; the caller's next words reach it by themselves."),
    properties={"request": {"type": "string",
                            "description": "What the caller wants, in their words, with any order number and name they gave."}},
    required=["request"],
)


class Desk(Protocol):
    """Where a job is opened: the Thinker's side of the robot."""

    def open_job(self, request: str) -> str:
        """Open a job (or pass the words to the open one). Returns the short status the Talker reads."""
        ...


class AskThinker:
    def __init__(self, desk: Desk, diary: Diary) -> None:
        self._desk = desk
        self._diary = diary

    def connect(self, llm: LLMService) -> None:
        # Never cancelled by a caller cutting in: opening a job takes no time, and a
        # half-opened job would be worse than one the caller talked over.
        llm.register_function(ASK_THINKER, self._handle, cancel_on_interruption=False)

    async def _handle(self, params: FunctionCallParams) -> None:
        request = str((params.arguments or {}).get("request") or "").strip()
        status = self._desk.open_job(request)
        self._diary.decision("ask thinker", request=request, status=status)
        # The Talker speaks next ("ek second"), from this answer alone.
        await params.result_callback({"status": status})

