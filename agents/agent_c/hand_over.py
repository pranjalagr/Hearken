"""One hand-over per call (decisions 13 and 19), done by code.

Agent A handed over 7 and 4 times in a row on food_05 and food_06 (practice run 7,
15 Nov 2026). Here a hand-over happens at most once: the backend's hand-over tool is
pressed once, the caller hears one fixed line, and the call ends after it has played
(the end_call pattern of agents/common/pipecat_hangup.py). Whoever asks first wins:
the Thinker pressing the tool (code routes it here), the repeat check, a failed step,
the Guard's ladder running out, or a change whose result stayed unknown.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from agents.agent_c.desk import JobDesk
from agents.common.diary import Diary
from agents.common.tools import Tools
from engine.service.replies import ERROR, ToolReply

HANDED_OVER = "handed_over"


class HandOver:
    def __init__(self, tools: Tools, tool: str, desk: JobDesk, say_and_end: Callable[[str], Awaitable[None]],
                 line: Callable[[], str], diary: Diary) -> None:
        self._tools = tools
        self.tool = tool  # the pack's hand-over tool name (settings.yaml)
        self._desk = desk
        self._say_and_end = say_and_end  # plays the line, then ends the call
        self._line = line  # the fixed line, in the call's language
        self._diary = diary
        self.reason: str | None = None

    @property
    def done(self) -> bool:
        return self.reason is not None

    async def now(self, reason: str, order_no: Any = None) -> ToolReply:
        """Press the hand-over tool (once), say so once, end the call."""
        if self.done:
            return ToolReply(ERROR, "Already handed over to a human on this call. Do not hand over again.")
        self.reason = reason
        inputs: dict[str, Any] = {"reason": reason}
        if order_no is not None:
            inputs["order_no"] = order_no
        try:
            reply = await self._tools.call(self.tool, inputs)
        except Exception as error:
            reply = ToolReply(ERROR, f"No answer from the service ({type(error).__name__}).")
        self._diary.decision("hand over", reason=reason, status=reply.status)
        self._desk.end_job(HANDED_OVER)
        await self._say_and_end(self._line())
        return reply
