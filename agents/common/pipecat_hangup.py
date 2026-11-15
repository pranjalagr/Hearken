"""The robot hangs up (decision 13: "End: once the work is done and the caller has
nothing more").

Agent A had no way to end a call on 15 Nov 2026: after a goodbye or a hand-over the
call ran on until the pretend caller gave up, one brain request per extra turn (50 in
one call, about 33,000 of the caller's tokens). Voice agents built on Pipecat
normally give the brain an end_call tool; this is it.

The goodbye plays first: the end is pushed downstream after the words already on
their way to the voice, so they are flushed before the pipeline ends. No extra brain
request follows the hang-up (run_llm=False).
"""

from __future__ import annotations

from pipecat.adapters.llm.schemas.function_schema import FunctionSchema
from pipecat.frames.frames import EndWorkerFrame, FunctionCallResultProperties
from pipecat.services.llm_service import FunctionCallParams, LLMService

from agents.common.diary import Diary

END_CALL = "end_call"
END_CALL_SCHEMA = FunctionSchema(
    name=END_CALL,
    description=("Hang up the call. Use it right after you said goodbye once the caller's problem is solved, "
                 "or right after you told the caller you handed them to a human agent."),
    properties={"reason": {"type": "string", "description": "Why the call ends, in a few words."}},
    required=[],
)


class HangUp:
    def __init__(self, diary: Diary) -> None:
        self._diary = diary
        self.reason: str | None = None  # set once the brain hangs up

    def connect(self, llm: LLMService) -> None:
        llm.register_function(END_CALL, self._handle)

    async def _handle(self, params: FunctionCallParams) -> None:
        self.reason = str((params.arguments or {}).get("reason") or "done")
        self._diary.decision("hang up", reason=self.reason)
        await params.result_callback({"status": "ok"}, properties=FunctionCallResultProperties(run_llm=False))
        await params.llm.push_frame(EndWorkerFrame(reason=self.reason))
