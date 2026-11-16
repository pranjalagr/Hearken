"""The Talker's mouth: sits between the Talker and the voice, and lets only checked,
filled words through (decisions 8 and 10).

The Talker's reply is held whole until it ends, then the speaker cuts it to two
sentences, runs the number check, fills the slots and fixes spellings; the result goes
to the voice as one line. The Talker's own history keeps its reply with the slots
still in, never the filled numbers: seeing "4521" in its past replies would only tempt
it to type numbers itself.
"""

from __future__ import annotations

from pipecat.frames.frames import (Frame, InterruptionFrame, LLMFullResponseEndFrame, LLMFullResponseStartFrame,
                                   LLMTextFrame, TTSSpeakFrame)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from agents.agent_c.talker.speaker import Speaker
from agents.common.diary import Diary


class Mouth(FrameProcessor):
    def __init__(self, speaker: Speaker, context: LLMContext, diary: Diary, **kwargs) -> None:
        super().__init__(**kwargs)
        self._speaker = speaker
        self._context = context
        self._diary = diary
        self._parts: list[str] | None = None  # the reply being written, None between replies

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, LLMFullResponseStartFrame):
            self._parts = []
        elif direction == FrameDirection.DOWNSTREAM and isinstance(frame, LLMTextFrame) and self._parts is not None:
            self._parts.append(frame.text)
        elif direction == FrameDirection.DOWNSTREAM and isinstance(frame, LLMFullResponseEndFrame):
            parts, self._parts = self._parts, None
            await self._speak("".join(parts or []))
        else:
            if isinstance(frame, InterruptionFrame):
                self._parts = None  # the caller cut in: what was being written is dropped
            await self.push_frame(frame, direction)

    async def _speak(self, reply: str) -> None:
        if not reply.strip():
            return
        speech = self._speaker.prepare(reply)
        self._context.add_message({"role": "assistant", "content": speech.reply})
        if speech.safe:
            self._diary.decision("safe line", reply=speech.reply, why=speech.why)
        await self.push_frame(TTSSpeakFrame(speech.spoken, append_to_context=False))
