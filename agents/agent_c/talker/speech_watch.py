"""Watches the pipeline's frames for when the robot's voice starts, stops and is cut off,
and when the Talker finished a reply.

An observer, like the diary's: it adds nothing to the pipeline. Pipecat pushes
"bot stopped speaking" both ways (two frames, one moment), and the same frame
passes many processors, so each moment is reported once.
"""

from __future__ import annotations

from collections.abc import Callable

from pipecat.frames.frames import BotStartedSpeakingFrame, BotStoppedSpeakingFrame, InterruptionFrame, LLMFullResponseEndFrame
from pipecat.observers.base_observer import BaseObserver, FramePushed

_WATCHED = (LLMFullResponseEndFrame, BotStartedSpeakingFrame, InterruptionFrame, BotStoppedSpeakingFrame)


class SpeechWatch(BaseObserver):
    def __init__(self, on_reply_written: Callable[[], None], on_speech_ended: Callable[[bool], None]) -> None:
        super().__init__()
        self._on_reply_written = on_reply_written
        self._on_speech_ended = on_speech_ended
        self._seen: set[int] = set()
        self.speaking = False
        self._cut = False

    async def on_push_frame(self, data: FramePushed) -> None:
        frame = data.frame
        if not isinstance(frame, _WATCHED) or frame.id in self._seen:
            return
        self._seen.add(frame.id)
        if isinstance(frame, LLMFullResponseEndFrame):
            self._on_reply_written()
        elif isinstance(frame, BotStartedSpeakingFrame):
            if not self.speaking:
                self.speaking, self._cut = True, False
        elif isinstance(frame, InterruptionFrame):
            if self.speaking:
                self._cut = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            if self.speaking:  # the second of the pair finds it already stopped
                self.speaking = False
                self._on_speech_ended(self._cut)
