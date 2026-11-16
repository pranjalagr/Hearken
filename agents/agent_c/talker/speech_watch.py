"""Watches the pipeline's frames for when the robot's voice starts, stops and is cut off,
and when the Talker finished a reply.

An observer, like the diary's: it adds nothing to the pipeline. Pipecat pushes
"bot stopped speaking" both ways (two frames, one moment), and the same frame
passes many processors, so each moment is reported once.
"""

from __future__ import annotations

from collections.abc import Callable

from pipecat.frames.frames import (BotStartedSpeakingFrame, BotStoppedSpeakingFrame, InterruptionFrame, LLMFullResponseEndFrame,
                                   TTSTextFrame)
from pipecat.observers.base_observer import BaseObserver, FramePushed

_WATCHED = (LLMFullResponseEndFrame, BotStartedSpeakingFrame, InterruptionFrame, BotStoppedSpeakingFrame, TTSTextFrame)


class SpeechWatch(BaseObserver):
    def __init__(self, on_reply_written: Callable[[], None], on_speech_ended: Callable[[bool, str], None]) -> None:
        super().__init__()
        self._on_reply_written = on_reply_written
        self._on_speech_ended = on_speech_ended
        self._seen: set[int] = set()
        self.speaking = False
        self._cut = False
        self._words: list[str] = []  # what the voice said since it started, for the Guard's mark
        # The output transport. Its text frames are counted only as it passes them on, which
        # it does in step with their audio: seen earlier, a question's text arrived while the
        # line before it was still playing, so the Guard's mark went out ~2 s before the
        # question ended and an early "haan" could count (food_01, 16 Nov 2026). None: every
        # text frame counts (tests without a transport).
        self.output: object | None = None

    async def on_push_frame(self, data: FramePushed) -> None:
        frame = data.frame
        if not isinstance(frame, _WATCHED) or frame.id in self._seen:
            return
        if isinstance(frame, TTSTextFrame) and self.output is not None and data.source is not self.output:
            return  # not played yet: counted when the output passes it on
        self._seen.add(frame.id)
        if isinstance(frame, LLMFullResponseEndFrame):
            self._on_reply_written()
        elif isinstance(frame, BotStartedSpeakingFrame):
            if not self.speaking:
                # The words are NOT cleared here: when the voice starts from silence, its
                # text arrives before its sound, and clearing here threw the question away.
                # The Guard then saw "" for a question that played in full, sent no mark,
                # and every yes counted as early (food_03, 16 Nov 2026). They are cleared
                # once reported, below.
                self.speaking, self._cut = True, False
        elif isinstance(frame, TTSTextFrame):
            self._words.append(frame.text)
        elif isinstance(frame, InterruptionFrame):
            if self.speaking:
                self._cut = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            if self.speaking:  # the second of the pair finds it already stopped
                self.speaking = False
                words, self._words = " ".join(self._words), []
                self._on_speech_ended(self._cut, words)
