"""Hands the ears' partial words to the Guard, so a "ruko" stops a ticket the moment it
is heard, not when the caller finishes (decision 21)."""

from __future__ import annotations

from collections.abc import Callable

from pipecat.frames.frames import InterimTranscriptionFrame, TranscriptionFrame
from pipecat.observers.base_observer import BaseObserver, FramePushed


class PartialWords(BaseObserver):
    def __init__(self, on_words: Callable[[str], None]) -> None:
        super().__init__()
        self._on_words = on_words
        self._seen: set[int] = set()

    async def on_push_frame(self, data: FramePushed) -> None:
        frame = data.frame
        if isinstance(frame, (InterimTranscriptionFrame, TranscriptionFrame)) and frame.id not in self._seen:
            self._seen.add(frame.id)
            self._on_words(frame.text)
