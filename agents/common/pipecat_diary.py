"""The diary of a Pipecat robot (L5), written by watching the frames.

A Pipecat observer sees every frame passed between processors, so the diary is
filled without adding anything to the pipeline itself: the robot runs exactly
as Pipecat would run it. A frame passes many processors on its way, so each
one is written once.
"""

from __future__ import annotations

from pipecat.frames.frames import (
    ErrorFrame,
    InterruptionFrame,
    MetricsFrame,
    TranscriptionFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.metrics.metrics import LLMUsageMetricsData, TTSUsageMetricsData
from pipecat.observers.base_observer import BaseObserver, FramePushed

from agents.common.diary import Diary

_WATCHED = (TranscriptionFrame, TTSStartedFrame, TTSTextFrame, TTSStoppedFrame, InterruptionFrame, MetricsFrame, ErrorFrame)


class DiaryObserver(BaseObserver):
    def __init__(self, diary: Diary) -> None:
        super().__init__()
        self._diary = diary
        self._seen: set[int] = set()
        self._saying: list[str] = []  # the reply the voice is speaking now

    async def on_push_frame(self, data: FramePushed) -> None:
        frame = data.frame
        if not isinstance(frame, _WATCHED) or frame.id in self._seen:
            return
        self._seen.add(frame.id)
        if isinstance(frame, TranscriptionFrame):
            self._diary.heard(frame.text, language=str(frame.language) if frame.language else None)
        elif isinstance(frame, TTSStartedFrame):
            self._saying = []
        elif isinstance(frame, TTSTextFrame):
            self._saying.append(frame.text)
        elif isinstance(frame, TTSStoppedFrame):
            self._said(cut=False)
        elif isinstance(frame, InterruptionFrame):
            self._said(cut=True)  # what was being said when the caller cut in
        elif isinstance(frame, MetricsFrame):
            self._costs(frame)
        elif isinstance(frame, ErrorFrame):
            self._diary.error(str(frame.error), fatal=bool(frame.fatal))

    def _said(self, cut: bool) -> None:
        if self._saying:
            self._diary.said(" ".join(part.strip() for part in self._saying if part.strip()), cut=cut)
        self._saying = []

    def _costs(self, frame: MetricsFrame) -> None:
        for item in frame.data:
            service = item.model or item.processor
            if isinstance(item, LLMUsageMetricsData):
                self._diary.cost(service, item.value.prompt_tokens, "input tokens")
                self._diary.cost(service, item.value.completion_tokens, "output tokens")
            elif isinstance(item, TTSUsageMetricsData):
                self._diary.cost(service, item.value, "characters")
