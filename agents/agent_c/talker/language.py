"""The call's language, kept by plain code (decision 10, 15 Nov 2026).

Start in Hinglish. A finished caller turn changes it only if it has at least a few
words and the ears report another language: one "okay" or "thank you" in a Hinglish
call must not flip the robot into English. The slot filler and the voice read this
one value, so an amount and the words around it are always in the same language.
"""

from __future__ import annotations

from pipecat.frames.frames import TranscriptionFrame
from pipecat.observers.base_observer import BaseObserver, FramePushed

HINGLISH = "hinglish"
ENGLISH = "english"


def from_ears(code: str | None) -> str | None:
    """The ears' language code ("en-IN", "hi-IN", Language.EN_IN) as ours; None if neither."""
    text = str(code or "").lower().replace("language.", "")
    if text.startswith("en"):
        return ENGLISH
    if text.startswith("hi"):
        return HINGLISH
    return None


class StickyLanguage:
    def __init__(self, min_words: int, start: str = HINGLISH) -> None:
        self._min_words = min_words
        self.value = start
        self.ears_said: str | None = None  # the language of the newest words the ears gave

    def turn_finished(self, words: str) -> str:
        heard = from_ears(self.ears_said)
        if heard is not None and heard != self.value and len(words.split()) >= self._min_words:
            self.value = heard
        return self.value


class EarsLanguage(BaseObserver):
    """Notes the language the ears reported for each piece of the caller's words."""

    def __init__(self, language: StickyLanguage) -> None:
        super().__init__()
        self._language = language

    async def on_push_frame(self, data: FramePushed) -> None:
        if isinstance(data.frame, TranscriptionFrame) and data.frame.language is not None:
            self._language.ears_said = str(data.frame.language)
