"""What becomes of a Talker reply before the voice (decisions 8, 10, 12, 18).

1. At most two sentences: code, not the prompt, keeps replies short.
2. The number check: a number the Talker typed itself, found in neither its notes nor
   the caller's words, or a slot no note gave, is never spoken: the safe line plays.
3. Slots are filled with the backend's values, in the call's one language.
4. Spellings the voice reads wrong are changed ("main" -> "mai").
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from agents.agent_c.talker.language import StickyLanguage
from agents.agent_c.talker.note_view import NoteView
from agents.agent_c.talker.slots import SLOT, stray_numbers, used_slots
from agents.agent_c.talker.voice_spelling import spell_for_voice
from engine.contract.spoken import SpokenWords

_SENTENCE_END = re.compile(r"(?<=[.?!।])\s+")


@dataclass(frozen=True)
class Speech:
    reply: str  # the Talker's words, cut to length, slots still in
    spoken: str  # what goes to the voice
    safe: bool  # True when the safe line replaced the reply
    why: str = ""


def first_sentences(text: str, most: int) -> str:
    return " ".join(_SENTENCE_END.split(text.strip())[:most])


class Speaker:
    def __init__(self, words: SpokenWords, view: NoteView, language: StickyLanguage, max_sentences: int) -> None:
        self._words = words
        self._view = view
        self._language = language
        self._max = max_sentences
        self._known: list[str] = []  # every note line given, and every caller turn

    def given(self, notes_line: str) -> None:
        self._known.append(notes_line)

    def heard(self, words: str) -> None:
        self._known.append(words)

    def safe_line(self) -> str:
        return self._words.phrase("line", "safe", self._language.value)

    def prepare(self, reply: str) -> Speech:
        language = self._language.value
        text = first_sentences(reply, self._max)
        if "notes:" in text.lower():
            # The Talker copied a NOTES line: never read the back office's notes to the caller.
            # 16 Nov 2026, food_03: "NOTES: refund missing item...; amount ₹380" was spoken, and
            # the number check let it through because the numbers were in the notes.
            return Speech(text, spell_for_voice(self.safe_line(), language), True, "a NOTES line, not a reply")
        stray = stray_numbers(text, self._known)
        unknown = sorted(used_slots(text) - set(self._view.slots))
        if stray or unknown:
            why = f"numbers no note gave: {stray}" if stray else f"slots no note gave: {unknown}"
            return Speech(text, spell_for_voice(self.safe_line(), language), True, why)

        def fill(match: re.Match[str]) -> str:
            value = self._view.slots[match.group(1)]
            return value if isinstance(value, str) else self._words.say(value, language)

        return Speech(text, spell_for_voice(SLOT.sub(fill, text), language), False)
