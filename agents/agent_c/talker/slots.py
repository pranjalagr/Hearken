"""The Talker's amount slots, and its number check (Part 2 decisions 8 and 10).

The Talker (a small AI) never writes an amount. Its notes show each amount as
a slot, like [amount]; it copies the slot into its reply where the amount goes,
and code fills the slot with the backend's exact words just before the voice:

    Talker writes:  "Ho gaya, [amount] aapke UPI par bhej diye gaye hain."
    code fills:     "Ho gaya, 62 rupaye 50 paise aapke UPI par bhej diye gaye hain."

Why: in the money check (1 Nov 2026) the Talker writing amounts itself scored
16 to 22 of 24 as its rule was tightened, and once said "5 paise" for ₹0.50;
wherever it only had to repeat what the notes said, it was right every time.

The number check is decision 8's net under that: a number the Talker typed
itself, found in neither its notes nor the caller's words, must never reach
the caller (a safe line plays instead).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

from engine.contract.action import Phrase
from engine.contract.spoken import SpokenWords

# Anything in square brackets is a slot: brackets never belong in speech, so
# "[Amount]" or "[refund]" is a slot the notes never gave, not text to say.
SLOT = re.compile(r"\[([^\[\]]{1,40})\]")
_NUMBER = re.compile(r"\d+")


class UnknownSlot(KeyError):
    """The reply names a slot no note gave: a made-up amount, never to be spoken."""


def used_slots(reply: str) -> set[str]:
    return set(SLOT.findall(reply))


def fill_slots(reply: str, slots: Mapping[str, Phrase], words: SpokenWords, language: str) -> str:
    """The reply with every slot put into words, in the reply's language: the
    same words, from the same pack file, as the yes-questions use."""

    def fill(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in slots:
            raise UnknownSlot(name)
        return words.say(slots[name], language)

    return SLOT.sub(fill, reply)


def stray_numbers(reply: str, known: Iterable[str]) -> list[str]:
    """Numbers the Talker typed itself that appear in nothing it was given
    (its notes and the caller's words). Slots are taken out first: their
    numbers come from code. "05" is not "5": a voice could say "zero five"."""
    allowed = {number for text in known for number in _NUMBER.findall(text)}
    return [number for number in _NUMBER.findall(SLOT.sub(" ", reply)) if number not in allowed]
