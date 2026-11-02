"""Spellings changed just before the voice (Part 2, decision 12), for words the
chosen voice (Sarvam sunny_enhi_customer) reads wrong.

The Talker writes "main" (the Hindi word for I), and the voice often reads it
like "mayy". Judged by ear on 2 Nov 2026, "mai" was right on 25 of 26 clips.

This is code, not a rule in the Talker's style guide, because that rule was
tried first and did harm: the refusal that must name the amount fell from 13 to
4 of 15, and one reply still wrote "main". Changing the spelling here can't
affect what the Talker decides to say.
"""

from __future__ import annotations

import re

# "Main" is kept before these, because there it is the English word in an
# address, as in the seed's "Main Market". A next-word-is-capital rule can't do
# this job: "Kya main UPI par bhej doon?" has a capital next word too. A place
# word missing from this list only costs an odd-sounding address.
PLACE_WORDS = ("market", "road", "gate", "bazaar", "bazar", "street", "chowk", "branch", "office")

_HINDI_MAIN = re.compile(rf"\b([Mm])ain\b(?!\s+(?i:{'|'.join(PLACE_WORDS)})\b)")


def spell_for_voice(text: str, language: str) -> str:
    """The reply as the voice should get it. Hinglish only: in an English reply
    "main" is the English word ("the main reason")."""
    if language != "hinglish":
        return text
    return _HINDI_MAIN.sub(r"\1ai", text)
