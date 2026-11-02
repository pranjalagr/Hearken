"""The spelling fix before the voice: Hindi "main" becomes "mai", and nothing
else changes (2 Nov 2026: the voice reads "main" like "mayy")."""

import pytest

from agents.agent_c.talker.voice_spelling import spell_for_voice


@pytest.mark.parametrize("written, spoken", [
    ("Main ek automated assistant hoon.", "Mai ek automated assistant hoon."),
    ("Rider ki live location main nahi dekh sakta.", "Rider ki live location mai nahi dekh sakta."),
    ("Ek second, main check kar raha hoon.", "Ek second, mai check kar raha hoon."),
    # A capital next word is still the Hindi word: only place words keep "Main".
    ("Kya main UPI par bhej doon?", "Kya mai UPI par bhej doon?"),
    ("Main aur aap.", "Mai aur aap."),
    ("Kya main?", "Kya mai?"),
    ("main main", "mai mai"),
])
def test_the_hindi_word_for_i_is_respelled_in_hinglish(written, spoken):
    assert spell_for_voice(written, "hinglish") == spoken


@pytest.mark.parametrize("text", [
    "Shop 4, Main Market, Lajpat Nagar, Delhi.",
    "main market ke paas hai.",
    "Main  Road par hai.",
    # Only the whole word: these are other words that contain "main".
    "Maine order dekha.",
    "Remain calm.",
    "Kitchen mein khana ban raha hai.",
    "Ho gaya.",
    "",
])
def test_everything_else_in_hinglish_stays_as_written(text):
    assert spell_for_voice(text, "hinglish") == text


def test_a_place_word_only_protects_as_a_whole_word():
    assert spell_for_voice("Main Marketing dekh raha hoon.", "hinglish") == "Mai Marketing dekh raha hoon."


def test_an_english_reply_is_never_changed():
    assert spell_for_voice("The main reason is the delay.", "english") == "The main reason is the delay."
