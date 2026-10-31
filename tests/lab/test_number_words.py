"""The pretend caller's number words (Part 3, C8)."""

import re
from pathlib import Path

import pytest

from lab.caller.mouth import number_words
from lab.caller.mouth.number_words import (
    ENGLISH_DIGITS,
    HINDI_0_TO_99,
    hindi_number,
    say_digits,
    say_rupees,
    speak_numbers,
)


def test_the_table_has_one_word_for_every_number_below_100():
    assert len(HINDI_0_TO_99) == 100
    assert len(set(HINDI_0_TO_99)) == 100  # no two numbers share a word
    assert all(re.fullmatch(r"[a-z]+", word) for word in HINDI_0_TO_99)  # English letters only
    assert len(ENGLISH_DIGITS) == 10


@pytest.mark.parametrize(
    ("n", "word"),
    [
        # The ones people (and tables) most often get wrong: the "un-" numbers
        # are one less than the next ten, and 23 must not sound like 30.
        (19, "unnees"), (29, "untees"), (39, "untaalees"), (49, "unchaas"), (59, "unsath"),
        (69, "unhattar"), (79, "unyaasi"), (89, "navaasi"), (99, "ninyaanve"),
        (23, "teyees"), (30, "tees"), (45, "paintaalees"), (21, "ikkees"), (12, "baarah"),
        (38, "adtees"), (48, "adtaalees"), (68, "adsath"), (67, "sadsath"), (28, "atthaees"),
        (0, "shoonya"), (6, "chhah"), (50, "pachaas"), (100 - 1, "ninyaanve"),
    ],
)
def test_tricky_hindi_numbers(n, word):
    assert HINDI_0_TO_99[n] == word


@pytest.mark.parametrize(
    ("n", "words"),
    [
        (7, "saat"),
        (100, "ek sau"),
        (105, "ek sau paanch"),
        (450, "chaar sau pachaas"),
        (999, "nau sau ninyaanve"),
        (1000, "ek hazaar"),
        (1250, "ek hazaar do sau pachaas"),
        (10_000, "das hazaar"),
        (99_999, "ninyaanve hazaar nau sau ninyaanve"),
        (1_00_000, "ek lakh"),
        (1_50_000, "ek lakh pachaas hazaar"),
        (1_23_45_678, "ek crore teyees lakh paintaalees hazaar chhah sau athhattar"),
    ],
)
def test_hindi_numbers_group_the_indian_way(n, words):
    assert hindi_number(n) == words


@pytest.mark.parametrize(
    ("digits", "style", "words"),
    [
        ("4530", "hindi_digits", "chaar paanch teen shoonya"),
        ("4530", "english_digits", "four five three zero"),
        ("4530", "hindi_pairs", "paintaalees tees"),
        ("4521", "hindi_pairs", "paintaalees ikkees"),
        ("4505", "hindi_pairs", "paintaalees shoonya paanch"),
        ("4500", "hindi_pairs", "paintaalees shoonya shoonya"),
        ("12345", "hindi_pairs", "ek teyees paintaalees"),
        ("0450", "hindi_digits", "shoonya chaar paanch shoonya"),
    ],
)
def test_ids_in_each_style(digits, style, words):
    assert say_digits(digits, style) == words


@pytest.mark.parametrize(
    ("amount", "words"),
    [
        ("450", "chaar sau pachaas rupaye"),
        ("1", "ek rupaya"),
        ("40", "chaalees rupaye"),
        ("249.50", "do sau unchaas rupaye pachaas paise"),
        ("0.50", "pachaas paise"),
        ("1.01", "ek rupaya ek paisa"),
        ("1,250", "ek hazaar do sau pachaas rupaye"),
    ],
)
def test_money_in_hindi_words(amount, words):
    assert say_rupees(amount) == words


@pytest.mark.parametrize("bad", ["abc", "1.234", "-5"])
def test_a_bad_amount_is_refused(bad):
    with pytest.raises(ValueError):
        say_rupees(bad)


@pytest.mark.parametrize(
    ("line", "style", "spoken"),
    [
        ("Mera order 4530 hai", "hindi_pairs", "Mera order paintaalees tees hai"),
        ("₹40 wapas chahiye", "english_digits", "chaalees rupaye wapas chahiye"),
        (
            "4512... nahi nahi, 4521",
            "hindi_pairs",
            "paintaalees baarah... nahi nahi, paintaalees ikkees",
        ),
        ("Total ₹1,250.50 tha", "hindi_digits", "Total ek hazaar do sau pachaas rupaye pachaas paise tha"),
        ("Flat 2C, 12B, Sector 21", "hindi_pairs", "Flat 2C, 12B, Sector 21"),  # not IDs: left for the voice
        ("Order 4521, ₹450", "hindi_digits", "Order chaar paanch do ek, chaar sau pachaas rupaye"),
    ],
)
def test_a_callers_line_is_turned_into_words(line, style, spoken):
    assert speak_numbers(line, style) == spoken


def test_unknown_styles_and_non_digits_are_refused():
    with pytest.raises(ValueError):
        speak_numbers("4530", "roman")
    with pytest.raises(ValueError):
        say_digits("45a0", "hindi_digits")
    with pytest.raises(ValueError):
        hindi_number(-1)


def test_this_file_never_borrows_from_robot_code():
    """C8: kept separate from the robot's number reader, so one shared mistake
    can't hide itself from the tests."""
    source = Path(number_words.__file__).read_text(encoding="utf-8")
    assert "from agents" not in source and "import agents" not in source
