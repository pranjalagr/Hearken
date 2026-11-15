"""The robot's number reader on its own, with no AI (Part 2, decision 27).

Each case gets the right words straight from the test set, so a failure here is the
reader's own mistake, never the AI's (the user asked for exactly this, 7 Nov 2026).
"""

import pytest

from agents.common.number_reader import INCOMPLETE, NONE, OK, TOO_LONG, UNREADABLE, read_number
from lab.caller.mouth.number_words import hindi_number, say_digits
from lab.week1.number_cases import ASK, load_cases

CASES = load_cases()
LENGTH = 4


@pytest.mark.parametrize("case", CASES, ids=[case.id for case in CASES])
def test_the_reader_gets_every_case_right_when_given_the_right_words(case):
    reading = read_number(case.span, case.heard, LENGTH)
    if case.expect == ASK:
        assert reading.digits is None and reading.status != NONE or not case.span, (case.note, reading)
    elif case.expect == NONE:
        assert reading.status == NONE, reading
    else:
        assert (reading.status, reading.digits) == (OK, case.expect), (case.note, reading)


@pytest.mark.parametrize("style", ["hindi_digits", "english_digits", "hindi_pairs", "written", "written pairs"])
def test_every_4_digit_number_in_every_style(style):
    # 10,000 numbers each, spoken by the caller's separate word table: a gap in the
    # reader's own table can't hide behind a test written from that same table.
    wrong = []
    for value in range(10_000):
        number = f"{value:04d}"
        if style == "written":
            spoken = number
        elif style == "written pairs":
            spoken = f"{number[:2]} {number[2:]}"
        else:
            spoken = say_digits(number, style)
        reading = read_number(spoken, spoken, LENGTH)
        if reading.digits != number:
            wrong.append((spoken, reading.status, reading.digits))
    assert not wrong, f"{len(wrong)} misread, e.g. {wrong[:5]}"


def test_every_number_said_as_a_value():
    wrong = [(value, reading) for value in range(1000, 10_000)
             if (reading := read_number(hindi_number(value), hindi_number(value), LENGTH)).digits != str(value)]
    assert not wrong, f"{len(wrong)} misread, e.g. {wrong[:5]}"


def test_picked_words_that_werent_heard_are_ignored():
    # The AI must point at what the ears wrote, never write a number itself.
    reading = read_number("chaar paanch do ek", "Order chaar paanch ek do hai", LENGTH)
    assert reading.status == NONE and "aren't in what was heard" in reading.why
    assert read_number("4521", "Order chaar paanch do ek hai", LENGTH).status == NONE  # digits it typed itself


def test_words_said_in_between_can_be_left_out():
    assert read_number("tiraasi sadsath", "Order tiraasi ji sadsath hai", 4).digits == "8367"
    assert read_number("paintaalees tees", "paintaalees ek sec tees", 4).digits == "4530"
    assert read_number("chaar paanch do ek", "chaar paanch ruko do minute do ek", 4).digits == "4521"
    # The start isn't the first place the word was said: "do baar" is not the number.
    assert read_number("do teen chaar paanch", "do baar bola do teen chaar paanch", 4).digits == "2345"


def test_a_clear_misspelling_is_read():
    assert read_number("chudah pndrah", "chudah pndrah", 4).digits == "1415"
    assert read_number("ek shuniya do teen", "ek shuniya do teen", 4).digits == "1023"


def test_a_word_that_is_only_nearly_a_number_is_refused_not_guessed():
    # Ordinary words, too short to be sure of, and a word close to two numbers (24 and 26).
    for words in ("dobara ek do teen", "hazari ek do teen", "fiv seven ate six", "chabis ek ek"):
        assert read_number(words, words, 4).status == UNREADABLE, words


def test_a_misspelled_digit_cant_be_left_out_of_the_middle():
    assert read_number("chaar paanch do ek", "chaar paanch shuniya do ek", 4).digits is None


def test_nothing_that_changes_the_number_can_be_left_out():
    for picked, heard in (("chaar paanch do ek", "chaar paanch nahi do ek"),  # a correction
                          ("chaar paanch do ek", "chaar paanch teen do ek"),  # a digit dropped
                          ("45 21", "45 ruko 33 21"),  # another number in between
                          ("chaar paanch do ek", "chaar do paanch ek")):  # out of order
        assert read_number(picked, heard, 4).digits is None, heard


def test_a_correction_inside_the_picked_words_is_never_read():
    reading = read_number("chaar paanch nahi do ek", "chaar paanch nahi do ek", LENGTH)
    assert reading.status == UNREADABLE and "correction" in reading.why  # says why: the AI picked the wrong stretch


def test_an_unknown_word_is_never_guessed():
    reading = read_number("chaar kaal do ek", "chaar kaal do ek", LENGTH)
    assert reading.status == UNREADABLE and "kal" in reading.why  # spellings are loosened: kaal is kal


def test_the_length_is_the_packs_and_short_or_long_is_asked_again():
    assert read_number("chaar paanch", "chaar paanch", LENGTH).status == INCOMPLETE
    assert read_number("chaar paanch do do ek", "chaar paanch do do ek", LENGTH).status == TOO_LONG
    assert read_number("chaar paanch do do ek", "chaar paanch do do ek", 5).digits == "45221"


def test_fillers_and_the_words_around_a_number_are_skipped():
    assert read_number("chaar paanch umm do ek", "chaar paanch umm do ek", LENGTH).digits == "4521"
    assert read_number("order 4521 hai", "mera order 4521 hai", LENGTH).digits == "4521"


def test_two_numbers_side_by_side_are_never_one_value():
    # "chaar hazaar paanch do" isn't 4052: a value said with hazaar has one word per place.
    assert read_number("chaar hazaar paanch do", "chaar hazaar paanch do", LENGTH).digits is None


def test_nothing_picked_is_no_number():
    assert read_number("", "haan bataiye", LENGTH).status == NONE
