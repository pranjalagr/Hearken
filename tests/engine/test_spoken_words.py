"""Spoken words: the engine's loader, its load-time checks, and yes-questions."""

import re

import pytest

from engine.contract.action import Phrase, SpokenQuestion
from engine.contract.spoken import SpokenWords
from packs.toy.pack import ToyPack

GOOD = """
status:
  ready:
    hinglish: "Taiyaar hai."
    english: "It's ready."
"""


def write(tmp_path, text):
    path = tmp_path / "spoken.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_reads_a_phrase_in_each_language(tmp_path):
    words = SpokenWords.from_yaml(write(tmp_path, GOOD))
    assert words.phrase("status", "ready", "hinglish") == "Taiyaar hai."
    assert words.phrase("status", "ready", "english") == "It's ready."
    assert words.values("status") == {"ready"}


def test_a_missing_phrase_fails_loudly(tmp_path):
    words = SpokenWords.from_yaml(write(tmp_path, GOOD))
    with pytest.raises(KeyError, match="status.gone"):
        words.phrase("status", "gone", "hinglish")


def test_an_unknown_language_is_refused(tmp_path):
    words = SpokenWords.from_yaml(write(tmp_path, GOOD))
    with pytest.raises(ValueError, match="unknown language"):
        words.phrase("status", "ready", "tamil")


def test_a_missing_language_fails_when_the_file_loads(tmp_path):
    text = 'status:\n  ready:\n    hinglish: "Taiyaar hai."\n'
    with pytest.raises(ValueError, match="status.ready needs exactly"):
        SpokenWords.from_yaml(write(tmp_path, text))


def test_hindi_script_fails_when_the_file_loads(tmp_path):
    """Part 2 decision 12: English letters everywhere."""
    text = 'status:\n  ready:\n    hinglish: "तैयार है"\n    english: "It\'s ready."\n'
    with pytest.raises(ValueError, match="Hindi script"):
        SpokenWords.from_yaml(write(tmp_path, text))


def test_a_pack_without_spoken_words_fails_loudly_when_asked():
    words = ToyPack().spoken_words()
    assert words.values("status") == set()
    with pytest.raises(KeyError):
        words.phrase("status", "anything", "english")


QUESTIONS = """
place:
  shelf:
    hinglish: "shelf"
    english: "the shelf"
yes_question:
  move:
    hinglish: "{thing} {place} par rakh doon? Haan ya nahi?"
    english: "Shall I put {thing} on {place}? Yes or no?"
"""


def test_a_yes_question_is_filled_with_values_and_phrases(tmp_path):
    words = SpokenWords.from_yaml(write(tmp_path, QUESTIONS))
    question = SpokenQuestion("move", values={"thing": "box 7"}, phrases={"place": Phrase("place", "shelf")})
    assert words.yes_question(question, "hinglish") == "box 7 shelf par rakh doon? Haan ya nahi?"
    assert words.yes_question(question, "english") == "Shall I put box 7 on the shelf? Yes or no?"


def test_braces_inside_a_value_are_kept_as_they_are(tmp_path):
    """A caller's address could contain braces; they must not be read as placeholders."""
    words = SpokenWords.from_yaml(write(tmp_path, QUESTIONS))
    question = SpokenQuestion("move", values={"thing": "Flat {5}"}, phrases={"place": Phrase("place", "shelf")})
    assert words.yes_question(question, "english") == "Shall I put Flat {5} on the shelf? Yes or no?"


def test_a_yes_question_with_a_missing_value_fails_loudly(tmp_path):
    words = SpokenWords.from_yaml(write(tmp_path, QUESTIONS))
    with pytest.raises(KeyError):
        words.yes_question(SpokenQuestion("move", phrases={"place": Phrase("place", "shelf")}), "english")


PRICED = """
price:
  with_paise:
    hinglish: "{rupees} rupaye {paise} paise"
    english: "{rupees} rupees {paise} paise"
yes_question:
  sell:
    hinglish: "{thing} {price} mein bech doon? Haan ya nahi?"
    english: "Shall I sell {thing} for {price}? Yes or no?"
"""


def test_a_phrase_says_its_own_numbers_in_the_callers_language(tmp_path):
    """Part 2 decision 10: money with paise needs words around its numbers,
    and the words differ by language."""
    words = SpokenWords.from_yaml(write(tmp_path, PRICED))
    price = Phrase("price", "with_paise", {"rupees": "249", "paise": "50"})
    question = SpokenQuestion("sell", values={"thing": "box 7"}, phrases={"price": price})
    assert words.yes_question(question, "hinglish") == "box 7 249 rupaye 50 paise mein bech doon? Haan ya nahi?"
    assert words.yes_question(question, "english") == "Shall I sell box 7 for 249 rupees 50 paise? Yes or no?"


def test_a_phrase_with_a_number_missing_fails_loudly(tmp_path):
    """Never a hole: "249 rupaye  paise" must not be spoken."""
    words = SpokenWords.from_yaml(write(tmp_path, PRICED))
    question = SpokenQuestion("sell", values={"thing": "box 7"},
                              phrases={"price": Phrase("price", "with_paise", {"rupees": "249"})})
    with pytest.raises(KeyError, match="paise"):
        words.yes_question(question, "hinglish")


def test_braces_inside_a_phrases_number_are_kept_as_they_are(tmp_path):
    words = SpokenWords.from_yaml(write(tmp_path, PRICED))
    price = Phrase("price", "with_paise", {"rupees": "{x}", "paise": "50"})
    question = SpokenQuestion("sell", values={"thing": "box"}, phrases={"price": price})
    assert words.yes_question(question, "english") == "Shall I sell box for {x} rupees 50 paise? Yes or no?"


@pytest.mark.parametrize(("hinglish", "english", "problem"), [
    ('"{rupees rupaye"', '"{rupees} rupees"', "broken {slot}"),
    ('"rupaye}"', '"rupees"', "broken {slot}"),
    ('"{rupees} rupaye {paise} paise"', '"{rupees} rupees"', "every language needs the same {slots}"),
])
def test_a_broken_or_lopsided_slot_fails_when_the_file_loads(tmp_path, hinglish, english, problem):
    """Found at load, not when a caller is waiting for the question."""
    text = f"price:\n  odd:\n    hinglish: {hinglish}\n    english: {english}\n"
    with pytest.raises(ValueError, match=re.escape(problem)):
        SpokenWords.from_yaml(write(tmp_path, text))


CURLY = """
mark:
  curly:
    hinglish: "{{ok}}"
    english: "{{ok}}"
yes_question:
  show:
    hinglish: "{mark} Haan ya nahi?"
    english: "{mark} Yes or no?"
"""


def test_a_doubled_brace_is_a_plain_brace_not_a_slot(tmp_path):
    words = SpokenWords.from_yaml(write(tmp_path, CURLY))
    question = SpokenQuestion("show", phrases={"mark": Phrase("mark", "curly")})
    assert words.yes_question(question, "english") == "{ok} Yes or no?"


def test_a_yes_question_must_end_with_the_closed_question(tmp_path):
    """The Guard always judges an answer to the same "yes or no"."""
    text = 'yes_question:\n  move:\n    hinglish: "Rakh doon?"\n    english: "Shall I? Yes or no?"\n'
    with pytest.raises(ValueError, match="must end with 'Haan ya nahi\\?'"):
        SpokenWords.from_yaml(write(tmp_path, text))


async def test_a_pack_without_questions_still_prepares_but_has_no_question(open_call):
    kit = await open_call(ToyPack())
    prepared = await kit.tool("prepare_move_money", from_box="a", to_box="b", amount_paise=500)
    assert prepared.status == "ok"
    with pytest.raises(LookupError, match="no spoken question"):
        await kit.guard.question(prepared.data["token"], "hinglish")


async def test_the_guard_refuses_an_unknown_token_or_language(open_call):
    kit = await open_call(ToyPack())
    with pytest.raises(LookupError, match="unknown token"):
        await kit.guard.question("T-99", "hinglish")
    with pytest.raises(ValueError, match="unknown language"):
        await kit.guard.question("T-1", "tamil")
