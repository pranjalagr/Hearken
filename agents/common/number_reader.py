"""The robot's number reader: the caller's spoken order number, as digits (Part 2,
decisions 25 and 27, moved earlier on 7 Nov 2026).

The Talker's AI points at the words where the caller said the number; this code turns
those words into digits and checks them. Each side does what it's good at: the AI knows
that the "do" in "cancel kar do" isn't a 2, and code never miscounts digits. When this
says anything but "ok", code asks again with a fixed line: it never guesses.

Kept apart from the pretend caller's number words (lab/caller/mouth/number_words.py),
with its own word table, so one shared mistake can't hide itself (decision C8).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from rapidfuzz.distance import Levenshtein

OK, INCOMPLETE, TOO_LONG, UNREADABLE, NONE = "ok", "incomplete", "too_long", "unreadable", "none"

# Hindi 0..99, in the spellings speech-to-text writes. Several spellings per number:
# the ears wrote "ikkis" and "paintalis" where a dictionary has "ikkees" and
# "paintaalees" (Nov 2026). Spellings are loosened further by _loose.
_HINDI = {
    0: "shunya shoonya sunya", 1: "ek", 2: "do", 3: "teen tin", 4: "chaar char", 5: "paanch panch",
    6: "chhah chhe chhah chah che chhai", 7: "saat sat", 8: "aath", 9: "nau",
    10: "das", 11: "gyaarah gyarah gyara", 12: "baarah barah bara", 13: "terah tera", 14: "chaudah chauda",
    15: "pandrah pandra", 16: "solah sola", 17: "satrah satra", 18: "athaarah atharah athara", 19: "unnees unnis",
    20: "bees bis", 21: "ikkees ikkis", 22: "baaees bais baais", 23: "teyees teis tais", 24: "chaubees chaubis",
    25: "pachchees pachchis pachis pacchis", 26: "chhabbees chhabbis", 27: "sattaees sattais", 28: "atthaees atthais athais",
    29: "untees untis", 30: "tees tis", 31: "iktees iktis ikatis", 32: "battees battis", 33: "taintees taintis tetis",
    34: "chauntees chauntis chautis", 35: "paintees paintis pentis", 36: "chhattees chhattis", 37: "saintees saintis",
    38: "adtees adtis artis", 39: "untaalees untalis", 40: "chaalees chalis chaalis", 41: "iktaalees iktalis",
    42: "bayaalees bayalis", 43: "taintaalees taintalis tetalis", 44: "chavaalees chawalis chauvalis",
    45: "paintaalees paintalis paintaalis", 46: "chhiyaalees chhiyalis", 47: "saintaalees saintalis",
    48: "adtaalees adtalis artalis", 49: "unchaas unchas", 50: "pachaas pachas", 51: "ikyaavan ikyavan",
    52: "baavan bavan", 53: "tirepan tirpan", 54: "chauvan chauwan", 55: "pachpan", 56: "chhappan",
    57: "sattaavan sattavan", 58: "atthaavan athavan", 59: "unsath", 60: "saath", 61: "iksath",
    62: "baasath basath", 63: "tirsath", 64: "chaunsath", 65: "painsath", 66: "chhiyaasath chhiyasath",
    67: "sadsath sarsath", 68: "adsath arsath", 69: "unhattar", 70: "sattar", 71: "ikhattar", 72: "bahattar",
    73: "tihattar", 74: "chauhattar", 75: "pachhattar", 76: "chhihattar", 77: "sathattar satattar",
    78: "athhattar athattar", 79: "unyaasi unasi", 80: "assi", 81: "ikyaasi ikyasi", 82: "bayaasi",
    83: "tiraasi", 84: "chauraasi", 85: "pachaasi", 86: "chhiyaasi", 87: "sattaasi", 88: "atthaasi",
    89: "navaasi nawasi", 90: "nabbe", 91: "ikyaanve", 92: "baanve", 93: "tiraanve", 94: "chauraanve",
    95: "pachaanve", 96: "chhiyaanve", 97: "sattaanve", 98: "atthaanve", 99: "ninyaanve ninyanve",
}
# Hindi letters for 0..9 only: our ears write English letters (decision 24), and other
# services wrote these (Nov 2026). A Hindi-letter pair is left unread: asked again.
_HINDI_LETTERS = {0: "शून्य ज़ीरो जीरो", 1: "एक", 2: "दो", 3: "तीन", 4: "चार", 5: "पांच पाँच", 6: "छह छः",
                  7: "सात", 8: "आठ", 9: "नौ"}
_ENGLISH = {0: "zero oh", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
            8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
            15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen"}
_ENGLISH_TENS = {20: "twenty", 30: "thirty", 40: "forty", 50: "fifty", 60: "sixty", 70: "seventy",
                 80: "eighty", 90: "ninety"}
# "so" and "aat" (8) were left out on purpose: loosened they're the English "so" and "at",
# far too common on a call to be read as numbers (7 Nov 2026).
_HUNDRED = "sau hundred सौ"
_THOUSAND = "hazaar hazar hajar thousand हज़ार हजार"
_REPEAT = {"double": 2, "triple": 3, "dabal": 2}
# Sounds people make while thinking: skipped inside a number.
_FILLERS = "a aa ah ahh aah um umm hmm hm uh er".split()
# A correction inside the picked words means the AI picked the wrong stretch.
_CORRECTIONS = "nahi nahin nai galat sorry matlab नहीं".split()
# Words that may sit at the edges of the picked words, harmlessly: "order 4521 hai".
_FRAME = "order number no nambar mera meri ka ki hai is my its it's".split()


def _loose(word: str) -> str:
    """One spelling for the many ways ears write a word: long vowels short."""
    word = word.lower()
    for long, short in (("aa", "a"), ("ee", "i"), ("oo", "u"), ("ii", "i"), ("uu", "u")):
        word = word.replace(long, short)
    return word


def _table() -> dict[str, tuple[str, int]]:
    words: dict[str, tuple[str, int]] = {}
    for table in (_HINDI, _HINDI_LETTERS, _ENGLISH):
        for value, spellings in table.items():
            for spelling in spellings.split():
                words[_loose(spelling)] = ("digit" if value < 10 else "pair", value)
    for value, spelling in _ENGLISH_TENS.items():
        words[_loose(spelling)] = ("tens", value)
    for spelling in _HUNDRED.split():
        words[_loose(spelling)] = ("scale", 100)
    for spelling in _THOUSAND.split():
        words[_loose(spelling)] = ("scale", 1000)
    for spelling, times in _REPEAT.items():
        words[_loose(spelling)] = ("repeat", times)
    return words


_WORDS = _table()
# Misspellings (7 Nov 2026, tested on 1,443 labelled words, make_projects/fuzzy_results.txt):
# a word not in the table is read as the closest number word only if it is at least this
# alike and this far ahead of the next different number. Looser limits matched real words
# ("dobara" became 12). A wrong match can become a wrong order number and a refused word
# only costs asking again, so these limits favour refusing.
_CLOSE_ENOUGH = 85.0
_CLEAR_LEAD = 12.0
# Words the table's lists already say what to do with, so they are never "misspellings".
_NEVER_MISSPELLINGS = set(_FILLERS + _CORRECTIONS + _FRAME)
# (spelling, which number it is.) Two spellings of one number are no tie, so spellings are
# compared by number. "double" isn't a number by itself, and Hindi letters don't misspell
# the way English-letter text does; neither was in the test that chose these limits.
_SPELLINGS: list[tuple[str, Any]] = [(spelling, meaning if meaning[0] == "scale" else meaning[1])
                                     for spelling, meaning in _WORDS.items()
                                     if meaning[0] != "repeat" and spelling.isascii()]
_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
# Written numbers, and words (Hindi letters kept whole: their vowel signs aren't "letters"
# to a regex). "45:30" and "4-5-2-1" are split, and punctuation dropped.
_TOKEN = re.compile(r"[0-9]+|[^\s0-9,.;:!?()\[\]\-/\"“”।|]+")


@dataclass(frozen=True)
class Reading:
    digits: str | None  # only when status is "ok"
    status: str
    why: str = ""


def words_of(text: str) -> list[str]:
    """The text as loose words, the same way for the picked words and what was heard."""
    return [_loose(token) for token in _TOKEN.findall(text.translate(_DEVANAGARI_DIGITS))]


def read_number(picked: str, heard: str, length: int) -> Reading:
    """Check the AI's picked words against what was heard, then read them as digits.
    `length` is the pack's rule for its numbers (4 for food orders)."""
    picked_words, heard_words = words_of(picked), words_of(heard)
    if not picked_words:
        return Reading(None, NONE, "no number was picked")
    # 1. The words are really there, in that order: the AI can't make them up. It leaves
    # out what the caller said in between ("tiraasi ji sadsath"), the user's call, 8 Nov 2026.
    if not _said_in_order(heard_words, picked_words):
        return Reading(None, NONE, f"the picked words aren't in what was heard, with only non-numbers between them: {picked!r}")
    words = [_number_word(word) or word for word in _trim_frame(picked_words)]
    # 3 before 2: a correction is also an unknown word, but saying so is clearer.
    if any(word in _CORRECTIONS for word in words):
        return Reading(None, UNREADABLE, "a correction inside the picked words")
    # 2. Every word can be read; an unknown one is never guessed.
    unknown = [word for word in words if word not in _WORDS and not word.isdigit() and word not in _FILLERS]
    if unknown:
        return Reading(None, UNREADABLE, f"words that aren't numbers: {unknown}")
    items = [word if word.isdigit() else _WORDS[word] for word in words if word not in _FILLERS]
    if not items:
        return Reading(None, NONE, "only fillers")
    # Every way to read the words. Two right-length answers can't come from these three
    # (merging "forty five" always drops a digit), so a check for that was left out: it
    # could never fire. It comes back if a word ever gets two meanings.
    readings = {reading for reading in (_concatenated(items, merge_tens=False), _concatenated(items, merge_tens=True),
                                        _spoken_value(items)) if reading is not None}
    if not readings:
        return Reading(None, UNREADABLE, "the words don't make one number")
    right = [reading for reading in readings if len(reading) == length]
    if right:
        return Reading(right[0], OK)
    # 5. The right length.
    longest = max(readings, key=len)
    if len(longest) < length:
        return Reading(None, INCOMPLETE, f"only {len(longest)} digits: {longest}")
    return Reading(None, TOO_LONG, f"{len(min(readings, key=len))} digits: {sorted(readings)}")


def _number_word(word: str) -> str | None:
    """The table's spelling for this word, or None if it isn't clearly one number word."""
    if word in _WORDS:
        return word
    if word in _NEVER_MISSPELLINGS or word.isdigit() or not word.isascii():
        return None
    best: dict[Any, tuple[float, str]] = {}
    for spelling, number in _SPELLINGS:
        score = Levenshtein.normalized_similarity(word, spelling) * 100
        if score > best.get(number, (-1.0, ""))[0]:
            best[number] = (score, spelling)
    (top, spelling), (second, _) = sorted(best.values(), reverse=True)[:2]
    return spelling if top >= _CLOSE_ENOUGH and top - second >= _CLEAR_LEAD else None


def _said_in_order(heard: list[str], picked: list[str]) -> bool:
    """The picked words appear in this order from some starting point, and nothing
    left out between them could change the number."""
    return any(_matches_from(heard, start + 1, picked[1:]) for start, word in enumerate(heard) if word == picked[0])


def _matches_from(heard: list[str], place: int, picked: list[str]) -> bool:
    """Tries both ways when a word could be the next one or left out: in "do minute do ek"
    the first "do" is the wait and the second is the digit."""
    if not picked:
        return True
    for here in range(place, len(heard)):
        if heard[here] == picked[0] and _matches_from(heard, here + 1, picked[1:]):
            return True
        if _changes_the_number(heard, here):
            return False
    return False


def _changes_the_number(heard: list[str], place: int) -> bool:
    """A left-out word that can't be dropped safely: a correction ("chaar paanch nahi do
    ek"), or a real part of a number. A spoken number followed by a non-number is let
    through, because "ek sec" and "do minute" are how callers say wait."""
    word = heard[place]
    if word in _CORRECTIONS or word.isdigit():
        return True
    if _number_word(word) is None:
        return False
    following = heard[place + 1] if place + 1 < len(heard) else ""
    return following.isdigit() or _number_word(following) is not None


def _trim_frame(words: list[str]) -> list[str]:
    start, end = 0, len(words)
    while start < end and words[start] in _FRAME and words[start] not in _WORDS:
        start += 1
    while end > start and words[end - 1] in _FRAME and words[end - 1] not in _WORDS:
        end -= 1
    return words[start:end]


def _concatenated(items: list, merge_tens: bool) -> str | None:
    """Digits as said, one after another: "chaar paanch do ek", "paintaalis ikkees",
    "45 30". With merge_tens, English "forty five" is 45, as English speakers mean."""
    digits: list[str] = []
    repeat = 1
    for n, item in enumerate(items):
        if isinstance(item, str):
            digits.append(item * repeat)
            repeat = 1
            continue
        kind, value = item
        if kind == "scale":
            return None  # "sau" and "hazaar" make a value, not a string of digits
        if kind == "repeat":
            repeat = value
            continue
        if kind == "tens" and merge_tens and n + 1 < len(items) and not isinstance(items[n + 1], str) \
                and items[n + 1][0] == "digit" and items[n + 1][1] > 0:
            continue  # joined to the digit after it, below
        if kind == "digit" and merge_tens and n > 0 and not isinstance(items[n - 1], str) \
                and items[n - 1][0] == "tens" and value > 0:
            digits.append(str(items[n - 1][1] + value) * repeat)
        else:
            digits.append(str(value) * repeat)
        repeat = 1
    return "".join(digits) if digits and repeat == 1 else None


def _spoken_value(items: list) -> str | None:
    """One number said as a value: "chaar hazaar paanch sau ikkees" is 4521."""
    if not any(not isinstance(item, str) and item[0] == "scale" for item in items):
        return None
    total, part = 0, None
    for item in items:
        if isinstance(item, str) or item[0] in ("repeat", "tens"):
            return None
        kind, value = item
        if kind == "scale":
            if part is None:
                return None
            if value == 1000:
                total, part = total + part * 1000, None
            elif part >= 10:
                return None  # "bees sau" isn't how an order number is said
            else:
                total, part = total + part * 100, None
        elif part is not None:
            return None  # two numbers side by side: not one value
        else:
            part = value
    return str(total + (part or 0))
