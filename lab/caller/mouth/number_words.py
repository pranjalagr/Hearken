"""How the pretend caller says numbers (Part 3, C8).

The caller's brain writes plain digits ("order 4530", "₹40"); this turns them
into words in the style the task picks, just before the voice. Two reasons:
- controlled variety: we choose the style, so we can see which styles break
  the robot's ears;
- the truth: we know exactly which words were said.

Written separately from the robot's own number reader on purpose (C8): if
both used one shared table with a mistake in it, the caller and the robot
would be wrong the same way, and no test would ever notice. This file must
never be imported by robot code, and this table is checked by ear.

Spelling: English letters only (decision 12); long vowels doubled ("ee",
"aa"), so a voice reads "tees" (30) and "teyees" (23) differently.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from engine.tasks.caller_spec import NUMBER_STYLES

# 0..99. Hindi has its own word for every number below 100.
HINDI_0_TO_99 = (
    "shoonya", "ek", "do", "teen", "chaar", "paanch", "chhah", "saat", "aath", "nau",
    "das", "gyaarah", "baarah", "terah", "chaudah", "pandrah", "solah", "satrah", "athaarah", "unnees",
    "bees", "ikkees", "baaees", "teyees", "chaubees", "pachchees", "chhabbees", "sattaees", "atthaees", "untees",
    "tees", "iktees", "battees", "taintees", "chauntees", "paintees", "chhattees", "saintees", "adtees", "untaalees",
    "chaalees", "iktaalees", "bayaalees", "taintaalees", "chavaalees", "paintaalees", "chhiyaalees", "saintaalees",
    "adtaalees", "unchaas",
    "pachaas", "ikyaavan", "baavan", "tirepan", "chauvan", "pachpan", "chhappan", "sattaavan", "atthaavan", "unsath",
    "saath", "iksath", "baasath", "tirsath", "chaunsath", "painsath", "chhiyaasath", "sadsath", "adsath", "unhattar",
    "sattar", "ikhattar", "bahattar", "tihattar", "chauhattar", "pachhattar", "chhihattar", "sathattar", "athhattar",
    "unyaasi",
    "assi", "ikyaasi", "bayaasi", "tiraasi", "chauraasi", "pachaasi", "chhiyaasi", "sattaasi", "atthaasi", "navaasi",
    "nabbe", "ikyaanve", "baanve", "tiraanve", "chauraanve", "pachaanve", "chhiyaanve", "sattaanve", "atthaanve",
    "ninyaanve",
)
ENGLISH_DIGITS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")

# The Indian way of grouping: hazaar, lakh, crore (not "million").
_SAU, _HAZAAR, _LAKH, _CRORE = 100, 1_000, 100_000, 10_000_000

# An ID: 4 or more digits standing alone (order numbers are 4 digits).
# Shorter numbers and digits inside words ("Flat 2C", "12B") are left for the
# voice to read; they are not what C8 tests.
_ID = re.compile(r"(?<![0-9A-Za-z₹.,])[0-9]{4,}(?![0-9A-Za-z])")
# Money the brain writes with a rupee sign: ₹40, ₹ 450, ₹1,250.50
_MONEY = re.compile(r"₹\s?([0-9][0-9,]*(?:\.[0-9]{1,2})?)(?![0-9])")


def hindi_number(n: int) -> str:
    """A whole number in Hindi words, the Indian way: 1250 -> "ek hazaar do sau pachaas"."""
    if n < 0:
        raise ValueError("no negative numbers on a phone call")
    if n < 100:
        return HINDI_0_TO_99[n]
    words: list[str] = []
    if n >= _CRORE:
        words += [hindi_number(n // _CRORE), "crore"]
        n %= _CRORE
    for size, name in ((_LAKH, "lakh"), (_HAZAAR, "hazaar")):
        if n >= size:
            words += [HINDI_0_TO_99[n // size], name]
            n %= size
    if n >= _SAU:
        words += [HINDI_0_TO_99[n // _SAU], "sau"]
        n %= _SAU
    if n:
        words.append(HINDI_0_TO_99[n])
    return " ".join(words)


def say_digits(digits: str, style: str) -> str:
    """An ID (like an order number) in the given style.

    hindi_digits:   4530 -> "chaar paanch teen shoonya"
    english_digits: 4530 -> "four five three zero"
    hindi_pairs:    4530 -> "paintaalees tees"; an odd length starts with one
                    digit (12345 -> "ek teyees paintaalees"), and a pair that
                    starts with 0 is said digit by digit ("05" -> "shoonya paanch").
    """
    if not digits.isdigit() or not digits.isascii():
        raise ValueError(f"not digits: {digits!r}")
    if style not in NUMBER_STYLES:
        raise ValueError(f"unknown number style {style!r}")
    if style == "hindi_digits":
        return " ".join(HINDI_0_TO_99[int(d)] for d in digits)
    if style == "english_digits":
        return " ".join(ENGLISH_DIGITS[int(d)] for d in digits)
    groups = [digits[0]] if len(digits) % 2 else []
    rest = digits[len(groups) :]
    groups += [rest[i : i + 2] for i in range(0, len(rest), 2)]
    words = []
    for group in groups:
        if len(group) == 2 and group[0] == "0":
            words += [HINDI_0_TO_99[int(d)] for d in group]
        else:
            words.append(HINDI_0_TO_99[int(group)])
    return " ".join(words)


def say_rupees(amount: str) -> str:
    """Money in Hindi words: "450" -> "chaar sau pachaas rupaye",
    "249.50" -> "do sau unchaas rupaye pachaas paise". Commas are allowed."""
    try:
        value = Decimal(amount.replace(",", ""))
    except InvalidOperation:
        raise ValueError(f"not an amount: {amount!r}") from None
    if value < 0 or value != value.quantize(Decimal("0.01")):
        raise ValueError(f"not an amount in rupees and paise: {amount!r}")
    rupees, paise = divmod(int(value * 100), 100)
    parts = []
    if rupees:
        parts.append(f"{hindi_number(rupees)} {'rupaya' if rupees == 1 else 'rupaye'}")
    if paise:
        parts.append(f"{hindi_number(paise)} {'paisa' if paise == 1 else 'paise'}")
    return " ".join(parts) or "shoonya rupaye"


def speak_numbers(text: str, style: str) -> str:
    """Turn a caller's line into what the voice should read: money in Hindi
    words, IDs in the task's style, everything else untouched."""
    if style not in NUMBER_STYLES:
        raise ValueError(f"unknown number style {style!r}")
    text = _MONEY.sub(lambda m: say_rupees(m.group(1)), text)
    return _ID.sub(lambda m: say_digits(m.group(0), style), text)
