"""The truth sheet, and the number check on what the robot said (Part 5, L10).

The truth sheet is written by code from the call's final records and its
timeline: what really happened, for the judge (L11) and for this check.

The number check is exact, so it's code, not AI: every number the robot
said must be on the truth sheet or said by the caller. Numbers are read in
any form: digits ("4521"), money ("₹249.50", "249 rupaye 50 paise", compared
as one whole amount) and Hindi or English number words ("chaar sau pachaas",
"four hundred fifty"). Set phrases like "ek second" are not facts, so skipped.

What it can't catch is a right number used wrongly ("₹450 UPI par" when it
went to the card): that's meaning, the judge's part.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from lab.caller.mouth.number_words import HINDI_0_TO_99
from lab.scorer.call_record import CallRecord

# Not facts: "ek" here means "a moment", not the number one.
SET_PHRASES = ("ek second", "ek minute", "ek baar", "ek pal", "one moment", "one second", "one minute")

_HINDI = {word: n for n, word in enumerate(HINDI_0_TO_99)}
_HINDI_SCALES = {"sau": 100, "hazaar": 1_000, "hazar": 1_000, "lakh": 100_000, "crore": 10_000_000}
_ENGLISH = {word: n for n, word in enumerate((
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
    "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"))}
_ENGLISH.update({"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
                 "ninety": 90})
_ENGLISH_SCALES = {"hundred": 100, "thousand": 1_000, "lakh": 100_000}

_MONEY_SIGN = re.compile(r"₹\s?(\d[\d,]*)(?:\.(\d{1,2}))?")
_MONEY_WORDS = re.compile(r"(\d[\d,]*)\s*(?:rupaye|rupees|rupee|rs\.?)(?:\s*(?:aur|and))?\s*(\d{1,2})\s*paise")
_PAISE_ONLY = re.compile(r"(?<![\d,.])(\d{1,2})\s*paise")
_DIGITS = re.compile(r"\d[\d,]*(?:\.\d+)?")


@dataclass
class Spoken:
    """The numbers in one piece of text: money in paise, everything else as is."""

    money_paise: set[int] = field(default_factory=set)
    plain: set[int] = field(default_factory=set)


def read_numbers(text: str) -> Spoken:
    found = Spoken()
    lowered = " ".join(text.lower().split())
    for phrase in SET_PHRASES:
        lowered = lowered.replace(phrase, " ")
    for regex, to_paise in ((_MONEY_WORDS, lambda m: _int(m[1]) * 100 + int(m[2])),
                            (_MONEY_SIGN, lambda m: _int(m[1]) * 100 + int((m[2] or "0").ljust(2, "0")))):
        for match in regex.finditer(lowered):
            found.money_paise.add(to_paise(match))
        lowered = regex.sub(" ", lowered)
    for match in _PAISE_ONLY.finditer(lowered):
        found.money_paise.add(int(match[1]))
    lowered = _PAISE_ONLY.sub(" ", lowered)
    for match in _DIGITS.finditer(lowered):
        text_number = match[0].replace(",", "")
        if "." in text_number:
            # A decimal outside money: compare it as an amount.
            whole, frac = text_number.split(".", 1)
            found.money_paise.add(int(whole) * 100 + int(frac[:2].ljust(2, "0")))
        else:
            found.plain.add(int(text_number))
    found.plain |= _word_numbers(lowered)
    return found


def _int(digits: str) -> int:
    return int(digits.replace(",", ""))


_SCALES = {**_HINDI_SCALES, **_ENGLISH_SCALES}
# A lone number word counts only before one of these: "kar do" (do = 2) and
# "ek order" are ordinary Hinglish, but "paanch minute" is a promise.
_UNITS = {"rupaye", "rupees", "rupee", "paise", "minute", "minutes", "min", "second", "seconds", "din", "dino",
          "ghante", "ghanta", "hours", "hour", "days", "day"}


def _word_numbers(text: str) -> set[int]:
    """Runs of Hindi or English number words that are real amounts:
    "do sau unchaas" -> 249 (has a scale), "teen din" -> 3 (before a unit)."""
    numbers: set[int] = set()
    total = current = 0
    in_number = scaled = False
    for word in [*re.findall(r"[a-z]+", text.replace("-", " ")), ""]:
        if word in _HINDI or word in _ENGLISH:
            current += _HINDI.get(word, _ENGLISH.get(word, 0))
            in_number = True
        elif in_number and word in _SCALES:
            scaled = True
            if _SCALES[word] == 100:
                current = max(current, 1) * 100
            else:
                total += max(current, 1) * _SCALES[word]
                current = 0
        elif in_number and word == "and":
            continue  # "two hundred and fifty"
        else:
            if in_number and (scaled or word in _UNITS):
                numbers.add(total + current)
            total = current = 0
            in_number = scaled = False
    return numbers


@dataclass
class TruthSheet:
    lines: list[str]
    money_paise: set[int]
    plain: set[int]

    def text(self) -> str:
        return "\n".join(self.lines)


def truth_sheet(record: CallRecord, order_nos: list[int], done: list[Any]) -> TruthSheet:
    """What really happened, from the records: the orders in this call, their
    money, and every change with when it was sent."""
    lines: list[str] = []
    money: set[int] = set()
    plain: set[int] = set(order_nos)
    with record.final_db() as db:
        for order_no in order_nos:
            order = db.execute("SELECT * FROM orders WHERE order_no = ?", (order_no,)).fetchone()
            if order is None:
                continue
            items = db.execute("SELECT item, quantity, price_paise FROM order_items WHERE order_no = ?",
                               (order_no,)).fetchall()
            refunds = db.execute("SELECT amount_paise, method, reason FROM refunds WHERE order_no = ?",
                                 (order_no,)).fetchall()
            money.add(order["amount_paise"])
            refunded = sum(r["amount_paise"] for r in refunds)
            money.add(order["amount_paise"] - refunded)  # what's left to refund
            for item in items:
                money |= {item["price_paise"] * n for n in range(1, item["quantity"] + 1)}
                plain.add(item["quantity"])
            money |= {r["amount_paise"] for r in refunds}
            lines.append(
                f"order {order_no}: status {order['status']}, paid ₹{order['amount_paise'] / 100:g} by "
                f"{order['payment_method']}, address {order['address']!r}; items "
                + ", ".join(f"{i['quantity']} x {i['item']} @ ₹{i['price_paise'] / 100:g}" for i in items)
                + ("; refunds " + ", ".join(f"₹{r['amount_paise'] / 100:g} to {r['method']} ({r['reason']})"
                                            for r in refunds) if refunds else "; no refunds")
            )
    for change in done:
        when = f"{change.sent_ms / 1000:.2f} s" if change.sent_ms is not None else "?"
        lines.append(f"done: {change.action} {change.inputs} (sent at {when})")
    for at in record.stops():
        lines.append(f"the caller said stop at {at / 1000:.2f} s")
    # Whole rupees count as plain numbers too: "450 rupaye" is said as 450.
    plain |= {paise // 100 for paise in money if paise % 100 == 0}
    # Any number in an order's facts (an address's "Flat 2C", "Sector 15") is a
    # fact too; the timing lines ("sent at 17.72 s") are not things to say.
    facts = "\n".join(line for line in lines if line.startswith("order "))
    plain |= {int(n) for n in re.findall(r"\d+", facts)}
    return TruthSheet(lines, money, plain)


def made_up_numbers(record: CallRecord, sheet: TruthSheet) -> list[dict[str, Any]]:
    """Every robot line with a number found neither on the truth sheet nor in the caller's words."""
    caller = Spoken()
    for line in record.caller_lines():
        spoken = read_numbers(line["text"])
        caller.money_paise |= spoken.money_paise
        caller.plain |= spoken.plain
    for line in record.director_lines():
        spoken = read_numbers(line["line"])
        caller.money_paise |= spoken.money_paise
        caller.plain |= spoken.plain
    allowed_money = sheet.money_paise | caller.money_paise | {n * 100 for n in caller.plain}
    allowed_plain = sheet.plain | caller.plain | {p // 100 for p in caller.money_paise if p % 100 == 0}
    found = []
    for said in record.robot_said():
        spoken = read_numbers(said.get("filled") or said["text"])
        bad_money = sorted(p for p in spoken.money_paise if p not in allowed_money)
        bad_plain = sorted(n for n in spoken.plain if n not in allowed_plain and n * 100 not in allowed_money)
        if bad_money or bad_plain:
            found.append({"said": said.get("filled") or said["text"],
                          "amounts": [f"₹{p / 100:g}" for p in bad_money], "numbers": bad_plain})
    return found
