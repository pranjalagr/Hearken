"""The order-number test set (Part 2, decision 27): real and hand-written cases from
number_cases.yaml, plus cases made here.

The made cases are spoken with the pretend caller's own number words
(lab/caller/mouth/number_words.py), a table written separately from the robot's
number reader: a mistake in one table can't hide itself in the other (decision C8).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import yaml

from lab.caller.mouth.number_words import hindi_number, say_digits

CASES = Path(__file__).with_name("number_cases.yaml")
ASK, NONE = "ask", "none"
ORDERS = ("4521", "4512", "4530", "4547", "4561", "4533", "4540", "4568", "4555", "4501", "4009", "4710")
# How the number is said; every style is read by the robot's own table, not this one.
STYLES = {
    "hindi digits": lambda n: say_digits(n, "hindi_digits"),
    "hindi pairs": lambda n: say_digits(n, "hindi_pairs"),
    "english digits": lambda n: say_digits(n, "english_digits"),
    "written": lambda n: n,
    "written pairs": lambda n: f"{n[:2]} {n[2:]}",
    "hindi value": lambda n: hindi_number(int(n)),
}
FRAMES = (
    "Mera order number {n} hai",
    "Order {n} cancel karna hai",
    "{n}, Rahul Verma",
    "Haan ji, order {n}",
    "Order {n} cancel kar do",
    "Ek minute, haan, {n}",
    "Order {n} par chaar sau pachaas rupaye wapas chahiye",
    "{n} wala order, do din se nahi aaya",
)
# The spellings ears really wrote, put in by chance.
SPELLINGS = (("chaar", "char"), ("paanch", "panch"), ("shoonya", "shunya"), ("teen", "tin"),
             ("paintaalees", "paintalis"), ("ikkees", "ikkis"), ("chhah", "chhe"))
NO_NUMBER = ("Haan, bataiye", "Cancel kar do please", "Ek second, check karta hoon", "Do minute rukiye",
             "Mujhe refund chahiye", "Paanch sau rupaye ka order tha", "Order kal aaya tha, do din pehle")


@dataclass(frozen=True)
class NumberCase:
    id: str
    heard: str  # the ears' text of one caller turn
    span: str  # the words that are the order number ("" when there's none to take)
    expect: str  # 4 digits, "ask" or "none"
    kind: str
    note: str = ""


def load_cases(path: Path = CASES, seed: int = 7) -> list[NumberCase]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    cases = []
    for group in ("real", "hard"):
        for n, case in enumerate(raw[group], 1):
            cases.append(NumberCase(f"{group}{n}", case["heard"], case["span"], str(case["expect"]), group, case.get("note", "")))
    return cases + made_cases(seed)


def _misspell(text: str, rng: random.Random) -> str:
    for right, written in SPELLINGS:
        if right in text and rng.random() < 0.5:
            text = text.replace(right, written)
    return text


def made_cases(seed: int = 7) -> list[NumberCase]:
    """Every order in every style, in a random frame and spelling, then the hard shapes:
    fillers inside, restarts, corrections, a noise word, unfinished, and no number."""
    rng = random.Random(seed)
    cases: list[NumberCase] = []

    def add(kind: str, heard: str, span: str, expect: str) -> None:
        cases.append(NumberCase(f"{kind.replace(' ', '_')}{len(cases) + 1}", heard, span, expect, kind))

    for number in ORDERS:
        for style, say in STYLES.items():
            spoken = _misspell(say(number), rng)
            add(style, rng.choice(FRAMES).format(n=spoken), spoken, number)
    for number in ORDERS:
        words = say_digits(number, "hindi_digits").split()
        right = " ".join(words)
        with_filler = " ".join([*words[:2], rng.choice(("aa", "umm", "ahh")), *words[2:]])
        add("filler", f"Order {with_filler} hai", with_filler, number)
        add("restart", f"{' '.join(words[:2])} ahh... {right}", right, number)
        wrong = list(number)
        wrong[3] = str((int(wrong[3]) + 1) % 10)
        wrong_words = say_digits("".join(wrong), "hindi_digits")
        add("correction", f"{wrong_words}, nahi nahi, {right}", right, number)
        noisy = [*words[:2], rng.choice(("do", "ek")), *words[2:]]
        add("noise word", " ".join(noisy), " ".join(noisy), ASK)
        add("unfinished", f"Order {' '.join(words[:2])}", " ".join(words[:2]), ASK)
    for sentence in NO_NUMBER:
        add("no number", sentence, "", NONE)
    return cases
