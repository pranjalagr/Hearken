"""The transcript test set for the AI's part of reading an order number (Part 2,
decision 27): one caller turn as the ears wrote it, with every word labelled.

Each case says which words ARE the order number, which are OTHER numbers (an amount,
a time, the wrong number before a correction, another order), and what the robot
should end up with: 4 digits, "ask" (unclear, unfinished, unsure) or "none". So a test
run can say, word by word, whether the AI picked the right ones (the user asked for
exactly this, 7 Nov 2026).

Built by code with a fixed seed, plus the real and hand-written cases in
number_cases.yaml. Numbers are spoken with the pretend caller's own word table
(lab/caller/mouth/number_words.py), not the robot reader's, so a gap in one can't
hide in the other (decision C8).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import yaml

from lab.caller.mouth.number_words import ENGLISH_DIGITS, HINDI_0_TO_99, HINDI_LETTERS_0_TO_99, hindi_number

HAND = Path(__file__).with_name("number_cases.yaml")
SEED = 2026
ASK, NONE = "ask", "none"
# What each piece of a sentence is. SAME: the order number again (said twice, or named
# before it's chosen); picking that mention is just as right.
NUMBER, SAME, OTHER, PLAIN = "number", "same", "other", "plain"

_TENS = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty", 6: "sixty", 7: "seventy", 8: "eighty", 9: "ninety"}
_TEENS = ("ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen")
# How the ears really misspell number words (this week's calls and the ears test), and
# the English sound-alikes they write. The meaning stays the caller's number.
MISSPELL = {
    "chaar": ["char", "chhar", "chr", "chaa"], "paanch": ["panch", "paach", "panj", "punch"], "teen": ["tin", "theen", "tee"],
    "do": ["doh", "dho"], "ek": ["aek", "yek", "eik"], "shoonya": ["shunya", "sunya", "shuniya"], "chhah": ["chhe", "che", "chha"],
    "saat": ["sat", "saad"], "aath": ["aat", "aad"], "nau": ["naw", "nao"], "four": ["for", "fore"], "five": ["fave", "fiv", "fife"],
    "two": ["to", "too"], "three": ["tree", "free"], "one": ["won", "wan"], "six": ["sex", "siks"], "eight": ["ate"],
    "nine": ["nain"], "zero": ["ziro", "jero"], "seven": ["sevan"], "paintaalees": ["paintalis", "pentalis", "paintalish"],
    "ikkees": ["ikkis", "ekkis"], "tees": ["tis", "thees"], "chaalees": ["chalis", "chaalis"], "pachaas": ["pachas"],
    "bees": ["bis"], "baarah": ["barah", "bara"],
}
FILLERS = ("ahh", "umm", "aa", "hmm", "uh")
WAITS = ("ek second", "ek minute", "ruko", "ek sec", "wait")
CORRECTIONS = ("nahi nahi", "nahi", "sorry", "galat bola", "matlab", "nahi, sahi number hai", "oh no, sorry", "galti se bola")
NAMES = ("Rahul Verma", "Priya Sharma", "Aman Gupta", "Neha Singh", "Sana Khan", "Vikram Rao")
# Sentences around the number. Many hold number words that aren't the order number
# (do, ek, teen, paanch, an amount, a time): the AI must leave those out.
FRAMES = [
    "Mera order number {n} hai", "Order {n} cancel karna hai", "{n}, {name}", "Haan ji, {n}", "Order {n} cancel kar do",
    "Ek minute, haan, {n}", "{n} par chaar sau pachaas rupaye wapas chahiye", "Do din pehle order kiya tha, number {n}",
    "Teen baje order kiya, {n}", "Order {n} ka address flat do C hai", "My order number is {n}", "It's {n}, please cancel it",
    "Order {n}, paid 450 rupees by card", "{n} wala order abhi tak nahi aaya, do ghante ho gaye", "Haan bhaiya, {n} hai, jaldi karo",
    "Number hai {n}, aur naam {name}", "{n}", "Order number? {n}", "Sector paanch mein rehta hoon, order {n}",
    "Order {n} ka refund paanch din mein milega kya?", "Ji {n}, ek baar check kar lijiye", "{n} hai mera order, {name} ke naam pe",
    "Bhai mera order {n}, do baar call kar chuka hoon", "Yes, {n}, that's my order", "Order {n} mein ek item kam aaya",
]
OTHER_ONLY = [  # number words, but no order number in them
    "Do sau pachaas rupaye wapas chahiye", "Teen baje order kiya tha", "Ek minute rukiye", "Do din ho gaye",
    "Paanch sau rupaye ka order tha", "Flat number do C", "450 rupees refund please", "Sector paanch, house chauhattar",
    "Ek do din mein aa jayega kya?", "Mere do order hain", "Chaar log the, chaar thali chahiye thi", "It was 250 rupees",
    "Ek baar aur check karo", "Do minute mein call karta hoon", "Paanch star dunga agar jaldi ho",
]
NO_NUMBER = [
    "Haan, bataiye", "Cancel kar do please", "Ek second, check karta hoon", "Mujhe refund chahiye", "Order abhi tak nahi aaya",
    "Hello, sun rahe ho?", "Mera order cancel karna hai", "Kya hua mera order ka?", "Theek hai, thank you", "Rider ka number dedo",
    "Wait, let me check", "I want my money back", "Naam Priya Sharma hai", "Aap kaun bol rahe ho?", "Kitna time lagega?",
    "Order number yaad nahi", "Haan haan", "Nahi nahi, rehne do", "Bas jaldi karo", "Address badalna hai",
]


@dataclass(frozen=True)
class TranscriptCase:
    id: str
    kind: str
    heard: str  # one caller turn, as the ears wrote it
    pieces: tuple[tuple[str, str], ...]  # (text, NUMBER / OTHER / PLAIN): the labels, word by word
    expect: str  # 4 digits, "ask" or "none"
    note: str = ""

    @property
    def number_words(self) -> str:
        """The words that ARE the order number, in order (fillers inside left out)."""
        return " ".join(text for text, role in self.pieces if role == NUMBER)

    @property
    def same_number(self) -> str:
        return " ".join(text for text, role in self.pieces if role == SAME)

    @property
    def other_numbers(self) -> str:
        return " | ".join(text for text, role in self.pieces if role == OTHER)

    def marked(self) -> str:
        """The sentence with [order number] and {other numbers} marked, for reading."""
        out = []
        for text, role in self.pieces:
            out.append(f"[{text}]" if role == NUMBER else f"<{text}>" if role == SAME else f"{{{text}}}" if role == OTHER else text)
        return " ".join(out).replace(" ,", ",")


# ---- ways to say a number ------------------------------------------------------------


def _english_pair(two: str) -> str:
    n = int(two)
    if two[0] == "0":
        return f"oh {ENGLISH_DIGITS[n]}" if n else "oh oh"
    if n < 10:
        return ENGLISH_DIGITS[n]
    if n < 20:
        return _TEENS[n - 10]
    return _TENS[n // 10] + ("" if n % 10 == 0 else f" {ENGLISH_DIGITS[n % 10]}")


def _english_value(n: int) -> str:
    words = [ENGLISH_DIGITS[n // 1000], "thousand"]
    if n % 1000 >= 100:
        words += [ENGLISH_DIGITS[n % 1000 // 100], "hundred"]
    rest = n % 100
    if rest:
        words.append(_english_pair(f"{rest:02d}") if rest >= 10 else ENGLISH_DIGITS[rest])
    return " ".join(words)


def _hindi_pairs(number: str) -> str:
    def pair(two: str) -> str:
        return f"{HINDI_0_TO_99[0]} {HINDI_0_TO_99[int(two[1])]}" if two[0] == "0" else HINDI_0_TO_99[int(two)]
    return f"{pair(number[:2])} {pair(number[2:])}"


STYLES = {
    "hindi digits": lambda n, r: " ".join(HINDI_0_TO_99[int(d)] for d in n),
    "hindi pairs": lambda n, r: _hindi_pairs(n),
    "english digits": lambda n, r: " ".join(ENGLISH_DIGITS[int(d)] for d in n),
    "english pairs": lambda n, r: f"{_english_pair(n[:2])} {_english_pair(n[2:])}",
    "written": lambda n, r: n,
    "written pairs": lambda n, r: f"{n[:2]} {n[2:]}",
    "written spaced": lambda n, r: " ".join(n),
    "mixed hindi english": lambda n, r: " ".join(r.choice((HINDI_0_TO_99, ENGLISH_DIGITS))[int(d)] for d in n),
    "hindi value": lambda n, r: hindi_number(int(n)),
    "english value": lambda n, r: _english_value(int(n)),
    "hindi letters": lambda n, r: " ".join(HINDI_LETTERS_0_TO_99[int(d)] for d in n),
    "pair then digits": lambda n, r: f"{HINDI_0_TO_99[int(n[:2])] if n[0] != '0' else n[:2]} {HINDI_0_TO_99[int(n[2])]} {HINDI_0_TO_99[int(n[3])]}",
}
EVERYDAY = ("hindi digits", "hindi digits", "hindi pairs", "english digits", "written", "written pairs", "mixed hindi english")
# Styles of two or more words, for cases that put something inside the number.
SPLITTABLE = tuple(style for style in EVERYDAY if style != "written")


def _numbers(rng: random.Random) -> list[str]:
    special = ["4521", "4512", "4530", "4547", "4561", "4555", "4501", "4009", "4710", "4400", "1111", "2020", "9090", "4000"]
    return special + [str(rng.randint(1000, 9999)) for _ in range(46)]


def _misspell(words: str, rng: random.Random, chance: float) -> str:
    out = []
    for word in words.split():
        options = MISSPELL.get(word)
        out.append(rng.choice(options) if options and rng.random() < chance else word)
    return " ".join(out)


def _wrong_version(number: str, rng: random.Random) -> str:
    """A near number, as people misremember: one digit off."""
    i = rng.randrange(4)
    digits = list(number)
    digits[i] = str((int(digits[i]) + rng.choice((1, 2, 7, 9))) % 10)
    wrong = "".join(digits)
    return wrong if wrong[0] != "0" and wrong != number else _wrong_version(number, rng)


# ---- the cases -------------------------------------------------------------------------


def _in_frame(frame: str, spoken: list[tuple[str, str]], rng: random.Random) -> list[tuple[str, str]]:
    before, after = frame.format(n="\0", name=rng.choice(NAMES)).split("\0")
    pieces = [(before.strip(), PLAIN)] if before.strip() else []
    pieces += spoken
    if after.strip():
        pieces.append((after.strip(), PLAIN))
    return pieces


def _case(cases: list, kind: str, pieces: list[tuple[str, str]], expect: str, note: str = "") -> None:
    heard = " ".join(text for text, _ in pieces).replace(" ,", ",")
    cases.append(TranscriptCase(f"{kind.replace(' ', '_')}_{len(cases) + 1}", kind, heard, tuple(pieces), expect, note))


def made_cases(seed: int = SEED) -> list[TranscriptCase]:
    rng = random.Random(seed)
    numbers = _numbers(rng)
    cases: list[TranscriptCase] = []

    def say(number: str, style: str | None = None, misspell: float = 0.0) -> str:
        return _misspell(STYLES[style or rng.choice(EVERYDAY)](number, rng), rng, misspell)

    def split_point(words: list[str]) -> int:
        return rng.randint(1, len(words) - 1)

    # 1. Every style, in a sentence; some misspelled the way the ears do.
    for style in STYLES:
        for _ in range(10):
            n = rng.choice(numbers)
            _case(cases, f"style: {style}", _in_frame(rng.choice(FRAMES), [(say(n, style), NUMBER)], rng), n)
    for _ in range(60):
        n = rng.choice(numbers)
        _case(cases, "misspelled", _in_frame(rng.choice(FRAMES), [(say(n, rng.choice(("hindi digits", "english digits", "mixed hindi english", "hindi pairs")), 0.5), NUMBER)], rng), n)
    # 2. Things inside the number: fillers, "ek second", "haan".
    for kind, inside, count in (("filler inside", FILLERS, 45), ("wait inside", WAITS, 40), ("haan inside", ("haan", "haan ji", "ji"), 25)):
        for _ in range(count):
            n = rng.choice(numbers)
            words = say(n, rng.choice(SPLITTABLE), 0.2).split()
            cut = split_point(words)
            spoken = [(" ".join(words[:cut]), NUMBER), (rng.choice(inside), PLAIN), (" ".join(words[cut:]), NUMBER)]
            _case(cases, kind, _in_frame(rng.choice(FRAMES), spoken, rng), n)
    # 3. Restarts: a part, then the whole number again.
    for _ in range(35):
        n = rng.choice(numbers)
        words = say(n, rng.choice(SPLITTABLE)).split()
        part = " ".join(words[:split_point(words)])
        spoken = [(part, OTHER), (rng.choice(FILLERS + ("...",)), PLAIN), (" ".join(words), NUMBER)]
        _case(cases, "restart", _in_frame(rng.choice(FRAMES), spoken, rng), n)
    for _ in range(30):  # the user's example: "chaar paanch ek second ek do chaar paanch ek do"
        n = rng.choice(numbers)
        words = say(n, "hindi digits").split()
        cut = split_point(words)
        spoken = [(" ".join(words[:cut]), OTHER), (rng.choice(WAITS), PLAIN), (" ".join(words[cut:]), OTHER),
                  (rng.choice(("", "haan", "matlab")), PLAIN), (" ".join(words), NUMBER)]
        _case(cases, "broken then whole", [p for p in spoken if p[0]], n)
    # 4. Corrections: the last number the caller settles on counts.
    for _ in range(50):
        n = rng.choice(numbers)
        style = rng.choice(EVERYDAY)
        spoken = [(say(_wrong_version(n, rng), style), OTHER), (rng.choice(CORRECTIONS), PLAIN), (say(n, style), NUMBER)]
        _case(cases, "correction", _in_frame(rng.choice(FRAMES), spoken, rng), n)
    for _ in range(40):  # the correction said another way: "4532 nahi, chaar paanch teen ek"
        n = rng.choice(numbers)
        first, second = rng.sample(EVERYDAY, 2)
        spoken = [(say(_wrong_version(n, rng), first), OTHER), (rng.choice(CORRECTIONS), PLAIN), (say(n, second), NUMBER)]
        _case(cases, "correction in another style", _in_frame(rng.choice(FRAMES), spoken, rng), n)
    for _ in range(15):  # the user's example: "4532 nahi four thousand 53 ek": a garbled correction
        n = rng.choice(numbers)
        garbled = f"{ENGLISH_DIGITS[int(n[0])]} thousand {n[1:3]} {HINDI_0_TO_99[int(n[3])]}"
        spoken = [(say(_wrong_version(n, rng), "written"), OTHER), (rng.choice(CORRECTIONS), PLAIN), (garbled, NUMBER)]
        _case(cases, "garbled correction", spoken, ASK, "the correction itself isn't a clear number")
    for _ in range(15):  # "nahi" with no new number: taken back
        n = rng.choice(numbers)
        spoken = [(say(n), OTHER), (rng.choice(("nahi wait, woh purana tha", "nahi, galat hai, ruko dekhta hoon", "matlab, yaad nahi")), PLAIN)]
        _case(cases, "taken back", _in_frame(rng.choice(FRAMES), spoken, rng), ASK, "no number left to act on")
    # 5. Two numbers: two orders (one chosen), or unsure between two.
    for _ in range(40):
        a, b = rng.sample(numbers, 2)
        frame = rng.choice(("Mere do order hain, {a} aur {b}, {b} wala cancel karo", "{a} deliver ho gaya, {b} cancel karo",
                            "{a} nahi, {b} wala order", "Pehle {a} tha, ab {b} cancel karna hai", "{b} cancel karo, {a} theek hai"))
        said_a, said_b = say(a), say(b)
        parts = frame.replace("{a}", "\0a\0").replace("{b}", "\0b\0").split("\0")
        spoken = []
        seen_b = frame.count("{b}")
        for part in parts:
            if part == "a":
                spoken.append((said_a, OTHER))
            elif part == "b":
                seen_b -= 1
                spoken.append((said_b, NUMBER if seen_b == 0 else SAME))
            elif part.strip():
                spoken.append((part.strip(" ,"), PLAIN))
        _case(cases, "two orders", spoken, b, "the one the caller wants acted on")
    for _ in range(15):
        a, b = rng.sample(numbers, 2)
        spoken = [(say(a), OTHER), ("ya", PLAIN), (say(b), OTHER), (rng.choice(("pakka nahi pata", "yaad nahi", "dono mein se ek")), PLAIN)]
        _case(cases, "unsure between two", spoken, ASK)
    # 6. Unfinished, stammers and the same number twice.
    for _ in range(30):
        n = rng.choice(numbers)
        words = say(n, rng.choice(("hindi digits", "english digits"))).split()
        spoken = [(" ".join(words[:rng.randint(1, 3)]), NUMBER), (rng.choice(FILLERS + ("...", "baaki yaad nahi")), PLAIN)]
        _case(cases, "unfinished", _in_frame(rng.choice(FRAMES), spoken, rng), ASK)
    for _ in range(25):
        n = rng.choice(numbers)
        words = say(n, "hindi digits").split()
        i = rng.randrange(len(words))
        words.insert(i, words[i])  # a digit said twice: a stammer, or a 5-digit number?
        _case(cases, "stammer", _in_frame(rng.choice(FRAMES), [(" ".join(words), NUMBER)], rng), ASK, "can't tell a stammer from a 5th digit")
    for _ in range(20):
        n = rng.choice(numbers)
        said = say(n)
        spoken = [(said, SAME), (rng.choice(("", "haan", "phir se,")), PLAIN), (said, NUMBER)]
        _case(cases, "said twice", [p for p in spoken if p[0]], n)
    # 7. The number next to an amount; words run together by the ears; Hindi letters mixed in.
    for _ in range(30):
        n = rng.choice(numbers)
        amount = rng.choice(("450", "do sau pachaas", "chaar sau", "240 rupees", "ek sau bees"))
        spoken = [(say(n), NUMBER), (rng.choice(("ka", "par", "mein")), PLAIN), (amount, OTHER), (rng.choice(("rupaye wapas", "refund chahiye", "kat gaye")), PLAIN)]
        _case(cases, "amount next to it", spoken, n)
    for _ in range(15):
        n = rng.choice(numbers)
        words = say(n, "hindi digits").split()
        cut = split_point(words)
        joined = "".join(words[:cut]) + " " + " ".join(words[cut:])
        _case(cases, "words run together", _in_frame(rng.choice(FRAMES), [(joined, NUMBER)], rng), n)
    for _ in range(20):
        n = rng.choice(numbers)
        said = " ".join(rng.choice((HINDI_LETTERS_0_TO_99[int(d)], HINDI_0_TO_99[int(d)], d)) for d in n)
        _case(cases, "scripts mixed", _in_frame(rng.choice(FRAMES), [(said, NUMBER)], rng), n)
    # 8. No order number at all.
    for sentence in OTHER_ONLY:
        _case(cases, "only other numbers", [(sentence, OTHER)], NONE)
    for sentence in NO_NUMBER:
        _case(cases, "no number", [(sentence, PLAIN)], NONE)
    return cases


def hand_cases(path: Path = HAND) -> list[TranscriptCase]:
    """The real ears text (ears test and this week's calls) and the hand-written hard
    cases. Their labels give the order
    number's words; the rest of the sentence isn't split further."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    cases = []
    for group in ("real", "hard", "calls"):
        for n, case in enumerate(raw[group], 1):
            heard, span = case["heard"], case["span"]
            if span and span.lower() in heard.lower():
                start = heard.lower().rindex(span.lower())
                pieces = [(heard[:start].strip(), PLAIN), (heard[start:start + len(span)], NUMBER), (heard[start + len(span):].strip(), PLAIN)]
            else:
                pieces = [(heard, PLAIN)]
            cases.append(TranscriptCase(f"{group}_{n}", group, heard, tuple(p for p in pieces if p[0]), str(case["expect"]), case.get("note", "")))
    return cases


def load_transcripts(seed: int = SEED) -> list[TranscriptCase]:
    return hand_cases() + made_cases(seed)
