"""Mark what each service heard, by code, and write the comparison.

Decision 24 scores key details (order numbers, names, amounts, haan/nahi/ruko)
on phone and phone + noise clips, with speed and cost breaking ties. Whole-text
accuracy isn't scored: "chaar" spelled "char" costs a caller nothing, but a wrong
digit can cancel the wrong order. Every miss is listed with what was really heard,
so a person can spot a form the answer key should also have accepted.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from lab.ears_test.clips import VERSIONS, Detail, EarsLine

_PUNCTUATION = re.compile(r"[.,!?;:\"“”‘’()₹/\-–—।॥…]")
_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_DIGIT_WORDS = {
    "zero": "0", "shunya": "0", "shoonya": "0", "शून्य": "0", "ज़ीरो": "0", "जीरो": "0",
    "one": "1", "ek": "1", "एक": "1",
    "two": "2", "do": "2", "दो": "2",
    "three": "3", "teen": "3", "tin": "3", "तीन": "3",
    "four": "4", "char": "4", "chaar": "4", "चार": "4",
    "five": "5", "panch": "5", "paanch": "5", "पांच": "5", "पाँच": "5",
    "six": "6", "chhe": "6", "che": "6", "chah": "6", "छह": "6", "छः": "6", "छे": "6",
    "seven": "7", "saat": "7", "sat": "7", "सात": "7",
    "eight": "8", "aath": "8", "आठ": "8",
    "nine": "9", "nau": "9", "नौ": "9",
}
_SPLIT_DIGITS = re.compile(r"(?<=\d) (?=\d)")
# Punctuation then a space ends a clause; "4,521" has no space, so it stays one number.
_CLAUSE_END = re.compile(r"[,.;:?!।…]\s+")


def flat(text: str) -> str:
    """Lower case, no punctuation, single spaces. A comma inside a number goes
    too, so "4,521" reads "4521" (an apostrophe stays, for "don't")."""
    text = text.lower().translate(_DEVANAGARI_DIGITS).replace("’", "'")
    return " ".join(_PUNCTUATION.sub("", text).split())


def with_digits(text: str) -> str:
    """Digit words as digits, joined: "chaar paanch do ek" and "4 5 2 1" both
    read "4521". Only ever one of two readings tried, because "do" is also the
    Hindi for "do it" and "ek" for "a"."""
    digits = " ".join(_DIGIT_WORDS.get(word, word) for word in text.split())
    return _SPLIT_DIGITS.sub("", digits)


def _within(phrase: str, text: str) -> bool:
    return f" {phrase} " in f" {text} "


def _readings(heard: str) -> list[str]:
    """Every fair way to read the text. Each clause also gets its own digits:
    in "4530, 45 tees" the comma ends one number, and joining across it would
    make "453045" and lose a number that was heard right."""
    plain = flat(heard)
    clauses = [flat(clause) for clause in _CLAUSE_END.split(heard)]
    return [plain, with_digits(plain), *(with_digits(clause) for clause in clauses)]


def heard_detail(detail: Detail, heard: str) -> bool:
    readings = _readings(heard)
    return any(_within(flat(form), reading) for form in detail.forms for reading in readings)


def latin_share(text: str) -> float | None:
    """How much of the writing is in English letters (decision 24 asks for them).
    None when there's no writing to judge."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return None
    return sum(ch.isascii() for ch in letters) / len(letters)


@dataclass
class Tally:
    right: int = 0
    total: int = 0

    def add(self, ok: bool) -> None:
        self.right += ok
        self.total += 1

    def __str__(self) -> str:
        return f"{self.right}/{self.total}" + (f" ({100 * self.right / self.total:.0f}%)" if self.total else "")


@dataclass
class ServiceScore:
    name: str
    details: Tally = field(default_factory=Tally)
    by_version: dict[str, Tally] = field(default_factory=dict)
    by_kind: dict[str, Tally] = field(default_factory=dict)
    clips: int = 0
    failed: int = 0
    retries: int = 0
    gave_partials: int = 0
    audio_minutes: float = 0.0
    first_text_s: list[float] = field(default_factory=list)
    final_after_s: list[float] = field(default_factory=list)
    final_latin: list[float] = field(default_factory=list)
    partial_latin: list[float] = field(default_factory=list)
    misses: list[dict[str, str]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def score_service(name: str, lines: Iterable[EarsLine], results: dict[str, Any], clips: dict[str, dict[str, Any]]) -> ServiceScore:
    by_id = {line.id: line for line in lines}
    score = ServiceScore(name)
    score.by_version = {version: Tally() for version in VERSIONS}
    for clip_key, clip in sorted(clips.items()):
        line = by_id.get(clip_key.split("/")[1])
        if line is None:
            continue
        for version in VERSIONS:
            row = results.get(f"{name}/{clip_key}/{version}")
            if row is None:
                continue  # never sent: left out of the numbers, and shown as a count
            score.retries += row.get("retries", 0)
            if "error" in row:
                score.failed += 1
                score.errors.append(f"{clip_key}/{version}: {row['error']}")
                continue
            score.clips += 1
            score.audio_minutes += clip["seconds"] / 60
            score.gave_partials += row["partials"] > 0
            if row["first_text_s"] is not None:
                score.first_text_s.append(row["first_text_s"])
            if row["final_after_speech_s"] is not None:
                score.final_after_s.append(row["final_after_speech_s"])
            for values, text in ((score.final_latin, row["final"]), (score.partial_latin, row["last_partial"])):
                share = latin_share(text)
                if share is not None:
                    values.append(share)
            for detail in line.expect:
                ok = heard_detail(detail, row["final"])
                score.details.add(ok)
                score.by_version[version].add(ok)
                score.by_kind.setdefault(detail.kind, Tally()).add(ok)
                if not ok:
                    score.misses.append({"clip": f"{clip_key}/{version}", "kind": detail.kind, "said": line.say,
                                         "heard": row["final"] or "(nothing)"})
    return score


def _median(values: list[float]) -> str:
    return f"{statistics.median(values):.2f} s" if values else "none"


def _share(values: list[float]) -> str:
    return f"{100 * statistics.mean(values):.0f}%" if values else "none"


def report(scores: list[ServiceScore], expected_clips: int, prices: dict[str, float | None], noise: str, when: str) -> str:
    """The comparison. Ranked by key details: speed and cost only break ties,
    and they appear beside the score for a person to weigh."""
    ranked = sorted(scores, key=lambda s: -(s.details.right / s.details.total if s.details.total else -1))
    kinds = sorted({kind for s in scores for kind in s.by_kind})
    lines = [
        "# Ears test (Part 2, decision 24)",
        "",
        f"- **Run:** {when}. Clips per service: {expected_clips}",
        f"- **Noise:** {noise}",
        "- **How:** each clip streamed at real speed, as a phone would. Key details marked by code on the final text.",
        "",
        "| Service | Key details right | Phone | Phone + noise | Clips done | Failed | Retries | Gave partial words | Words start coming (median) | Final after caller stops (median) | Final in English letters | Partial in English letters | Cost of this run |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in ranked:
        price = prices.get(s.name)
        cost = "price not checked" if price is None else f"${price * s.audio_minutes:.2f}"
        lines.append(
            f"| {s.name} | {s.details} | {s.by_version['phone']} | {s.by_version['noise']} | {s.clips} | {s.failed} | {s.retries} "
            f"| {s.gave_partials}/{s.clips} | {_median(s.first_text_s)} | {_median(s.final_after_s)} "
            f"| {_share(s.final_latin)} | {_share(s.partial_latin)} | {cost} |"
        )
    lines += ["", "## By kind of detail", "", "| Service | " + " | ".join(kinds) + " |", "|---|" + "---|" * len(kinds)]
    for s in ranked:
        lines.append(f"| {s.name} | " + " | ".join(str(s.by_kind.get(kind, Tally())) for kind in kinds) + " |")
    for s in ranked:
        lines += ["", f"## {s.name}: what it missed ({len(s.misses)})", ""]
        if s.misses:
            lines += ["| Clip | Detail | Said | Heard |", "|---|---|---|---|"]
            lines += [f"| {m['clip']} | {m['kind']} | {m['said']} | {m['heard'].replace('|', '/')} |" for m in s.misses]
        else:
            lines.append("Nothing.")
        if s.errors:
            lines += ["", "Failures:", *[f"- {error}" for error in s.errors]]
    return "\n".join(lines) + "\n"
