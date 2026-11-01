"""Spoken words: what the robot says out loud for a pack's fixed values.

A pack keeps them in a data file, not in code, so changing the wording never
touches logic (Part 2 decisions 8 and 9). The engine knows nothing about what
the groups mean; a group is just a name such as "status", and each value in it
has one ready phrase per language.

Why ready phrases at all: the Talker is a small AI, and turning a raw value
like `out_for_delivery` into Hinglish is exactly where it can say something
wrong. A fixed phrase leaves it nothing to translate.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from engine.contract.action import Phrase, SpokenQuestion

# Part 2 decision 12: the robot mirrors the caller's mix, Hinglish or English.
LANGUAGES = ("hinglish", "english")

# The group holding each risky action's yes-question lines (Part 2 decision 9).
YES_QUESTION = "yes_question"
# Every yes-question ends with a closed question, in the same words each time,
# so the Guard always judges an answer to "yes or no" and nothing looser.
QUESTION_ENDINGS = {"hinglish": "Haan ya nahi?", "english": "Yes or no?"}

# Part 2 decision 12: English letters everywhere ("haan", not "हाँ"), so every
# part of the system matches the same word the same way.
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")


@dataclass(frozen=True)
class SpokenWords:
    # group -> value -> language -> phrase
    groups: dict[str, dict[str, dict[str, str]]] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path) -> SpokenWords:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        # Checked once, at load: a missing language or a Hindi-script phrase
        # should fail when the pack loads, not in the middle of a call.
        problems = [problem for group, values in raw.items() for problem in _problems(group, values)]
        if problems:
            raise ValueError(f"bad spoken words in {path.name}: " + "; ".join(problems))
        return cls(raw)

    def phrase(self, group: str, value: str, language: str) -> str:
        if language not in LANGUAGES:
            raise ValueError(f"unknown language {language!r}, expected one of {LANGUAGES}")
        try:
            return self.groups[group][value][language]
        except KeyError:
            # Loud on purpose: a missing phrase is a pack bug, and a silent
            # fallback would let the Talker improvise exactly what this file prevents.
            raise KeyError(f"no spoken phrase for {group}.{value}") from None

    def values(self, group: str) -> set[str]:
        return set(self.groups.get(group, {}))

    def say(self, phrase: Phrase, language: str) -> str:
        """One phrase in `language`, with its own slots filled ("{rupees}
        rupaye {paise} paise" -> "249 rupaye 50 paise"). The one way every
        caller of this class says a phrase: yes-questions and the Talker's
        [amount] slots (Part 2 decision 10) come out the same."""
        # A slot with no number raises KeyError: never a hole in what is said.
        return self.phrase(phrase.group, phrase.value, language).format(**phrase.fill)

    def yes_question(self, question: SpokenQuestion, language: str) -> str:
        """Fill the action's line with its values, and its phrases in `language`.

        A placeholder with no value raises KeyError: the question must never
        be spoken with a hole in it.
        """
        template = self.phrase(YES_QUESTION, question.line, language)
        filled = dict(question.values)
        for name, phrase in question.phrases.items():
            filled[name] = self.say(phrase, language)
        # str.format reads only the template, so braces inside a caller's
        # address are copied as they are, never treated as placeholders.
        return template.format(**filled)


def _slots(text: Any) -> set[str] | None:
    """The {names} a phrase will be filled with; None if it isn't text or a
    brace is broken, which would otherwise crash mid-call instead of at load."""
    if not isinstance(text, str):
        return None
    try:
        return {name for _, name, _, _ in string.Formatter().parse(text) if name is not None}
    except ValueError:
        return None


def _problems(group: str, values: Any) -> list[str]:
    if not isinstance(values, dict) or not values:
        return [f"group {group!r} has no values"]
    problems = []
    for value, phrases in values.items():
        where = f"{group}.{value}"
        if not isinstance(phrases, dict) or set(phrases) != set(LANGUAGES):
            problems.append(f"{where} needs exactly {LANGUAGES}")
            continue
        for language, text in phrases.items():
            if not isinstance(text, str) or not text.strip():
                problems.append(f"{where}.{language} is empty")
            elif _DEVANAGARI.search(text):
                problems.append(f"{where}.{language} uses Hindi script; write it in English letters")
            elif _slots(text) is None:
                problems.append(f"{where}.{language} has a broken {{slot}}; write {{name}}, or {{{{ for a brace")
            elif group == YES_QUESTION and not text.rstrip().endswith(QUESTION_ENDINGS[language]):
                problems.append(f"{where}.{language} must end with {QUESTION_ENDINGS[language]!r}")
        # Every language is filled from the same numbers, so a slot only one
        # of them has would be a hole in the other.
        each = [_slots(phrases[language]) for language in LANGUAGES]
        if None not in each and any(slots != each[0] for slots in each):
            problems.append(f"{where}: every language needs the same {{slots}}, got {dict(zip(LANGUAGES, map(sorted, each)))}")
    return problems
