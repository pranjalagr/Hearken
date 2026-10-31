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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Part 2 decision 12: the robot mirrors the caller's mix, Hinglish or English.
LANGUAGES = ("hinglish", "english")

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
    return problems
