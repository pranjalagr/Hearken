"""The haan vs ruko phrase list, turned into clip plans: what is said, in which voice, with
which answer, and whether it is held back for testing (Part 4).

Everything here is fixed, not random from run to run, so a rerun plans the same clips and
the clip maker never pays for one twice.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

import yaml

PHRASES_FILE = Path(__file__).with_name("phrases.yaml")
# The lab's five caller voices, male and female (C4).
VOICES = ("aviraj", "rhea", "sanjay", "nirupma", "inaaya")
# Short phrases are cheap, so all five voices say them: the model must learn the sound of
# "haan" in general, not one person's "haan". Long sentences are dearer and plentiful, so two.
SENTENCE_VOICES = 2
KEEP, STOP, MIXED = "keep", "stop", "mixed"
SEED = 2026_11_10  # fixes which sentences are picked and held back


@dataclass(frozen=True)
class Plan:
    id: str
    group: str  # keep, stop, mixed or sentence
    answer: str  # keep, stop, or mixed (keep until the first comma, stop after)
    text: str
    voice: str
    held_back: bool  # only for testing: never trained on


def _short_id(text: str) -> str:
    """By the words, not by place in the list, so editing the list doesn't rename clips."""
    return hashlib.sha256(text.lower().encode("utf-8")).hexdigest()[:8]


def load_phrases(path: Path = PHRASES_FILE) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    for group in (KEEP, STOP, MIXED):
        entries = data[group]["train"] + data[group]["held_back"]
        if any(not isinstance(entry, str) for entry in entries):
            # YAML reads a bare yes, no, on or off as true or false: they need quotes.
            raise ValueError(f"{group}: every phrase must be text; quote words like yes and no")
    lowered = [e.lower() for g in (KEEP, STOP, MIXED) for e in data[g]["train"] + data[g]["held_back"]]
    repeated = sorted({e for e in lowered if lowered.count(e) > 1})
    if repeated:
        raise ValueError(f"phrases listed twice: {repeated}")
    missing_comma = [m for m in data[MIXED]["train"] + data[MIXED]["held_back"] if "," not in m]
    if missing_comma:
        raise ValueError(f"mixed phrases need a comma where the answer changes: {missing_comma}")
    return data


def pick_sentences(pool: list[str], count: int, held_back_share: float) -> tuple[list[str], list[str]]:
    """A fixed sample of ordinary caller sentences: (for training, held back)."""
    unique = sorted({" ".join(s.split()) for s in pool if s.strip()})
    chosen = random.Random(SEED).sample(unique, min(count, len(unique)))
    cut = round(len(chosen) * held_back_share)
    return chosen[cut:], chosen[:cut]


def sentence_pool() -> list[str]:
    """The lab's existing caller sentences: the order-number turns and the end-of-turn lines."""
    from lab.eot_test.clips import load_lines
    from lab.week1.number_transcripts import load_transcripts

    return [case.heard for case in load_transcripts()] + [line.replace("...", "") for line in load_lines()]


def plan_clips(data: dict, sentences: list[str] | None = None) -> list[Plan]:
    plans: list[Plan] = []
    for group, answer in ((KEEP, KEEP), (STOP, STOP), (MIXED, MIXED)):
        for held_back, entries in ((False, data[group]["train"]), (True, data[group]["held_back"])):
            for text in entries:
                plans += [Plan(f"{group}-{_short_id(text)}-{voice}", group, answer, text, voice, held_back) for voice in VOICES]
    spec = data["sentences"]
    trained, held = pick_sentences(sentences if sentences is not None else sentence_pool(), spec["count"], spec["held_back_share"])
    for n, (held_back, text) in enumerate([(False, s) for s in trained] + [(True, s) for s in held]):
        for k in range(SENTENCE_VOICES):
            voice = VOICES[(n + 2 * k) % len(VOICES)]  # spread the voices over the sentences
            plans.append(Plan(f"sentence-{_short_id(text)}-{voice}", "sentence", STOP, text, voice, held_back))
    return plans
