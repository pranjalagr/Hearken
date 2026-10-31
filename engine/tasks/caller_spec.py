"""What a task tells its pretend caller, and what it tells the director.

`caller:` (Part 3, C6) holds only what a real customer would know: the
caller's brain sees this section and nothing else, never the timeline, the
expected result or the forbidden actions.

`race:` (C10) and `drop:` (C12) are for the director, which times exact
moments; the brain never sees them either.

Every list below is closed on purpose: a typo in a task file ("in_hurry")
must fail when the task loads, not quietly give a caller with no personality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# C7
PERSONALITIES = ("patient", "angry", "elderly", "in_a_hurry", "quiet")
LANGUAGES = ("mostly_hindi", "mixed", "mostly_english")
# C4: 4-6 voices, male and female, young and older. Each label maps to a real
# voice once the voice company is picked (week 1).
VOICES = ("female_young", "female_older", "male_young", "male_older")
# C8: the first three number styles.
NUMBER_STYLES = ("hindi_digits", "english_digits", "hindi_pairs")
# C7: the 10 tricks, in the plan's order.
TRICKS = (
    "switches_language",
    "haan_haan",
    "interrupts",
    "corrects_number_once",
    "changes_mind",
    "indian_numbers",
    "goes_silent",
    "yes_with_condition",
    "asks_out_of_scope",
    "drops_call",
)
# C7: each task picks 1-2 tricks; some pick none.
MAX_TRICKS = 2

# C10: the 4 race moments.
RACE_MOMENTS = ("while_asking", "right_after_sending", "while_running", "saved_not_heard")
# C12: where a drop test hangs up, and how.
DROP_MOMENTS = (
    "mid_caller_sentence",
    "mid_robot_sentence",
    "long_silence",
    "after_haan",
    "while_action_runs",
    "after_done",
    "random",
)
DROP_KINDS = ("clean", "silent")


@dataclass(frozen=True)
class CallerSpec:
    name: str
    facts: dict[str, Any]  # the only facts the caller may say (checked against the seed)
    wants: str
    personality: str
    language: str
    voice: str
    number_style: str
    opening_line: str  # the same first turn for every robot (C13)
    tricks: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class RaceSpec:
    moment: str
    action: str  # the risky action whose moment is raced
    line: str  # what the caller says at that moment
    then_wants: str  # what the caller wants after saying it


@dataclass(frozen=True)
class DropSpec:
    moment: str
    kind: str  # "clean": the red button; "silent": the line just dies


def parse_caller(raw: Any) -> CallerSpec:
    body = _mapping(raw, "caller")
    required = {"name", "facts", "wants", "personality", "language", "voice", "number_style", "opening_line"}
    _exact_keys(body, required, optional={"tricks"}, where="caller")
    for key in required - {"facts"}:
        _text(body[key], f"caller.{key}")
    facts = _mapping(body["facts"], "caller.facts")
    if not facts:
        raise ValueError("caller.facts is empty: the caller must know something")
    for name, value in facts.items():
        if not isinstance(value, (str, int, float)) or isinstance(value, bool):
            raise ValueError(f"caller.facts.{name} must be text or a number")
    _one_of(body["personality"], PERSONALITIES, "caller.personality")
    _one_of(body["language"], LANGUAGES, "caller.language")
    _one_of(body["voice"], VOICES, "caller.voice")
    _one_of(body["number_style"], NUMBER_STYLES, "caller.number_style")
    tricks = body.get("tricks") or []
    if not isinstance(tricks, list):
        raise ValueError("caller.tricks must be a list")
    for trick in tricks:
        _one_of(trick, TRICKS, "caller.tricks")
    if len(tricks) != len(set(tricks)):
        raise ValueError("caller.tricks lists a trick twice")
    if len(tricks) > MAX_TRICKS:
        # More tricks at once and a failed call can't say which one broke the robot.
        raise ValueError(f"caller.tricks: at most {MAX_TRICKS} per task")
    if "indian_numbers" in tricks and body["number_style"] == "english_digits":
        raise ValueError("the indian_numbers trick needs a Hindi number_style")
    return CallerSpec(
        name=body["name"].strip(),
        facts=dict(facts),
        wants=body["wants"].strip(),
        personality=body["personality"],
        language=body["language"],
        voice=body["voice"],
        number_style=body["number_style"],
        opening_line=body["opening_line"].strip(),
        tricks=tuple(tricks),
    )


def parse_race(raw: Any) -> RaceSpec:
    body = _mapping(raw, "race")
    _exact_keys(body, {"moment", "action", "line", "then_wants"}, optional=set(), where="race")
    for key in body:
        _text(body[key], f"race.{key}")
    _one_of(body["moment"], RACE_MOMENTS, "race.moment")
    return RaceSpec(body["moment"], body["action"], body["line"].strip(), body["then_wants"].strip())


def parse_drop(raw: Any) -> DropSpec:
    body = _mapping(raw, "drop")
    _exact_keys(body, {"moment", "kind"}, optional=set(), where="drop")
    _one_of(body["moment"], DROP_MOMENTS, "drop.moment")
    _one_of(body["kind"], DROP_KINDS, "drop.kind")
    return DropSpec(body["moment"], body["kind"])


def _mapping(raw: Any, where: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"{where} must be a mapping")
    return raw


def _exact_keys(body: dict[str, Any], required: set[str], optional: set[str], where: str) -> None:
    missing = required - set(body)
    extra = set(body) - required - optional
    if missing:
        raise ValueError(f"{where} is missing {sorted(missing)}")
    if extra:
        raise ValueError(f"{where} has unknown keys {sorted(extra)}")


def _text(value: Any, where: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where} must be non-empty text")


def _one_of(value: Any, allowed: tuple[str, ...], where: str) -> None:
    if value not in allowed:
        raise ValueError(f"{where}: {value!r} is not one of {list(allowed)}")
