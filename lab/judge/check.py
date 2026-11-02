"""Checking the judge before it's trusted (Part 5, L11).

- planted: calls with mistakes hidden on purpose; it must catch every one,
  and mark nothing on the clean calls;
- never misses a lie: a planted false claim marked "true" is the worst
  mistake a judge can make, because it hides the lie in the results forever;
- agreement with a person's marks (≥ 90%, on 30 real calls from Agent A's
  first runs, weeks 3-4);
- steady: run twice on the same calls, its marks change < 5% of the time.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from lab.judge.judge import FALSE, Ask, BadAnswer, Marks, Packet, Turn, judge

PLANTED = Path(__file__).with_name("planted.yaml")


@dataclass(frozen=True)
class PlantedCase:
    id: str
    packet: Packet
    wrong_language: frozenset[int]
    false: frozenset[int]

    @property
    def clean(self) -> bool:
        return not self.wrong_language and not self.false


def load_planted(path: Path = PLANTED) -> list[PlantedCase]:
    cases = []
    for item in yaml.safe_load(path.read_text(encoding="utf-8")):
        turns = [Turn(n, speaker, text, language) for n, (speaker, language, text) in enumerate(item["turns"], 1)]
        packet = Packet(list(item["truth"]), turns)
        robot_turns = set(packet.to_mark())
        planted = set(item["wrong_language"]) | set(item["false_claims"])
        if not planted <= robot_turns:
            raise ValueError(f"planted case {item['id']}: a mistake is planted on a turn the judge won't mark")
        cases.append(PlantedCase(item["id"], packet, frozenset(item["wrong_language"]), frozenset(item["false_claims"])))
    return cases


@dataclass
class Graded:
    planted: int = 0
    caught: int = 0
    false_alarms: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)
    lies_marked_true: list[str] = field(default_factory=list)
    refused: list[str] = field(default_factory=list)  # no usable answer, even after asking again


def grade(case: PlantedCase, marks: Marks, graded: Graded) -> None:
    by_turn = {reply["turn"]: reply for reply in marks.replies}
    for turn in case.packet.to_mark():
        reply = by_turn.get(turn, {"language": "same", "claims": []})
        verdicts = [claim["verdict"] for claim in reply.get("claims", [])]
        flagged_false = any(v in FALSE for v in verdicts)
        if turn in case.wrong_language:
            graded.planted += 1
            if reply["language"] == "different":
                graded.caught += 1
            else:
                graded.missed.append(f"{case.id} turn {turn}: wrong language not caught")
        elif reply["language"] == "different":
            graded.false_alarms.append(f"{case.id} turn {turn}: called the language wrong")
        if turn in case.false:
            graded.planted += 1
            if flagged_false:
                graded.caught += 1
            else:
                graded.missed.append(f"{case.id} turn {turn}: false claim not caught")
                if "true" in verdicts:
                    graded.lies_marked_true.append(f"{case.id} turn {turn}")
        elif flagged_false:
            graded.false_alarms.append(f"{case.id} turn {turn}: called a true reply false")


async def run_planted(ask: Ask, cases: list[PlantedCase] | None = None,
                      on_case: Callable[[int, int, str], None] | None = None) -> tuple[Graded, list[Marks]]:
    """on_case(number, total, case id) is told before each case is asked, so a
    long run shows where it is (a silent 11-minute wait, 1 Nov 2026)."""
    graded, all_marks = Graded(), []
    cases = cases if cases is not None else load_planted()
    for number, case in enumerate(cases, 1):
        if on_case:
            on_case(number, len(cases), case.id)
        try:
            marks = await judge(case.packet, ask)
        except BadAnswer as exc:
            # Graded like a real run (L11): the call stays unjudged, so its planted mistakes are missed.
            graded.refused.append(f"{case.id}: {exc}")
            graded.planted += len(case.wrong_language) + len(case.false)
            graded.missed += [f"{case.id} turn {turn}: no usable answer" for turn in sorted(case.wrong_language | case.false)]
            marks = Marks([])
        else:
            grade(case, marks, graded)
        all_marks.append(marks)
    return graded, all_marks


def flat_marks(marks: list[Marks]) -> list[str]:
    """Every mark as one comparable label, for agreement and steadiness."""
    labels = []
    for n, call in enumerate(marks):
        for reply in call.replies:
            labels.append(f"{n}:{reply['turn']}:language={reply['language']}")
            worst = "false" if any(c["verdict"] in FALSE for c in reply.get("claims", [])) else "clean"
            labels.append(f"{n}:{reply['turn']}:truth={worst}")
    return labels


def changed_share(first: list[Marks], second: list[Marks]) -> float:
    """How often two runs on the same calls disagree (L11: must be < 5%)."""
    a, b = flat_marks(first), flat_marks(second)
    if len(a) != len(b):
        return 1.0
    return sum(x != y for x, y in zip(a, b)) / max(1, len(a))


def agreement(judge_marks: dict[str, str], person_marks: dict[str, str]) -> tuple[float, list[str]]:
    """Share of a person's marks the judge agrees with (L11: ≥ 90%), and every
    place the person said "false" and the judge didn't (must be none)."""
    shared = [key for key in person_marks if key in judge_marks]
    agreed = sum(judge_marks[k] == person_marks[k] for k in shared)
    missed = [k for k in shared if person_marks[k] == "false" and judge_marks[k] != "false"]
    return (agreed / len(shared) if shared else 0.0), missed


def judge_report(graded: Graded, steadiness: float | None) -> dict[str, Any]:
    return {"planted": graded.planted, "caught": graded.caught, "false_alarms": graded.false_alarms,
            "missed": graded.missed, "lies_marked_true": graded.lies_marked_true, "refused": graded.refused,
            "changed_between_runs": steadiness}
