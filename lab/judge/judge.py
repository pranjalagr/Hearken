"""The judge agent (Part 5, L10 and L11): marks language and truth, which
code can't.

It reads the truth sheet (written by code from the records) and the call
turn by turn, and answers in a fixed form that code checks before using it:
every claim must quote the robot's real words, and every robot reply after
the caller's first turn must be marked. A malformed answer is never guessed
at: it's an error, and the call stays unjudged.

It never knows which robot it is marking (A, B or C), and "can't tell" goes
to a person, never counted as a lie.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lab.scorer.call_record import CallRecord

RULES = Path(__file__).with_name("rules.md")
MODEL = "gemini-3.8-flash"
VERDICTS = ("true", "false", "unsupported", "cant_tell")
FALSE = ("false", "unsupported")  # both are claims the robot had no right to make

# The answer form (a JSON schema the model must follow).
FORM = {
    "type": "object",
    "properties": {
        "replies": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "turn": {"type": "integer"},
                    "language": {"type": "string", "enum": ["same", "different"]},
                    "claims": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "quote": {"type": "string"},
                                "verdict": {"type": "string", "enum": list(VERDICTS)},
                                "why": {"type": "string"},
                            },
                            "required": ["quote", "verdict", "why"],
                        },
                    },
                },
                "required": ["turn", "language", "claims"],
            },
        }
    },
    "required": ["replies"],
}


@dataclass(frozen=True)
class Turn:
    number: int
    speaker: str  # caller, director or robot
    text: str
    language: str | None = None  # caller turns only


@dataclass
class Packet:
    truth: list[str]
    turns: list[Turn]

    def text(self) -> str:
        lines = ["THE TRUTH SHEET:", *self.truth, "", "THE CALL:"]
        for turn in self.turns:
            if turn.speaker == "caller":
                lines.append(f"{turn.number}. caller [{turn.language}]: {turn.text}")
            elif turn.speaker == "director":
                lines.append(f"{turn.number}. caller (director): {turn.text}")
            else:
                lines.append(f"{turn.number}. robot: {turn.text}")
        return "\n".join(lines)

    def to_mark(self) -> list[int]:
        """The robot turns the judge must mark: every one after the caller's first turn."""
        first = next((t.number for t in self.turns if t.speaker == "caller"), None)
        return [t.number for t in self.turns if t.speaker == "robot" and first is not None and t.number > first]


@dataclass
class Marks:
    """The judge's marks for one call, checked."""

    replies: list[dict[str, Any]]
    language_same: int = 0
    language_judged: int = 0
    false_claims: list[dict[str, Any]] = field(default_factory=list)
    cant_tell: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self, model: str) -> dict[str, Any]:
        return {"model": model, "replies": self.replies, "language_same": self.language_same,
                "language_judged": self.language_judged, "false_claims": self.false_claims,
                "cant_tell": self.cant_tell}


class BadAnswer(ValueError):
    """The judge's answer broke its form: not used, never guessed at."""


def packet_from_call(record: CallRecord) -> Packet:
    """The call as the judge reads it: every turn in time order, the caller's
    with its logged language, and the truth sheet from the score."""
    timed: list[tuple[float, str, str, str | None]] = []
    for line in record.caller_lines():
        timed.append((line["start_ms"], "caller", line["text"], line.get("language")))
    for line in record.director_lines():
        timed.append((line["caller_ms"], "director", line["line"], None))
    for said in record.robot_said():
        timed.append((record.clock.to_ms(said["at_ns"]), "robot", said.get("filled") or said["text"], None))
    timed.sort(key=lambda item: item[0])
    turns = [Turn(n, speaker, text, language) for n, (_at, speaker, text, language) in enumerate(timed, 1)]
    score = json.loads((record.folder / "score.json").read_text(encoding="utf-8"))
    return Packet(score["truth_sheet"], turns)


def check_answer(packet: Packet, answer: dict[str, Any]) -> Marks:
    """Use the answer only if it keeps the form: every robot turn marked once,
    every quote really in that turn."""
    replies = answer.get("replies")
    if not isinstance(replies, list):
        raise BadAnswer("no replies list")
    by_turn = {t.number: t for t in packet.turns}
    wanted = set(packet.to_mark())
    seen = [reply.get("turn") for reply in replies]
    if sorted(seen) != sorted(wanted):
        raise BadAnswer(f"marked turns {sorted(seen)}, must be exactly {sorted(wanted)}")
    marks = Marks(replies)
    for reply in replies:
        turn = by_turn[reply["turn"]]
        if reply.get("language") not in ("same", "different"):
            raise BadAnswer(f"turn {turn.number}: language must be same or different")
        marks.language_judged += 1
        marks.language_same += reply["language"] == "same"
        for claim in reply.get("claims", []):
            if claim.get("verdict") not in VERDICTS:
                raise BadAnswer(f"turn {turn.number}: unknown verdict {claim.get('verdict')!r}")
            if _flat(claim.get("quote", "")) not in _flat(turn.text) or not claim.get("quote", "").strip():
                raise BadAnswer(f"turn {turn.number}: the quote {claim.get('quote')!r} isn't the robot's words")
            entry = {"turn": turn.number, "quote": claim["quote"], "why": claim.get("why", "")}
            if claim["verdict"] in FALSE:
                marks.false_claims.append({**entry, "verdict": claim["verdict"]})
            elif claim["verdict"] == "cant_tell":
                marks.cant_tell.append(entry)
    return marks


def _flat(text: str) -> str:
    return " ".join(text.lower().split())


Ask = Callable[[str, str], Awaitable[dict[str, Any]]]


async def judge(packet: Packet, ask: Ask) -> Marks:
    if not packet.to_mark():
        return Marks([])  # nothing the robot said after the caller spoke: nothing to judge
    answer = await ask(RULES.read_text(encoding="utf-8"), packet.text())
    return check_answer(packet, answer)


def gemini_judge(client: Any, model: str = MODEL, on_retry: Callable[[BaseException, float], None] | None = None) -> Ask:
    """The real judge: Gemini, answering in the form, allowed to think (it
    runs after the call, so the client's time limit, the lab's
    judge_timeout_s, is far longer than a live reply's)."""
    from google.genai import types

    from lab.week1.flash_lite_check import with_retries

    async def ask(rules: str, text: str) -> dict[str, Any]:
        config = types.GenerateContentConfig(
            system_instruction=rules, response_mime_type="application/json", response_schema=FORM)
        response = await with_retries(lambda: client.aio.models.generate_content(model=model, contents=text,
                                                                                 config=config), on_retry=on_retry)
        try:
            return json.loads(response.text or "")
        except ValueError as exc:
            raise BadAnswer(f"not JSON: {exc}") from exc

    return ask
