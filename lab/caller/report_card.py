"""Is the caller itself any good? (Part 3, C15.)

A bad caller makes a good robot look bad, so the caller is marked too, per batch.
Automatic checks first, from each call's folder. Then an AI reads the call for what
code can't see: did the caller stay in role, stay on its goal, and do its trick? An
AI's claim counts only with a quote from the call, because a caller bug "needs clear
proof in the transcript; if unsure, it counts against the robot" (C15).
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from common.wire.protocol import PIECE_MS
from lab.caller.brain.brain import TRICK_DONE, numbers_in
from lab.caller.director.plan import DIRECTOR_TRICKS
from lab.caller.settings import CallerSettings
from lab.runner.run_call import LAB_PROBLEM
from lab.scorer.call_record import CallRecord, load_call

# C15's bar before official runs.
MAX_CALLER_BUGS_PER_100 = 2
MIN_RACE_HIT_RATE = 0.98
# Reply gap limits (C9), in seconds, from the robot's last sound to the caller's first.
MIN_GAP_S, MAX_GAP_S = 0.3, 3.0
AI_RULES = """You check a pretend customer used to test phone support robots. You read one call and report only the
customer's mistakes, never the robot's. Quote the customer's exact words as proof, and mark only what is clearly there.

Lines marked "planned" were said on purpose by the test: never report them.

You get what the customer was asked to do, then the call. Report each that happened:
- role_flip: the customer talks like the support agent, or says it's an AI, an actor or a test.
- info_dump: the customer gives many details at once, unprompted, in one turn.
- forgot_goal: the customer stops pursuing what it wants without being refused or told it's done.
- early_hangup: the customer ends the call while its goal is still open and nothing stopped it.
- mishearing: the customer answers something the robot didn't say, because its hearing got the words wrong.
"""
AI_KINDS = ("role_flip", "info_dump", "forgot_goal", "early_hangup", "mishearing")
AI_FORM = {
    "type": "object", "additionalProperties": False, "required": ["mistakes"],
    "properties": {"mistakes": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["kind", "quote", "why"],
        "properties": {"kind": {"type": "string", "enum": list(AI_KINDS)}, "quote": {"type": "string"}, "why": {"type": "string"}},
    }}},
}
Ask = Callable[[str, str], Awaitable[dict[str, Any]]]


@dataclass
class CallCard:
    call_id: str
    task_id: str
    mistakes: list[dict[str, Any]] = field(default_factory=list)
    caught: list[str] = field(default_factory=list)  # slips code fixed: the robot never faced them
    reply_gaps_s: list[float] = field(default_factory=list)
    gaps_out_of_range: int = 0
    interruptions: list[float] = field(default_factory=list)  # accidental: never a director's line
    race_hit: bool | None = None
    brain_ms: list[int] = field(default_factory=list)
    voice_ms: list[int] = field(default_factory=list)
    dropped_thoughts: int = 0  # replies thrown away unused: not a bug, but each is a wasted request
    brain_retries: int = 0
    tokens: int = 0
    characters: dict[str, int] = field(default_factory=dict)
    ears_seconds: float = 0.0
    ears_company: str = ""

    @property
    def caller_bug(self) -> bool:
        return bool(self.mistakes or self.interruptions)


def _switched_language(lines: list[dict[str, Any]], _facts: dict[str, Any]) -> bool:
    return len({line.get("language") for line in lines}) >= 2


def _gave_a_condition(lines: list[dict[str, Any]], _facts: dict[str, Any]) -> bool:
    return any(TRICK_DONE["yes_with_condition"].search(line["text"]) for line in lines)


def _misstated_the_number(lines: list[dict[str, Any]], facts: dict[str, Any]) -> bool:
    right = str(facts.get("order_no", ""))
    return any(len(number) == len(right) and sum(a != b for a, b in zip(number, right)) == 1
               for line in lines for number in numbers_in(line["text"]))


# Tricks code can see in the call. The AI marked them on real calls and was nearly always
# wrong: it called "Haan, lekin SMS aayega na?" no condition (5 Nov 2026). The director's
# tricks and changes_mind happen at planned moments, so they can't be the brain's miss.
TRICK_CHECKS = {"switches_language": _switched_language, "yes_with_condition": _gave_a_condition,
                "corrects_number_once": _misstated_the_number}


def mark_call(record: CallRecord) -> CallCard:
    """Everything code can see, from the call's folder alone."""
    about, outcome = record.about, record.outcome
    card = CallCard(about.get("call_id", record.folder.name), about.get("task_id", ""),
                    ears_company=str(about.get("ears", "")))
    for event in record.events:
        kind = event["event"]
        if kind == "caller.mistake":
            card.mistakes.append({"kind": event.get("kind", ""), "detail": event.get("detail", ""), "from": "code"})
        elif kind == "caller.caught":
            card.caught.append(event.get("detail", ""))
        elif kind == "caller.thought":
            card.brain_ms.append(event["brain_ms"])
            card.voice_ms.append(event["voice_ms"])
        elif kind == "caller.thought_dropped":
            card.dropped_thoughts += 1
        elif kind == "caller.brain_retry":
            card.brain_retries += 1
        elif kind == "caller.cost":
            if event.get("service") == "brain":
                card.tokens += event.get("tokens", 0)
            elif event.get("service") == "voice":
                model = event.get("model", "")
                card.characters[model] = card.characters.get(model, 0) + event.get("characters", 0)
            elif event.get("service") == "ears":
                card.ears_seconds += event.get("seconds", 0.0)
    for line in record.caller_lines():
        gap = _gap_before(line["start_ms"], record)
        if gap is None:
            continue
        if gap < 0:
            card.interruptions.append(line["start_ms"] / 1000)
            continue
        card.reply_gaps_s.append(round(gap / 1000, 2))
        card.gaps_out_of_range += not MIN_GAP_S <= gap / 1000 <= MAX_GAP_S
    caller = about.get("caller") or {}
    for trick in caller.get("tricks") or []:
        if trick in TRICK_CHECKS and not TRICK_CHECKS[trick](record.caller_lines(), caller.get("facts") or {}):
            card.mistakes.append({"kind": "trick_not_done", "detail": trick, "from": "code"})
    race = outcome.get("race") or {}
    card.race_hit = race.get("hit")
    return card


def _gap_before(start_ms: float, record: CallRecord) -> float | None:
    """How long after the robot's last sound this line started; negative: over it."""
    last_end = None
    for span in record.robot_sound:
        if span.start_ms <= start_ms < span.end_ms:
            return span.start_ms - start_ms if span.end_ms - start_ms > PIECE_MS else 0.0
        if span.end_ms <= start_ms:
            last_end = span.end_ms
    return None if last_end is None else start_ms - last_end


def transcript(record: CallRecord) -> str:
    timed = []
    for line in record.caller_lines():
        timed.append((line["start_ms"], f"customer: {line['text']}"))
    for line in record.director_lines():
        now_wants = f"; from now on it wants: {line['then_wants']}" if line.get("then_wants") else ""
        timed.append((line["caller_ms"], f"customer (planned{now_wants}): {line['line']}"))
    for said in record.robot_said():
        timed.append((record.clock.to_ms(said["at_ns"]), f"robot: {said.get('filled') or said['text']}"))
    heard = [(e["caller_ms"], f"(the customer's hearing got: {e['text']})") for e in record.events if e["event"] == "caller.heard"]
    return "\n".join(text for _at, text in sorted(timed + heard, key=lambda item: item[0]))


async def ai_marks(record: CallRecord, card: CallCard, ask: Ask) -> None:
    """Adds what the AI found, keeping only claims it can quote (C15)."""
    caller = record.about.get("caller") or {}
    text = transcript(record)
    # Without these, the AI called "Haan, lekin SMS aayega na?" an info dump (5 Nov 2026).
    on_purpose = [trick for trick in caller.get("tricks") or [] if trick not in DIRECTOR_TRICKS]
    brief = (f"What it wanted: {caller.get('wants')}\nIt does these on purpose, so they're no mistake: "
             f"{on_purpose or 'none'}\n\nTHE CALL:\n{text}")
    answer = await ask(AI_RULES, brief)
    planned = [_flat(line["line"]) for line in record.director_lines()]
    for mistake in answer.get("mistakes", []):
        quote = _flat(mistake.get("quote", ""))
        # An empty quote is "in" any text, so it must be refused by name. A planned line
        # is the test's own: the AI reported them despite its rules (5 Nov 2026).
        if quote and quote in _flat(text) and not any(quote in line for line in planned):
            card.mistakes.append({"kind": mistake["kind"], "detail": f"{mistake.get('quote')} ({mistake.get('why')})", "from": "ai"})


def _flat(text: str) -> str:
    return " ".join(str(text).lower().split())


@dataclass
class ReportCard:
    calls: int
    caller_bugs: int
    bugs_per_100: float
    caught_by_code: int
    mistakes_by_kind: dict[str, int]
    median_reply_gap_s: float | None
    gaps_out_of_range: int
    interruptions: int
    race_hits: int
    race_total: int
    median_brain_ms: float | None
    median_voice_ms: float | None
    dropped_thoughts: int
    brain_retries: int
    dollars: float
    dollars_per_call: float
    tokens_per_call: float
    passes: bool
    cards: list[CallCard]

    def markdown(self) -> str:
        race = f"{self.race_hits}/{self.race_total}" if self.race_total else "none in this batch"
        lines = [
            "# The caller's report card (Part 3, C15)", "",
            f"- **Bar for official runs: {'met' if self.passes else 'NOT met'}** (caller bugs <= {MAX_CALLER_BUGS_PER_100} per 100, race hits >= {MIN_RACE_HIT_RATE:.0%}, accidental interruptions = 0)",
            f"- **Calls:** {self.calls}", f"- **Caller bugs:** {self.caller_bugs} ({self.bugs_per_100:.0f} per 100)",
            f"- **Slips caught and fixed by code:** {self.caught_by_code} (the robot never faced these)",
            f"- **Accidental interruptions:** {self.interruptions}", f"- **Race hits:** {race}",
            f"- **Reply gap:** median {self.median_reply_gap_s} s; {self.gaps_out_of_range} outside {MIN_GAP_S}-{MAX_GAP_S} s",
            f"- **Brain:** median {self.median_brain_ms} ms. **Voice:** median {self.median_voice_ms} ms",
            f"- **Replies thrown away unused:** {self.dropped_thoughts}. **Brain retries:** {self.brain_retries}",
            f"- **Cost:** ${self.dollars:.3f} in all, ${self.dollars_per_call:.3f} a call; {self.tokens_per_call:.0f} brain tokens a call", "",
            "## Mistakes", "",
        ]
        lines += [f"- {kind}: {count}" for kind, count in sorted(self.mistakes_by_kind.items())] or ["None."]
        lines += ["", "## Each call", "", "| Call | Bug? | Mistakes | Caught by code | Gaps (s) |", "|---|---|---|---|---|"]
        for card in self.cards:
            said = "; ".join(f"{m['kind']}: {m['detail']}"[:160] for m in card.mistakes).replace("|", "/")
            caught = "; ".join(text[:160] for text in card.caught).replace("|", "/")
            lines.append(f"| {card.call_id} | {'yes' if card.caller_bug else 'no'} | {said or '-'} | {caught or '-'} | {card.reply_gaps_s} |")
        return "\n".join(lines) + "\n"


def summarize_cards(cards: list[CallCard], settings: CallerSettings) -> ReportCard:
    mistakes: dict[str, int] = {}
    for card in cards:
        for mistake in card.mistakes:
            mistakes[mistake["kind"]] = mistakes.get(mistake["kind"], 0) + 1
    gaps = [gap for card in cards for gap in card.reply_gaps_s]
    races = [card.race_hit for card in cards if card.race_hit is not None]
    bugs = sum(card.caller_bug for card in cards)
    per_100 = 100 * bugs / len(cards) if cards else 0.0
    hit_rate = sum(races) / len(races) if races else 1.0
    interruptions = sum(len(card.interruptions) for card in cards)
    dollars = sum(
        sum(count / 10000 * settings.price_per_10k_characters.get(model, 0.0) for model, count in card.characters.items())
        + card.ears_seconds / 60 * settings.ears_price_per_minute.get(card.ears_company, 0.0)
        for card in cards
    )
    brain = [ms for card in cards for ms in card.brain_ms]
    voice = [ms for card in cards for ms in card.voice_ms]
    return ReportCard(
        calls=len(cards), caller_bugs=bugs, bugs_per_100=per_100, caught_by_code=sum(len(card.caught) for card in cards),
        mistakes_by_kind=mistakes,
        median_reply_gap_s=round(statistics.median(gaps), 2) if gaps else None,
        gaps_out_of_range=sum(card.gaps_out_of_range for card in cards), interruptions=interruptions,
        race_hits=sum(races), race_total=len(races),
        median_brain_ms=statistics.median(brain) if brain else None,
        median_voice_ms=statistics.median(voice) if voice else None,
        dropped_thoughts=sum(card.dropped_thoughts for card in cards),
        brain_retries=sum(card.brain_retries for card in cards),
        dollars=dollars, dollars_per_call=dollars / len(cards) if cards else 0.0,
        tokens_per_call=sum(card.tokens for card in cards) / len(cards) if cards else 0.0,
        passes=bool(cards) and per_100 <= MAX_CALLER_BUGS_PER_100 and hit_rate >= MIN_RACE_HIT_RATE and interruptions == 0,
        cards=cards,
    )


async def report_on_run(run_dir: Path, settings: CallerSettings, sound_level: float, ask: Ask | None = None) -> ReportCard:
    cards = []
    for folder in sorted((run_dir / "calls").iterdir()):
        if not (folder / "clock.json").exists():
            continue
        record = load_call(folder, sound_level)
        if record.outcome.get("label") == LAB_PROBLEM:
            continue  # rerun, and cut short by the lab: it says nothing about the caller
        card = mark_call(record)
        if ask is not None:
            await ai_marks(record, card, ask)
        cards.append(card)
    report = summarize_cards(cards, settings)
    (run_dir / "caller_report.md").write_text(report.markdown(), encoding="utf-8")
    (run_dir / "caller_report.json").write_text(json.dumps(asdict(report), indent=1, ensure_ascii=False), encoding="utf-8")
    return report
