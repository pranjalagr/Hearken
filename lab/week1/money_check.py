"""Does the Talker get amounts right? (Part 2, decisions 8 and 10.)

The Talker never writes an amount: its notes show [amount], it copies the
slot, and code fills in the backend's exact words (agents/agent_c/talker/
slots.py). Before that, it wrote amounts itself from a prompt rule and scored
16 to 22 of 24, once saying "5 paise" for ₹0.50 (1 Nov 2026).

Each reply is judged by code, not by ear:
- no slot the notes never gave, and no number the Talker typed itself
  (decision 8's number check: a wrong amount can't get past it);
- the amount used where one is needed (a left-out amount is counted);
- after filling, what the voice gets has no decimal point and says the
  amount as expected.
Each situation is asked several times: the Talker runs at temperature 1.0.

Reuses the Flash-Lite check's parts (same model, settings, packet, pacing),
so this measures the real Talker setup.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import yaml

from agents.agent_c.talker.slots import UnknownSlot, fill_slots, stray_numbers, used_slots
from engine.contract.spoken import LANGUAGES, SpokenWords
from lab.week1.flash_lite_check import (
    MODEL,
    REQUEST_TIMEOUT_S,
    ROOT,
    STYLE_PROMPT,
    Reply,
    Situation,
    _cell,
    ask_all,
    gemini_asker,
    situation_from,
)
from packs.food.money import format_rupees, spoken_amount
from packs.food.pack import FoodPack

SITUATIONS = Path(__file__).with_name("money_situations.yaml")
TRIES = 3  # per situation
SLOT = "[amount]"
_EXTRA = frozenset({"amount_paise", "language", "say", "never"})

# Any of these means a voice would be handed a decimal to misread.
_DECIMAL = re.compile(r"\d\.\d")
_DECIMAL_WORDS = re.compile(r"\b(point|dashamlav|decimal)\b", re.IGNORECASE)


@dataclass(frozen=True)
class MoneySituation:
    situation: Situation  # as the backend answered, with the amount written out
    amount_paise: int  # what code fills into [amount]
    language: str  # the call's language: picks "rupaye" or "rupees"
    say: tuple[str, ...]  # the filled reply must contain one of these; empty: no amount needed
    never: tuple[str, ...]  # the filled reply must contain none of these

    @property
    def seen(self) -> Situation:
        """What the Talker sees: the notes with the amount swapped for the slot."""
        written = format_rupees(self.amount_paise)
        return replace(self.situation, notes=tuple(note.replace(written, SLOT) for note in self.situation.notes))


def load_money_situations(path: Path = SITUATIONS) -> list[MoneySituation]:
    """Strict, like the Flash-Lite check's loader: a typo fails here."""
    loaded = []
    for item in yaml.safe_load(path.read_text(encoding="utf-8")):
        if not _EXTRA <= set(item):
            raise ValueError(f"situation {item.get('id')}: needs amount_paise, language, say and never")
        situation = situation_from(item, extra_keys=_EXTRA)
        if item["language"] not in LANGUAGES:
            raise ValueError(f"situation {situation.id}: language must be one of {LANGUAGES}")
        case = MoneySituation(situation, int(item["amount_paise"]), item["language"],
                              tuple(map(str, item["say"])), tuple(map(str, item["never"])))
        # A note must carry the amount exactly as the backend writes it, or no
        # slot would replace it and the Talker would see the raw amount.
        if not any(SLOT in note for note in case.seen.notes):
            raise ValueError(f"situation {situation.id}: no note says {format_rupees(case.amount_paise)}")
        loaded.append(case)
    if [m.situation.id for m in loaded] != list(range(1, len(loaded) + 1)):
        raise ValueError("situation ids must be 1, 2, 3... in order")
    return loaded


def _contains(text: str, phrase: str) -> bool:
    """A whole-word match, ignoring case and spacing: "0 rupaye" must not be
    found inside "450 rupaye", nor "5 paise" inside "05 paise"."""
    flat = " ".join(text.lower().split())
    wanted = re.escape(" ".join(phrase.lower().split()))
    return re.search(rf"(?<![\w₹]){wanted}(?!\w)", flat) is not None


def verdict(spoken: str, case: MoneySituation) -> str | None:
    """None if what the voice gets says money right, else why not."""
    if _DECIMAL.search(spoken) or _DECIMAL_WORDS.search(spoken):
        return "says a decimal amount"
    for phrase in case.never:
        if _contains(spoken, phrase):
            return f'says "{phrase}"'
    if case.say and not any(_contains(spoken, phrase) for phrase in case.say):
        return f'doesn\'t say the amount as "{case.say[0]}"'
    return None


def slot_verdict(reply: str, case: MoneySituation, words: SpokenWords) -> tuple[str | None, str]:
    """(None if the reply is right, else why not; what the voice would get)."""
    try:
        spoken = fill_slots(reply, {"amount": spoken_amount(case.amount_paise)}, words, case.language)
    except UnknownSlot as error:
        return f"made up a slot [{error.args[0]}]", reply
    given = [*case.seen.notes, *(text for speaker, text in case.seen.turns if speaker == "caller")]
    if typed := stray_numbers(reply, given):
        return f"typed a number itself: {', '.join(typed)}", spoken
    if case.say and "amount" not in used_slots(reply):
        return "left the amount out", spoken
    return verdict(spoken, case), spoken


def report(cases: list[MoneySituation], replies: list[Reply], when: datetime,
           words: SpokenWords) -> tuple[str, int]:
    """A table of every reply with its verdict; returns (text, how many passed)."""
    by_id = {case.situation.id: case for case in cases}
    rows, passed = [], 0
    for reply in replies:
        case = by_id[reply.situation.id]
        if reply.error:
            problem, spoken = f"error: {reply.error}", ""
        else:
            problem, spoken = slot_verdict(reply.text, case, words)
        passed += problem is None
        rows.append(f"| {reply.situation.id} | {_cell(case.situation.good)} | {_cell(reply.text)} "
                    f"| {_cell(spoken)} | {'✅' if problem is None else '❌ ' + _cell(problem)} |")
    lines = [
        "# Money check (Part 2, decisions 8 and 10)",
        "",
        f"- **Model:** `{MODEL}`, run on {when:%d %b %Y at %H:%M}, {TRIES} tries per situation",
        f"- **Style prompt:** `{STYLE_PROMPT.relative_to(ROOT).as_posix()}`",
        f"- **How:** the notes show {SLOT}; code fills it (`agents/agent_c/talker/slots.py`)",
        f"- **Passed:** {passed} of {len(replies)}",
        "",
        "| # | A good reply... | Talker wrote | The voice gets | Verdict |",
        "|---|---|---|---|---|",
        *rows,
    ]
    return "\n".join(lines) + "\n", passed


async def run(out_dir: Path = ROOT / "runs" / "week1") -> tuple[Path, int, int]:
    from common.env import load_env, require
    from common.google import google_client

    load_env()
    cases = load_money_situations()
    client = google_client(require("GEMINI_API_KEY"), REQUEST_TIMEOUT_S)
    try:
        asks = [case.seen for case in cases for _ in range(TRIES)]
        replies = await ask_all(asks, gemini_asker(client, MODEL, STYLE_PROMPT.read_text(encoding="utf-8")))
    finally:
        await client.aio.aclose()
    when = datetime.now()
    text, passed = report(cases, replies, when, FoodPack().spoken_words())
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"money_check_{when:%Y%m%d_%H%M}.md"
    path.write_text(text, encoding="utf-8")
    return path, passed, len(replies)
