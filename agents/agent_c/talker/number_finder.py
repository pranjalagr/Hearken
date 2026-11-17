"""Hearing an order number in a caller's turn (decision 27, 17 Nov 2026).

Two parts, each doing what it is good at:
1. The finder (Sarvam's AI) points at the number words in the turn and tags each one:
   "order" (the order number), "other" (an amount, a time) or "unsure".
2. Code (agents/common/number_reader.py) turns only those words into digits and checks
   they were really said, in that order. It never guesses.

The finder runs only on a turn the Talker's AI says holds a number (decided 17 Nov 2026).
Brought over from the speed branch (agents/agent_c/speed/finder.py, 16 Nov 2026).
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from agents.common.number_reader import INCOMPLETE, OK, TOO_LONG, UNREADABLE, read_number

FIND_RULES = Path(__file__).resolve().parent / "prompts" / "find_numbers.md"

# The answer's shape, in Sarvam's strict JSON mode (every object closed, every field required).
FIND_FORM = {"type": "object", "additionalProperties": False, "required": ["numbers"], "properties": {"numbers": {
    "type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["words", "tag"],
                               "properties": {"words": {"type": "string"},
                                              "tag": {"type": "string", "enum": ["order", "other", "unsure"]}}}}}}

FindNumbers = Callable[[str], Awaitable[list[dict[str, str]]]]

# What the Talker does with a turn's order number.
HEARD = "heard"  # one full number: say "number mil gaya" and pass it on
SHORT = "short"  # fewer or more digits than an order number has: ask for the full number
FOR_THINKER = "for_thinker"  # number words the reader can't read: the Thinker's AI reads them (15 Nov 2026)
NO_NUMBER = "no_number"  # no order number in the turn: ask for it


@dataclass(frozen=True)
class Heard:
    what: str  # one of the four above
    digits: str = ""  # the reader's digits, when HEARD


def sarvam_find(client: httpx.AsyncClient, model: str, rules: str | None = None) -> FindNumbers:
    """The real finder: one request to Sarvam's chat, answered in JSON."""
    rules = rules or FIND_RULES.read_text(encoding="utf-8")
    answer_form = {"type": "json_schema", "json_schema": {"name": "numbers", "strict": True, "schema": FIND_FORM}}

    async def find(words: str) -> list[dict[str, str]]:
        # Thinking off with a null reasoning_effort: Sarvam refused "none" (8 Nov 2026).
        body = {"model": model, "reasoning_effort": None, "response_format": answer_form,
                "messages": [{"role": "system", "content": rules}, {"role": "user", "content": words}]}
        response = await client.post("/chat/completions", json=body)
        if response.is_error:
            # Raised with the response, so common/retry.py can see a 429 or a 5xx.
            raise httpx.HTTPStatusError(f"Sarvam said {response.status_code}", request=response.request,
                                        response=response)
        answer = json.loads(response.json()["choices"][0]["message"]["content"] or "")
        return list(answer.get("numbers") or [])

    return find


def what_was_heard(numbers: list[Mapping[str, Any]], heard: str, length: int) -> Heard:
    """Code's part: from the finder's tagged number words, what the Talker should do.
    Only words tagged "order" count; an unsure tag or two different numbers are left
    to the Thinker, which reads the caller's whole sentence."""
    if any(item.get("tag") == "unsure" for item in numbers):
        return Heard(FOR_THINKER)
    readings = [read_number(str(item.get("words", "")), heard, length) for item in numbers
                if item.get("tag") == "order"]
    if not readings:
        return Heard(NO_NUMBER)
    found = {reading.digits for reading in readings if reading.status == OK}
    if len(found) == 1 and all(reading.status == OK for reading in readings):
        return Heard(HEARD, found.pop())
    if any(reading.status in (INCOMPLETE, TOO_LONG) for reading in readings) and not found:
        return Heard(SHORT)
    # Unreadable words, or two different numbers: the Thinker's AI reads the sentence.
    if any(reading.status == UNREADABLE for reading in readings) or len(found) > 1:
        return Heard(FOR_THINKER)
    return Heard(NO_NUMBER)
