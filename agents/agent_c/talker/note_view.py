"""How a note reads to the Talker: the backend's answer in plain words, with every
order number and amount as a slot (decisions 8 and 10).

The notebook keeps the backend's raw answer; this turns it into the Talker's "NOTES:"
line at the moment it is given. Order numbers become [order], amounts [amount] (or
the field's own name, as [refunded]), outcomes [result]; a status or a payment method
becomes its ready phrase from the pack's spoken words, so the Talker has nothing to
translate. The slot's real value goes to the speaker, which fills it just before the
voice.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from agents.agent_c.notebook import FACT, FAILED, NEED_INFO, RESULT, Note
from engine.contract.action import Phrase
from engine.contract.spoken import SpokenWords
from packs.food.money import rupees_to_paise, spoken_amount

ORDER = "order"
RESULT_SLOT = "result"
SlotValue = str | Phrase  # a plain value is said as written; a phrase in the call's language

_MONEY = re.compile(r"₹\s?(\d+(?:\.\d{1,2})?)")
# Fields shown as the pack's ready phrase, from the group of the same meaning.
PHRASE_GROUPS = {"status": "status", "payment": "payment_method"}


class NoteView:
    """Remembers every slot it has given, so the Thinker's own words can be slotted
    the same way ("order 4521" in its reply becomes "order [order]")."""

    def __init__(self, words: SpokenWords) -> None:
        self._words = words
        self.slots: dict[str, SlotValue] = {}
        self._raw: dict[str, str] = {}  # what a slot stood for in the backend's answer -> slot name

    def line(self, note: Note, language: str) -> str:
        """The note as the Talker reads it."""
        if note.kind == FACT and "tool" in note.data:
            return self._fact(note.data, language)
        if note.kind == RESULT:
            return self._result(note.data)
        text = self._slot_known(note.text)
        if note.kind == NEED_INFO:
            return f"ask the caller: {text}"
        if note.kind == FAILED:
            return f"failed: {text} Apologise and offer to connect a team member."
        return text

    # ---- inside -------------------------------------------------------------------

    def _fact(self, data: Mapping[str, Any], language: str) -> str:
        fields = data.get("data") or {}
        parts = [f"{data['tool']}: {data['status']}"]
        for name, value in fields.items():
            if name == "token":
                continue  # the Guard's business, never the Talker's
            parts.append(f"{name.replace('_', ' ')} {self._value(name, value, language)}")
        if data["status"] != "ok" or not fields:
            parts.append(self._slot_known(str(data.get("message") or "")))
        return "; ".join(part for part in parts if part)

    def _value(self, name: str, value: Any, language: str) -> str:
        if name == "order_no":
            return self._slot(ORDER, str(value), str(value))
        if isinstance(value, str) and _MONEY.fullmatch(value.strip()):
            paise = rupees_to_paise(_MONEY.fullmatch(value.strip()).group(1))
            return self._slot(name, spoken_amount(paise), value.strip()) if paise is not None else value
        group = PHRASE_GROUPS.get(name)
        if group and isinstance(value, str) and value in self._words.values(group):
            return self._words.phrase(group, value, language)
        if isinstance(value, list):
            return ", ".join(str(item) for item in value)
        return self._slot_known(str(value))

    def _result(self, data: Mapping[str, Any]) -> str:
        self.slots[RESULT_SLOT] = Phrase("result", f"{data['action']}_{data['outcome']}")
        reason = self._slot_known(str(data.get("reason") or ""))
        return f"{data['action'].replace('_', ' ')} [{RESULT_SLOT}]" + (f"; reason: {reason}" if reason else "")

    def _slot(self, name: str, value: SlotValue, raw: str) -> str:
        if raw in self._raw:
            return f"[{self._raw[raw]}]"  # the same value seen again keeps its slot
        if name in self.slots:
            # A second, different value for the same name gets its own slot, so two
            # orders or two amounts in one call never swap.
            number = 2
            while f"{name}_{number}" in self.slots:
                number += 1
            name = f"{name}_{number}"
        self.slots[name] = value
        self._raw[raw] = name
        return f"[{name}]"

    def _slot_known(self, text: str) -> str:
        """Put slots where the text repeats a value a slot already stands for."""
        for raw, name in sorted(self._raw.items(), key=lambda item: -len(item[0])):
            pattern = re.escape(raw).replace("₹", "₹?") if raw.startswith("₹") else rf"(?<!\d){re.escape(raw)}(?!\d)"
            text = re.sub(pattern, f"[{name}]", text)
        return text
