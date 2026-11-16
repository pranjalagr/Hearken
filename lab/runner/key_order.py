"""Which key of each service a lab run uses, and moving to the next one when a key fails.

The user keeps several free keys per service (lab/key_order.yaml). On 15 Nov 2026 runs
stopped three times on a full or refused key and waited for the user to pick another,
so the user asked (16 Nov 2026) for an order to follow by itself. The choice is made
by setting the HEARKEN_*_KEY names that common/env.py reads: robots are started as new
programs and read their keys at start, so a switch reaches the next call, never a live one.

Only names are ever printed, never keys.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from common.env import GEMINI_KEY_CHOICE, GROQ_KEY_CHOICE, SARVAM_KEY_CHOICE

ORDER_FILE = Path(__file__).resolve().parents[1] / "key_order.yaml"

# The variable common/env.py reads for each service.
CHOICE_OF = {"gemini": GEMINI_KEY_CHOICE, "sarvam": SARVAM_KEY_CHOICE, "groq": GROQ_KEY_CHOICE}


def load_order(path: Path = ORDER_FILE) -> dict[str, list[str]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    unknown = set(raw) - set(CHOICE_OF)
    if unknown:
        raise ValueError(f"key order: unknown services {sorted(unknown)}")
    return {service: [str(name) for name in names] for service, names in raw.items()}


@dataclass
class KeyOrder:
    """`checks`: for each service this run needs, a tiny request with the chosen key;
    None when it answers, otherwise why not."""

    order: dict[str, list[str]]
    checks: dict[str, Callable[[], str | None]]
    say: Callable[[str], None] = print
    _at: dict[str, int] = field(default_factory=dict)

    def chosen(self) -> dict[str, str]:
        return {service: self.order[service][at] for service, at in self._at.items()}

    def choose(self) -> list[str]:
        """Keep each service's current key if it answers, else move down its list.
        Returns the services with no working key left (the run must stop); empty if all
        are fine. Keys passed over are never tried again in this run: a key full for
        the day stays full."""
        problems = []
        for service, check in self.checks.items():
            names = self.order.get(service) or []
            at = self._at.get(service, 0)
            while at < len(names):
                os.environ[CHOICE_OF[service]] = names[at]
                why = check()
                if why is None:
                    break
                self.say(f"key {names[at]} failed: {why[:160]}")
                at += 1
            if at >= len(names):
                problems.append(f"{service}: every key in lab/key_order.yaml failed ({', '.join(names) or 'none listed'})")
                continue
            if self._at.get(service) != at:
                self.say(f"{service}: using {names[at]}")
            self._at[service] = at
        return problems
