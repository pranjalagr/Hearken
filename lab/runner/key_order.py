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

from common.env import GEMINI_KEY_CHOICE, GEMINI_POOL_CHOICE, GROQ_KEY_CHOICE, GROQ_POOL_CHOICE, SARVAM_KEY_CHOICE

ORDER_FILE = Path(__file__).resolve().parents[1] / "key_order.yaml"

# The variable common/env.py reads for each service.
CHOICE_OF = {"gemini": GEMINI_KEY_CHOICE, "sarvam": SARVAM_KEY_CHOICE, "groq": GROQ_KEY_CHOICE}
# Services whose robots can take turns over several keys, and the variable naming them all.
POOL_OF = {"gemini": GEMINI_POOL_CHOICE}
# Single-key services whose whole list, the chosen key first, is also handed to robots that
# move to the next key on a refusal (Agent C's Groq "decide" step, 17 Nov 2026). Keys the
# caller used up stay in, at the end: Groq counts limits per model, so a key full for the
# caller's model may still serve the robot's.
LIST_OF = {"groq": GROQ_POOL_CHOICE}


def load_order(path: Path = ORDER_FILE) -> tuple[dict[str, list[str]], set[str]]:
    """The key names per service, and the services that rotate over all their keys."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rotate = set(raw.pop("rotate", None) or [])
    unknown = (set(raw) - set(CHOICE_OF)) | (rotate - set(POOL_OF))
    if unknown:
        raise ValueError(f"key order: unknown services {sorted(unknown)}")
    return {service: [str(name) for name in names] for service, names in raw.items()}, rotate


@dataclass
class KeyOrder:
    """`checks`: for each service this run needs, a tiny request with the chosen key;
    None when it answers, otherwise why not."""

    order: dict[str, list[str]]
    checks: dict[str, Callable[[], str | None]]
    say: Callable[[str], None] = print
    rotate: set[str] = field(default_factory=set)  # services whose robots take turns over every working key
    _at: dict[str, int] = field(default_factory=dict)
    _pool: dict[str, list[str]] = field(default_factory=dict)  # the working keys of a rotated service
    _dropped: set[str] = field(default_factory=set)  # keys that failed: never tried again this run

    def chosen(self) -> dict[str, str]:
        chosen = {service: self.order[service][at] for service, at in self._at.items()}
        chosen.update({service: ",".join(pool) for service, pool in self._pool.items()})
        return chosen

    def drop(self, service: str) -> None:
        """The service's current key failed in a call though its tiny check passes: a Groq
        key with 199,307 of its 200,000 daily tokens used still answers a 1-token check
        (16 Nov 2026), so the lab kept it and every rerun failed. Move past it."""
        if service in self._at:
            name = self.order[service][self._at[service]]
            self._dropped.add(name)
            self.say(f"key {name} failed in a call: dropped")
            self._at[service] += 1

    def choose(self) -> list[str]:
        problems = self._choose_rotated()
        return problems + self._choose_single()

    def _choose_rotated(self) -> list[str]:
        """Every key of the service that still answers goes into its pool; the first is also
        the single chosen key (Agent C's Talker, and any robot that holds one key)."""
        problems = []
        for service in [s for s in self.checks if s in self.rotate]:
            working = []
            for name in self.order.get(service) or []:
                if name in self._dropped:
                    continue
                os.environ[CHOICE_OF[service]] = name
                why = self.checks[service]()
                if why is None:
                    working.append(name)
                else:
                    self._dropped.add(name)
                    self.say(f"key {name} failed: {why[:160]}")
            if not working:
                problems.append(f"{service}: every key in lab/key_order.yaml failed")
                continue
            os.environ[CHOICE_OF[service]] = working[0]
            os.environ[POOL_OF[service]] = ",".join(working)
            if self._pool.get(service) != working:
                self.say(f"{service}: taking turns over {', '.join(working)}")
            self._pool[service] = working
        return problems

    def _choose_single(self) -> list[str]:
        """Keep each service's current key if it answers, else move down its list.
        Returns the services with no working key left (the run must stop); empty if all
        are fine. Keys passed over are never tried again in this run: a key full for
        the day stays full."""
        problems = []
        for service, check in self.checks.items():
            if service in self.rotate:
                continue
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
            if service in LIST_OF:
                os.environ[LIST_OF[service]] = ",".join(names[at:] + names[:at])
        return problems
