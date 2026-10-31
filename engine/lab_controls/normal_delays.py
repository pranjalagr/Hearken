"""The normal time a tool call takes (Part 2 decision 6), in seconds.

The backend itself answers in about 2 ms, which no real company does. At that
speed "silence while a tool runs" would look trivially short, so every tool
call first waits a fixed, realistic time. The waits live in the backend every
agent shares, so A, B and C get exactly the same ones.

These are guesses, not measurements of a real company: say so in the write-up.
0 turns a wait off, for measuring the backend's raw speed and for old tests
that check exact pretend times.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NormalDelays:
    # Reads and prepares change nothing, so a quick answer is realistic.
    lookup_s: float = 0.3
    # A risky action's save: confirm, or the action itself in plain mode.
    # The backend waits this BEFORE the save, so a "ruko" can still cancel the
    # ticket while the action is "running" (race moment 3).
    change_s: float = 0.8

    def __post_init__(self) -> None:
        if self.lookup_s < 0 or self.change_s < 0:
            raise ValueError("normal delays must be >= 0")

    def for_action(self, risky: bool) -> float:
        # `risky` already marks the actions that change money or orders, so it
        # decides the wait too: no second flag to keep in step with it. A
        # hand-over writes a row but moves no money, so it gets the short wait.
        return self.change_s if risky else self.lookup_s


NO_DELAYS = NormalDelays(lookup_s=0.0, change_s=0.0)
