"""How long tokens and tickets live (decision 6), in pretend-clock seconds.

These are starting guesses. Tune them from the logged gaps after practice
calls: set each one a bit above what 99 out of 100 calls need.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Lifetimes:
    # prepare -> confirm. Long enough for the agent to read the summary and the
    # caller to think or ask a side question.
    prepared_s: float = 120.0
    # The Guard hears a real yes -> confirm. Confirm follows within one AI
    # turn, so this is short: an old yes must not approve anything later.
    ticket_s: float = 15.0
    # A done token never expires during the call. Forgetting it would let a
    # late retry act twice, so it has no setting here on purpose.
