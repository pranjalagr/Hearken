"""Asking again when the caller's order can't be identified (decision 27), run by code.

On 16 Nov 2026 Agent C's Thinker handed callers over after one "No order found", asked
for the name when the number was wrong, and counted its own guessed name ("customer")
as asking. A prompt rule did not hold, so code takes these steps itself:

    number not found:  ask again -> digit by digit -> type it on the keypad -> hand over
    name not matching: ask again -> ask once more -> hand over

The backend marks these refusals with a fixed tag (packs/food/actions.py), so code
never reads the caller-facing words to know which it was. A typed number can't be
misheard: the keypad step catches the ears that heard "4533" as "45 35" three times.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz

ORDER_NOT_FOUND = "order_not_found"  # the backend's tags (packs/food/actions.py)
NAME_MISMATCH = "name_mismatch"
NAME_UNHEARD = "name_unheard"  # code's own: the Thinker used a name the caller never said

# The fixed lines, in order, by what went wrong (keys of the "ladder" group in
# talker/words.yaml). Past the last, a hand-over is allowed.
STEPS = {
    ORDER_NOT_FOUND: ("number_again", "number_digits", "number_keypad"),
    NAME_MISMATCH: ("name_again", "name_once_more"),
    NAME_UNHEARD: ("name_again", "name_once_more"),
}
KEYPAD_STEP = "number_keypad"
KEYPAD_END = "#"


@dataclass
class OrderLadder:
    digits: int  # an order number's length: the keypad stops listening after this many
    misses: dict[str, int] = field(default_factory=dict)
    waiting_keys: bool = False  # the caller was asked to type the number
    _typed: str = ""

    def miss(self, problem: str) -> str | None:
        """The next line to say for this miss, or None once the steps are used up (or the
        problem is not one this ladder knows). Both name tags count as one kind."""
        kind = NAME_MISMATCH if problem == NAME_UNHEARD else problem
        steps = STEPS.get(kind)
        if steps is None:
            return None
        self.misses[kind] = self.misses.get(kind, 0) + 1
        if self.misses[kind] > len(steps):
            return None
        step = steps[self.misses[kind] - 1]
        self.waiting_keys, self._typed = step == KEYPAD_STEP, ""
        return step

    def key(self, digit: str) -> str | None:
        """A key the caller pressed while asked to type the number. Returns the typed
        number once it is complete (the right length, or ended with #), else None."""
        if not self.waiting_keys:
            return None
        if digit == KEYPAD_END:
            number = self._typed
        elif digit.isdigit():
            self._typed += digit
            number = self._typed if len(self._typed) >= self.digits else None
        else:
            return None
        if number:
            self.waiting_keys, self._typed = False, ""
            return number
        return None

    def stop_waiting(self) -> bool:
        """The caller typed nothing in time. True if the ladder was still waiting."""
        waiting, self.waiting_keys, self._typed = self.waiting_keys, False, ""
        return waiting


def caller_said(name: str, caller_words: str, min_ratio: int) -> bool:
    """Was this name (every word of it) in the caller's words? Fuzzy per word, because the
    ears spell names their own way ("Vikram Ro" for "Vikram Rao"). A guess like "customer"
    fails: the caller never said it. No list of names, no AI."""
    heard = _words(caller_words)
    wanted = _words(name)
    return bool(wanted) and all(any(fuzz.ratio(word, said) >= min_ratio for said in heard) for word in wanted)


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())
