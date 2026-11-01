"""Money helpers. Money is always whole paise inside the shop."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation

from engine.contract.action import Phrase


def rupees_to_paise(amount: float | int) -> int | None:
    """₹120.50 -> 12050. None when it is not a real amount (more than 2 decimals).

    Goes through Decimal, because float maths can turn 0.1 + 0.2 into
    0.30000000000000004 and a refund must never be off by a paisa.
    """
    try:
        paise = Decimal(str(amount)) * 100
    except InvalidOperation:
        return None
    if paise != paise.to_integral_value():
        return None
    return int(paise)


def format_rupees(paise: int) -> str:
    """45000 -> '₹450', 24950 -> '₹249.50'. For notes, logs and tool replies;
    a yes-question says the amount with spoken_amount instead."""
    rupees, rest = divmod(paise, 100)
    return f"₹{rupees}" if rest == 0 else f"₹{rupees}.{rest:02d}"


def spoken_amount(paise: int) -> Phrase:
    """How a yes-question says an amount, in the caller's language (Part 2
    decision 10): 45000 -> "₹450"; 24950 -> "249 rupaye 50 paise"; 50 -> "50 paise".

    "₹249.50" was read as a decimal by 4 of 5 voices in the voice test, and a
    caller must clearly hear the amount they say yes to. The words for each
    case live in spoken.yaml's `money` group."""
    rupees, rest = divmod(paise, 100)
    if rest == 0:
        return Phrase("money", "whole", {"rupees": str(rupees)})
    if rupees == 0:
        # Nobody says "0 rupaye 50 paise"; people say "50 paise".
        return Phrase("money", "only_paise", {"paise": str(rest)})
    return Phrase("money", "with_paise", {"rupees": str(rupees), "paise": str(rest)})
