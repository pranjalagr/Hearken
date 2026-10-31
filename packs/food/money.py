"""Money helpers. Money is always whole paise inside the shop."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation


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
    """45000 -> '₹450', 24950 -> '₹249.50'. Short, because the agent says it out loud."""
    rupees, rest = divmod(paise, 100)
    return f"₹{rupees}" if rest == 0 else f"₹{rupees}.{rest:02d}"
