"""The action contract: the one shape every business action fills in.

The engine only ever talks to this shape, never to "orders" or "bills"
directly. That is what lets a new pack plug in with zero engine changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

from engine.storage.database import Tx

INTEGER = "integer"
NUMBER = "number"
STRING = "string"


@dataclass(frozen=True)
class Input:
    """One input of an action, as the AI sees it in the tool's schema."""

    type: str  # INTEGER, NUMBER or STRING
    description: str
    required: bool = True
    choices: tuple[str, ...] = ()  # allowed values for a STRING; empty means any


@dataclass(frozen=True)
class Check:
    """The answer of a rule check: allowed, or refused with a reason.

    The reason is written for the caller, because the agent repeats it out
    loud ("order 4521 is already out for delivery, so it can't be cancelled").
    """

    allowed: bool
    reason: str = ""

    @staticmethod
    def ok() -> Check:
        return Check(True)

    @staticmethod
    def refuse(reason: str) -> Check:
        return Check(False, reason)


@dataclass(frozen=True)
class ActionContext:
    """Facts the engine hands to `do`, so packs never read a clock or make tokens."""

    now: float  # pretend-clock seconds since the call started
    token: str | None  # the token being confirmed; None in plain mode


@dataclass(frozen=True)
class Phrase:
    """A ready phrase from the pack's spoken words, said in the caller's language.

    `fill` holds numbers for the phrase's own slots: the phrase
    "{rupees} rupaye {paise} paise" with fill {"rupees": "249", "paise": "50"}.
    Part 2 decision 10: "₹249.50" was read as a decimal by 4 of 5 voices, so
    money with paise needs words around its numbers, and the words differ by
    language; a plain value can't do that, and a fixed phrase has no numbers.
    """

    group: str
    value: str
    fill: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SpokenQuestion:
    """The yes-question before it is put into words (Part 2 decision 9).

    The action picks the line and hands over the backend's own numbers; the
    words come from the pack's spoken words file. So no AI ever writes the
    question, and the amount can never be left out.
    """

    line: str  # a key in the pack's `yes_question` group
    # Said exactly as given, in every language: order number, address.
    values: dict[str, str] = field(default_factory=dict)
    # Said as the pack's phrase, in the caller's language: payment method, reason, amount.
    phrases: dict[str, Phrase] = field(default_factory=dict)


class Action(ABC):
    """Base class for every business action ("cancel order", "pay bill", ...)."""

    name: ClassVar[str]  # tool name, e.g. "cancel_order"
    description: ClassVar[str]  # kept short: the AI reads it on every turn
    risky: ClassVar[bool] = False  # True: safe mode adds prepare/confirm, token and ticket
    inputs: ClassVar[dict[str, Input]] = {}
    # Which inputs make two requests "the same request" (decision 14). Empty
    # means all of them. Leave out free text such as names, so a retry that
    # spells the name differently still finds the earlier token.
    same_request_inputs: ClassVar[tuple[str, ...]] = ()

    @abstractmethod
    async def check(self, tx: Tx, inputs: dict[str, Any]) -> Check:
        """Business rules only. Runs at prepare and again at confirm."""

    async def summarize(self, tx: Tx, inputs: dict[str, Any]) -> str:
        """The line read to the caller before asking for a yes. Risky actions only."""
        raise NotImplementedError(f"risky action {self.name} must define summarize()")

    async def spoken_question(self, tx: Tx, inputs: dict[str, Any]) -> SpokenQuestion | None:
        """The yes-question for a risky action, at prepare time.

        `summarize` stays: it is the English line for logs and the Thinker.
        This one is what the caller hears. None for packs that never talk to
        a caller, like the toy pack.
        """
        return None

    @abstractmethod
    async def do(self, tx: Tx, inputs: dict[str, Any], ctx: ActionContext) -> dict[str, Any]:
        """The real change, returning short facts for the agent.

        The engine calls it inside the one transaction, after every check has
        passed, so `do` must never commit or check rules on its own.
        """


def clean_inputs(
    specs: dict[str, Input], raw: dict[str, Any]
) -> tuple[dict[str, Any], str | None]:
    """Check what the AI sent against the schema.

    Returns the cleaned inputs, or a problem the AI can fix and retry. A wrong
    input is refused here, before any rule runs, so rules can trust their inputs.
    """
    unknown = sorted(set(raw) - set(specs))
    if unknown:
        return {}, f"Unknown input: {', '.join(unknown)}."
    cleaned: dict[str, Any] = {}
    for name, spec in specs.items():
        value = raw.get(name)
        if value is None or (isinstance(value, str) and not value.strip()):
            if spec.required:
                return {}, f"Missing input: {name}."
            continue
        value, problem = _clean_value(name, spec, value)
        if problem:
            return {}, problem
        cleaned[name] = value
    return cleaned, None


def _clean_value(name: str, spec: Input, value: Any) -> tuple[Any, str | None]:
    # bool is a kind of int in Python, but True is never a valid order number.
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
    if spec.type == INTEGER:
        # Accept 4521.0 as 4521: some models send whole numbers as decimals.
        if is_number and float(value).is_integer():
            return int(value), None
        return None, f"{name} must be a whole number."
    if spec.type == NUMBER:
        if is_number:
            return value, None
        return None, f"{name} must be a number."
    if not isinstance(value, str):
        return None, f"{name} must be text."
    value = value.strip()
    if spec.choices and value not in spec.choices:
        return None, f"{name} must be one of: {', '.join(spec.choices)}."
    return value, None
