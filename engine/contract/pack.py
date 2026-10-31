"""What every pack must provide to plug into the engine."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from engine.contract.action import Action, Check
from engine.contract.spoken import SpokenWords
from engine.storage.database import Tx


class Pack(ABC):
    """One kind of business: its tables, starting data, actions and timeline rules."""

    name: ClassVar[str]

    @abstractmethod
    def actions(self) -> list[Action]:
        """Every action the agent may use in this business."""

    @abstractmethod
    def schema_sql(self) -> str:
        """The pack's own tables (CREATE TABLE ... STRICT)."""

    @abstractmethod
    async def load_seed(self, tx: Tx) -> None:
        """Insert the starting data that every call begins with."""

    @abstractmethod
    async def apply_change(self, tx: Tx, target: str, change: str) -> Check:
        """Apply one planned timeline change (the pretend kitchen and rider).

        Refuse it when the pack's rules forbid it, for example a cancelled order
        is never sent out for delivery. The engine then marks it skipped.
        """

    def spoken_words(self) -> SpokenWords:
        """Ready phrases for the pack's fixed values (statuses, reasons...).

        Not abstract: a pack that never talks to a caller, like the toy pack,
        has none, and asking it for a phrase fails loudly.
        """
        return SpokenWords()
