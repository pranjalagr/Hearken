"""The caller's fixed lines (Part 3, C13, C17): opening lines, race lines and the
director's own, in the caller's voice, made once and reused.

The director asks for a line inside the 20 ms beat, at the exact moment it matters,
so every line it can ask for is loaded before the call is placed.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import yaml

from lab.caller.mouth.number_words import speak_numbers_letters
from lab.caller.mouth.voice import Voice

LINES_FILE = Path(__file__).with_name("fixed_lines.yaml")


def load_fixed_lines(path: Path = LINES_FILE) -> dict[str, str]:
    return {str(line): str(hindi) for line, hindi in yaml.safe_load(path.read_text(encoding="utf-8")).items()}


class FixedLines:
    """Recordings for the director, in the caller's own voice (C4)."""

    def __init__(self, voice: Voice, number_style: str, table: dict[str, str] | None = None) -> None:
        self._voice = voice
        self._style = number_style
        self._table = table if table is not None else load_fixed_lines()
        self._made: dict[str, list[bytes]] = {}

    def spoken(self, line: str) -> str:
        """What the voice reads for a fixed line. A line missing from the table stops
        the call before it starts: it would reach the voice in letters it reads wrong."""
        if line not in self._table:
            raise ValueError(f"no Hindi-letter version of the fixed line {line!r}: add it to {LINES_FILE.name}")
        return speak_numbers_letters(self._table[line], self._style)

    async def make(self, lines: Iterable[str]) -> None:
        for line in lines:
            if line not in self._made:
                self._made[line] = await self._voice.pieces(self.spoken(line))

    def pieces(self, line: str) -> list[bytes]:
        return self._made[line]
