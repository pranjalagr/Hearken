"""The pretend caller's settings, read from settings.yaml (Part 3, C18).

Strict, like the lab's: a missing or misspelt name fails when the lab starts, not
in the middle of a batch of paid calls.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from engine.tasks.caller_spec import PERSONALITIES, VOICES

SETTINGS_FILE = Path(__file__).with_name("settings.yaml")


@dataclass(frozen=True)
class VoiceChoice:
    voice: str
    model: str


@dataclass(frozen=True)
class CallerSettings:
    brain_company: str
    brain_model: str
    brain_timeout_s: float
    ears_company: str
    voices: dict[str, VoiceChoice]
    angry_voices: dict[str, VoiceChoice]
    robot_done_s: float
    extra_gap_s: dict[str, float]
    think_after_quiet_s: float
    reply_stuck_s: float
    late_words_s: float
    opening_wait_s: float
    heard_nothing_after_s: float
    give_up_same_question: int
    give_up_s: dict[str, float]
    goodbye_wait_s: float
    robot_sound_level: float
    price_per_10k_characters: dict[str, float]
    ears_price_per_minute: dict[str, float]

    def voice_for(self, label: str, personality: str) -> VoiceChoice:
        """C4: one voice per caller for the whole call, fitted to its personality."""
        if personality == "angry" and label in self.angry_voices:
            return self.angry_voices[label]
        return self.voices[label]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_caller_settings(path: Path = SETTINGS_FILE) -> CallerSettings:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    names = {field.name for field in fields(CallerSettings)}
    missing, extra = names - set(raw), set(raw) - names
    if missing or extra:
        raise ValueError(f"caller settings: missing {sorted(missing)}, unknown {sorted(extra)}")
    values = dict(raw)
    values["voices"] = {label: VoiceChoice(**choice) for label, choice in raw["voices"].items()}
    values["angry_voices"] = {label: VoiceChoice(**choice) for label, choice in raw["angry_voices"].items()}
    if set(values["voices"]) != set(VOICES):
        raise ValueError(f"caller settings: voices must name exactly {sorted(VOICES)}")
    for name in ("extra_gap_s", "give_up_s"):
        if set(raw[name]) != set(PERSONALITIES):
            raise ValueError(f"caller settings: {name} must name exactly {sorted(PERSONALITIES)}")
    return CallerSettings(**values)
