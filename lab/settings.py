"""The lab's settings, read from lab/settings.yaml (Part 5, L14).

Strict on purpose: a missing or misspelt number fails when the lab starts,
not halfway through a batch of paid calls.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

import yaml

SETTINGS_FILE = Path(__file__).with_name("settings.yaml")


@dataclass(frozen=True)
class LabSettings:
    calls_at_once: int
    robot_ready_timeout_s: float
    robot_no_audio_s: float
    robot_max_call_s: float
    robot_change_wait_s: float
    max_reruns: int
    lab_problem_share_flag: float
    freeze_limit_ms: float
    sound_level: float
    stop_within_s: float
    wrong_stop_quiet_s: float
    busy_lab_tolerance_ms: float
    judge_timeout_s: float
    judge_gap_s: float
    reshuffle_rounds: int
    reshuffle_seed: int

    def with_changes(self, **changes: Any) -> LabSettings:
        """For tests and quick runs: the same settings with a few numbers changed."""
        return replace(self, **changes)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_settings(path: Path = SETTINGS_FILE) -> LabSettings:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    names = {field.name: field.type for field in fields(LabSettings)}
    missing, extra = set(names) - set(raw), set(raw) - set(names)
    if missing or extra:
        raise ValueError(f"lab settings: missing {sorted(missing)}, unknown {sorted(extra)}")
    values = {}
    for name, kind in names.items():
        value = raw[name]
        wanted = int if kind == "int" else float
        if isinstance(value, bool) or not isinstance(value, (int, float)) or (wanted is int and not isinstance(value, int)):
            raise ValueError(f"lab settings: {name} must be a {wanted.__name__}, got {value!r}")
        if value < 0:
            raise ValueError(f"lab settings: {name} must not be negative")
        values[name] = wanted(value)
    return LabSettings(**values)
