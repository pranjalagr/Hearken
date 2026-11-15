"""Agent C's settings: every number and service name lives in settings.yaml, with the
decision behind it; prompts are text files next to the code that uses them."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
TALKER_PROMPTS = HERE / "talker" / "prompts"


@dataclass(frozen=True)
class AgentCSettings:
    greeting: str
    talker: dict[str, Any]
    thinker: dict[str, Any]
    ears: dict[str, Any]
    voice: dict[str, Any]
    talker_prompt: str
    raw: dict[str, Any]  # the whole file, for the sections each part reads itself

    @property
    def models(self) -> dict[str, str]:
        """Kept with every run (run.json), so an old run keeps its meaning."""
        return {"talker": self.talker["model"], "thinker": self.thinker["model"], "ears": self.ears["model"],
                "voice": f"{self.voice['model']}/{self.voice['speaker']}"}

    def section(self, name: str) -> dict[str, Any]:
        return dict(self.raw.get(name) or {})


def talker_prompt(folder: Path = TALKER_PROMPTS) -> str:
    """The Talker's role, then the style guide the week-1 checks were run with."""
    return (folder / "role.md").read_text(encoding="utf-8").strip() + "\n\n" + (folder / "style.md").read_text(encoding="utf-8")


def load_settings(path: Path = HERE / "settings.yaml", prompts: Path = TALKER_PROMPTS) -> AgentCSettings:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return AgentCSettings(data["greeting"], data["talker"], data["thinker"], data["ears"], data["voice"],
                          talker_prompt(prompts), data)
