"""Agent C's settings: every number and service name lives in settings.yaml, with the
decision behind it; prompts are text files next to the code that uses them (each graph
step reads its own, talker/prompts/steps and thinker/prompts/steps)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class AgentCSettings:
    greeting: str
    talker: dict[str, Any]
    thinker: dict[str, Any]
    ears: dict[str, Any]
    voice: dict[str, Any]
    raw: dict[str, Any]  # the whole file, for the sections each part reads itself

    @property
    def models(self) -> dict[str, str]:
        """Kept with every run (run.json), so an old run keeps its meaning."""
        models = {"talker": self.talker["model"], "thinker": self.thinker["model"], "ears": self.ears["model"],
                  "voice": f"{self.voice['model']}/{self.voice['speaker']}"}
        if self.thinker.get("decide_model"):
            models["thinker_decide"] = self.thinker["decide_model"]  # on Groq since 17 Nov 2026
        judge = (self.raw.get("guard") or {}).get("judge_model")
        if judge:
            models["yes_judge"] = judge  # on Groq since 18 Nov 2026, as the Talker
        check = self.raw.get("address_check") or {}
        if check.get("model"):
            models["address_check"] = f"{check.get('provider', '')} {check['model']}"  # 19 Nov 2026
        yes_check = self.raw.get("yes_check") or {}
        if yes_check.get("model"):
            models["yes_check"] = f"{yes_check.get('provider', '')} {yes_check['model']}"  # 20 Nov 2026
        return models

    def section(self, name: str) -> dict[str, Any]:
        return dict(self.raw.get(name) or {})


def load_settings(path: Path = HERE / "settings.yaml") -> AgentCSettings:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return AgentCSettings(data["greeting"], data["talker"], data["thinker"], data["ears"], data["voice"], data)
