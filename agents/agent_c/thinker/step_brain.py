"""The Thinker graph's AI: one question per graph step, one JSON answer back.

The graph (thinker/graph.py) decides which step runs and which exits that step has.
The AI only fills in the step's answer: read the caller's words, pick one of the allowed
exits, and give the details. It never presses a tool: code does that with what it said.

Tests use a scripted fake with the same shape, so they need no key and no quota.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

# One short prompt per AI step, each in its own file, so a prompt can be read on its own.
PROMPTS = Path(__file__).parent / "prompts" / "steps"


class StepBrain(Protocol):
    async def ask(self, step: str, facts: dict[str, Any]) -> dict[str, Any]:
        """Ask the AI about one step. `facts` is everything it may use. The answer is a
        dict, and it always has "exit": which way the graph goes next."""
        ...


def prompt_for(step: str) -> str:
    """The step's prompt, read from prompts/steps/<step>.md."""
    return (PROMPTS / f"{step}.md").read_text(encoding="utf-8")


class GeminiStepBrain:
    """The real AI: Gemini Flash-Lite, asked to answer in JSON only.

    Each step's question stands alone (the facts carry the whole call, decision 15), so
    there is no history to keep, and a failed request can simply be asked again."""

    def __init__(self, client: Any, model: str) -> None:
        from google.genai import types

        self._client = client
        self._model = model
        self._types = types

    async def ask(self, step: str, facts: dict[str, Any]) -> dict[str, Any]:
        # The step's prompt is the instruction; the facts go in as JSON text.
        types = self._types
        config = types.GenerateContentConfig(
            system_instruction=prompt_for(step), temperature=1.0, response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL))
        response = await self._client.aio.models.generate_content(
            model=self._model, contents=json.dumps(facts, ensure_ascii=False), config=config)
        answer = json.loads(response.text or "{}")
        # Anything but a JSON object counts as no answer: the graph then asks again.
        return answer if isinstance(answer, dict) else {}
