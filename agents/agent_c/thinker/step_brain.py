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

from agents.common.ai_log import NO_LOG, AiLog, gemini_details, openai_style_details, timed
from common.google import take_client

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

    def __init__(self, client: Any, model: str, log: AiLog | None = NO_LOG) -> None:
        from google.genai import types

        self._client = client  # a ClientRing (one key per request) or one client
        self._model = model
        self._types = types
        self._log = log  # one diary row per request (agents/common/ai_log.py)

    async def ask(self, step: str, facts: dict[str, Any]) -> dict[str, Any]:
        # The step's prompt is the instruction; the facts go in as JSON text.
        types = self._types
        config = types.GenerateContentConfig(
            system_instruction=prompt_for(step), temperature=1.0, response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL))
        key, client = take_client(self._client)
        async with timed(self._log, "gemini", self._model, key, f"thinker {step}") as req:
            response = await client.aio.models.generate_content(
                model=self._model, contents=json.dumps(facts, ensure_ascii=False), config=config)
            req.add(gemini_details(response))
        answer = json.loads(response.text or "{}")
        # Anything but a JSON object counts as no answer: the graph then asks again.
        return answer if isinstance(answer, dict) else {}


class GroqStepBrain:
    """A step's AI on Groq (OpenAI-style chat, JSON answers), over several keys: the first
    key, and on a refusal for its limits (429) the next, for the rest of the call (the
    owner's rule, 16 Nov 2026: the first working key, the next on failure).

    Used for "decide" since 17 Nov 2026 (the owner's choice): replaying run 7c's failed
    decide steps, Flash-Lite made the order number "45 30" the new address 3 of 3 times and
    said "Veg Mojo is not in your order" (speech-to-text for Veg Momos) 3 of 3 times;
    openai/gpt-oss-120b asked for the address and refunded the Veg Momos, 3 of 3 each, in
    about 1 s and 1,500 tokens.
    """

    def __init__(self, clients: list[Any], model: str, on_switch: Any = lambda why: None,
                 names: list[str] | None = None, log: AiLog | None = NO_LOG) -> None:
        if not clients:
            raise ValueError("a Groq brain needs at least one key")
        self._clients = clients
        self._model = model
        self._at = 0
        self._on_switch = on_switch
        self._names = list(names or [])  # each key's .env name, for the request log
        self._log = log

    async def ask(self, step: str, facts: dict[str, Any]) -> dict[str, Any]:
        import httpx

        body = {"model": self._model, "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": prompt_for(step)},
                             {"role": "user", "content": json.dumps(facts, ensure_ascii=False)}]}
        while True:
            key = self._names[self._at] if self._at < len(self._names) else ""
            async with timed(self._log, "groq", self._model, key, f"thinker {step}") as req:
                response = await self._clients[self._at].post("/chat/completions", json=body)
                req.add(openai_style_details(response))
                if response.is_error:
                    req.refused(response.status_code, response.text, response.headers)
            if response.status_code == 429 and self._at + 1 < len(self._clients):
                self._at += 1
                self._on_switch(f"groq key {self._at + 1} of {len(self._clients)} after: {response.text[:120]}")
                continue
            if response.is_error:
                raise httpx.HTTPStatusError(f"Groq said {response.status_code}: {response.text[:300]}",
                                            request=response.request, response=response)
            content = response.json()["choices"][0]["message"].get("content") or "{}"
            answer = json.loads(content)  # not JSON: a ValueError, asked again by the graph
            return answer if isinstance(answer, dict) else {}


class StepRouter:
    """Each step on its own AI: the named steps on theirs, every other step on the default
    (17 Nov 2026: "decide" on Groq gpt-oss-120b, the small steps stay on Flash-Lite)."""

    def __init__(self, default: StepBrain, by_step: dict[str, StepBrain]) -> None:
        self._default = default
        self._by_step = by_step

    async def ask(self, step: str, facts: dict[str, Any]) -> dict[str, Any]:
        return await self._by_step.get(step, self._default).ask(step, facts)
