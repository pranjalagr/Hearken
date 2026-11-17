"""ToolCall: one tool and its inputs, as code presses it and copies its answer
(thinker/copy_notes.py, thinker/graph.py).

The rest is the old free Thinker's brain (one step = press tools or stop with a reply),
from before the two graphs of 17 Nov 2026; the Thinker graph asks its AI through
thinker/step_brain.py instead, and nothing uses GeminiThinker any more.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from engine.service.tool_catalog import ToolSpec


@dataclass(frozen=True)
class ToolCall:
    name: str
    inputs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Step:
    calls: tuple[ToolCall, ...] = ()
    text: str = ""


class ThinkerBrain(Protocol):
    async def step(self, news: Sequence[str], results: Sequence[tuple[ToolCall, dict[str, Any]]]) -> Step:
        """One step. `news`: lines that came in since the last step (the request, the
        caller's words, the Guard's readings, change results). `results`: the answers
        to the tools the last step pressed. A failed step must leave the brain as it
        was, so the same step can be tried again."""
        ...


class GeminiThinker:
    """The real brain. It keeps the conversation in Gemini's own form, so the model's
    function calls go back exactly as they came (Gemini 3 checks their signatures)."""

    def __init__(self, client: Any, model: str, system: str, tools: Sequence[ToolSpec],
                 on_usage: Callable[[int, int], None] = lambda prompt, answer: None) -> None:
        from google.genai import types

        self._client = client
        self._model = model
        self._types = types
        self._history: list[Any] = []
        self._on_usage = on_usage
        declarations = [types.FunctionDeclaration(name=t.name, description=t.description,
                                                  parameters_json_schema=_plain_schema(t.input_schema)) for t in tools]
        self._config = types.GenerateContentConfig(
            system_instruction=system, temperature=1.0,
            tools=[types.Tool(function_declarations=declarations)] if declarations else None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL))

    async def step(self, news: Sequence[str], results: Sequence[tuple[ToolCall, dict[str, Any]]]) -> Step:
        types = self._types
        parts = [types.Part.from_function_response(name=call.name, response=reply) for call, reply in results]
        parts += [types.Part.from_text(text=line) for line in news]
        asked = self._history + ([types.Content(role="user", parts=parts)] if parts else [])
        response = await self._client.aio.models.generate_content(model=self._model, contents=asked, config=self._config)
        usage = getattr(response, "usage_metadata", None)
        self._on_usage(getattr(usage, "prompt_token_count", 0) or 0, getattr(usage, "candidates_token_count", 0) or 0)
        content = response.candidates[0].content if response.candidates else None
        # Kept only once the request worked, so a retry asks the very same thing.
        self._history = asked + ([content] if content is not None else [])
        calls = tuple(ToolCall(fc.name, dict(fc.args or {})) for fc in (response.function_calls or []))
        text = "" if calls else (response.text or "").strip()
        return Step(calls, text)


def _plain_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Gemini's function format has no "additionalProperties" (the same fix as Agent A's)."""
    return {key: value for key, value in schema.items() if key != "additionalProperties"}
