"""The Talker graph's AI: reads one caller turn and talks back (decided 17 Nov 2026).

Unlike the Thinker's AI, its words go to the caller, so they stream: each sentence is
handed to the voice as soon as it is written, not after the whole reply. Its exits are
tool calls ("open_job", "end_call", "heard_number"): a reply with no tool call is just a
reply. The graph says which tools a step offers; the AI never sees the graph.

Tests use a scripted fake with the same shape.
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

PROMPTS = Path(__file__).parent / "prompts" / "steps"
REPLY = "reply"  # the exit when the AI only talked

Speak = Callable[[str], Awaitable[None]]

# A sentence ends at . ? ! or the Hindi full stop; the voice gets one sentence at a time.
_SENTENCE_END = re.compile(r"(?<=[.?!।])\s+")


class TalkerAI(Protocol):
    async def reply(self, step: str, facts: dict[str, Any], exits: Sequence[str], speak: Speak) -> str:
        """Answer one caller turn. Each sentence goes to `speak` as soon as it is written.
        Returns the exit: the name of a tool it called (one of `exits`), or "reply"."""
        ...


def prompt_for(step: str) -> str:
    """The step's prompt, read from talker/prompts/steps/<step>.md."""
    return (PROMPTS / f"{step}.md").read_text(encoding="utf-8")


class GeminiTalkerAI:
    """The real AI: Gemini Flash-Lite, streamed, with the step's exits as tools."""

    def __init__(self, client: Any, model: str) -> None:
        from google.genai import types

        self._client = client
        self._model = model
        self._types = types

    async def reply(self, step: str, facts: dict[str, Any], exits: Sequence[str], speak: Speak) -> str:
        types = self._types
        tools = [types.Tool(function_declarations=[types.FunctionDeclaration(name=name, description=_TOOL_HELP[name])
                                                   for name in exits])] if exits else None
        config = types.GenerateContentConfig(
            system_instruction=prompt_for(step), temperature=1.0, tools=tools,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL))
        exit_, pending = REPLY, ""
        stream = await self._client.aio.models.generate_content_stream(
            model=self._model, contents=json.dumps(facts, ensure_ascii=False), config=config)
        async for chunk in stream:
            for part in (chunk.candidates[0].content.parts or []) if chunk.candidates and chunk.candidates[0].content else []:
                if part.function_call is not None and part.function_call.name in exits:
                    exit_ = part.function_call.name
                elif part.text:
                    # Hand over every finished sentence at once; keep the unfinished end.
                    pending += part.text
                    *done, pending = _SENTENCE_END.split(pending)
                    for sentence in done:
                        await speak(sentence.strip())
        if pending.strip():
            await speak(pending.strip())
        return exit_


# What each exit tool means, as the AI reads it.
_TOOL_HELP = {
    "open_job": "The caller needs something looked up or changed for an order, or wants a person.",
    "end_call": "The caller is done and said goodbye.",
    "heard_number": "The caller's words include their order number (in digits or in number words).",
}
