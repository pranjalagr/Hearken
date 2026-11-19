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

from agents.common.ai_log import NO_LOG, AiLog, gemini_details, openai_style_details, timed

PROMPTS =Path(__file__).parent / "prompts" / "steps"
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

    def __init__(self, client: Any, model: str, key: str = "", log: AiLog | None = NO_LOG) -> None:
        from google.genai import types

        self._client = client
        self._model = model
        self._types = types
        self._key = key  # the key's .env name, for the request log; never the key
        self._log = log

    async def reply(self, step: str, facts: dict[str, Any], exits: Sequence[str], speak: Speak) -> str:
        types = self._types
        tools = [types.Tool(function_declarations=[types.FunctionDeclaration(name=name, description=_TOOL_HELP[name])
                                                   for name in exits])] if exits else None
        config = types.GenerateContentConfig(
            system_instruction=prompt_for(step), temperature=1.0, tools=tools,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL))
        exit_, pending = REPLY, ""
        async with timed(self._log, "gemini", self._model, self._key, f"talker {step}") as req:
            stream = await self._client.aio.models.generate_content_stream(
                model=self._model, contents=json.dumps(facts, ensure_ascii=False), config=config)
            last = None
            async for chunk in stream:
                if last is None:
                    req.first_piece()  # a long wait for this is the service's queue
                    req.add(gemini_details(chunk))  # the first piece carries the timing headers
                last = chunk
                for part in (chunk.candidates[0].content.parts or []) if chunk.candidates and chunk.candidates[0].content else []:
                    if part.function_call is not None and part.function_call.name in exits:
                        exit_ = part.function_call.name
                    elif part.text:
                        # Hand over every finished sentence at once; keep the unfinished end.
                        pending += part.text
                        *done, pending = _SENTENCE_END.split(pending)
                        for sentence in done:
                            async with req.our_time():  # the voice may wait for the caller: not the AI's time
                                await speak(sentence.strip())
            if last is not None:
                req.add(gemini_details(last))  # the last piece carries the tokens used
        if pending.strip():
            await speak(pending.strip())
        return exit_


class GroqTalkerAI:
    """The Talker's AI on Groq (OpenAI-style chat, the step's exits as tools), on one key.

    Added 18 Nov 2026 (the owner's choice), when Gemini Flash-Lite took 6-8 s per request on 4
    of 6 keys (Google's own server time) and Groq answered a Talker turn in ~0.6 s.

    The whole reply is read before anything is said (~0.6 s, so little is lost by not streaming
    to the voice): qwen, asked for end_call, also wrote "The caller said: ..." beside the call.
    A reply that calls an exit says nothing; the exits have their own fixed lines."""

    def __init__(self, client: Any, model: str, key: str = "", log: AiLog | None = NO_LOG,
                 extra: dict[str, Any] | None = None) -> None:
        self._client = client  # an httpx.AsyncClient on Groq, with its key in its headers
        self._model = model
        self._key = key  # the key's .env name, for the request log; never the key
        self._log = log
        self._extra = dict(extra or {})  # model options from settings.yaml (reasoning_effort)

    async def reply(self, step: str, facts: dict[str, Any], exits: Sequence[str], speak: Speak) -> str:
        import httpx

        body: dict[str, Any] = {"model": self._model, "temperature": 1.0, **self._extra,
                                "messages": [{"role": "system", "content": prompt_for(step)},
                                             {"role": "user", "content": json.dumps(facts, ensure_ascii=False)}]}
        if exits:
            body["tools"] = [{"type": "function", "function": {"name": name, "description": _TOOL_HELP[name],
                                                               "parameters": {"type": "object", "properties": {}}}}
                             for name in exits]
        async with timed(self._log, "groq", self._model, self._key, f"talker {step}") as req:
            response = await self._client.post("/chat/completions", json=body)
            req.add(openai_style_details(response))
            if response.is_error:
                req.refused(response.status_code, response.text, response.headers)
                # The status in the words, so FailoverTalkerAI sees a 429 and moves to the next key.
                raise httpx.HTTPStatusError(f"Groq said {response.status_code}: {response.text[:300]}",
                                            request=response.request, response=response)
        message = response.json()["choices"][0]["message"]
        called = [call["function"]["name"] for call in message.get("tool_calls") or []
                  if call.get("function", {}).get("name") in exits]
        if called:
            return called[0]
        for sentence in _SENTENCE_END.split((message.get("content") or "").strip()):
            if sentence.strip():
                await speak(sentence.strip())
        return REPLY


# What each exit tool means, as the AI reads it.
_TOOL_HELP = {
    "open_job": "The caller needs something looked up or changed for an order, or wants a person.",
    "end_call": "The caller is done and said goodbye.",
    "heard_number": "The caller's words include their order number (in digits or in number words).",
}


class FailoverTalkerAI:
    """The Talker's AI over several keys: its own key first, and on a refusal for quota
    (429) the next key, for the rest of the call. The owner's rule (16 Nov 2026): the first
    working key, the next on failure. 17 Nov 2026, run 6, food_04: the Talker's one key hit
    its daily quota (tiny test requests still passed the lab's key check), every reply in
    the call failed, and the caller only ever heard the safe line.

    A key is changed only if nothing was said yet, so a half-said reply is never repeated.
    """

    def __init__(self, ais: Sequence[TalkerAI], on_switch: Callable[[str], None] = lambda why: None) -> None:
        if not ais:
            raise ValueError("the Talker needs at least one key")
        self._ais = list(ais)
        self._at = 0
        self._on_switch = on_switch

    async def reply(self, step: str, facts: dict[str, Any], exits: Sequence[str], speak: Speak) -> str:
        said = False

        async def counted(sentence: str) -> None:
            nonlocal said
            said = True
            await speak(sentence)

        while True:
            try:
                return await self._ais[self._at].reply(step, facts, exits, counted)
            except Exception as error:
                if said or self._at + 1 >= len(self._ais) or not _quota_refusal(error):
                    raise
                self._at += 1
                self._on_switch(f"talker key {self._at} of {len(self._ais)} after: {str(error)[:120]}")


def _quota_refusal(error: Exception) -> bool:
    """A "too many requests / quota used up" answer: Gemini's, or Groq's 429 (its per-minute
    token limit, 8K per account and model, is reached by a busy call: the next key goes on)."""
    text = str(error)
    return "429" in text or "RESOURCE_EXHAUSTED" in text
