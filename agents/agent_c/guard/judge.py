"""The Guard's two voices on what the caller answered (decision 21).

- The judge: Flash-Lite answers one word, under a ~1 s limit. A timeout, an error
  or any word outside the five labels is "unclear", never a yes.
- The word list: "ruko", "nahi", "mat"... on the partial words. A no or a stop needs
  only one voice, so a stop is heard the moment the ears give the word, without
  waiting for the caller to finish or for the judge.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Protocol

from agents.agent_c.live_retry import try_live

YES, NO, STOP, UNCLEAR, OTHER = "yes", "no", "stop", "unclear", "other"
LABELS = (YES, NO, STOP, UNCLEAR, OTHER)


class YesJudge(Protocol):
    async def label(self, question: str, words: str) -> str: ...


def word_list_label(words: str, stop_words: Mapping[str, str]) -> str | None:
    """"no" or "stop" if the words hold a listed word, else None."""
    found = [stop_words[word] for word in re.findall(r"[a-z]+", words.lower()) if word in stop_words]
    if STOP in found:
        return STOP  # "nahi, ruko": stopping is the stronger wish
    return found[0] if found else None


def read_label(text: str) -> str:
    """The judge's first word, if it is a label; anything else is unclear."""
    match = re.match(r"\s*([a-z]+)", (text or "").lower())
    return match.group(1) if match and match.group(1) in LABELS else UNCLEAR


class TimedJudge:
    """Any judge, held to its time limit; a timeout or a failure is "unclear"."""

    def __init__(self, ask: Callable[[str, str], Awaitable[str]], limit_s: float,
                 on_request: Callable[[], Awaitable[None]], on_error: Callable[[str], None]) -> None:
        self._ask = ask
        self._limit_s = limit_s
        self._on_request = on_request  # counts the request against the call's brain cap
        self._on_error = on_error

    async def label(self, question: str, words: str) -> str:
        await self._on_request()
        try:
            # No retry: a second try could not fit in the ~1 s the caller waits.
            return read_label(await try_live(lambda: self._ask(question, words), self._limit_s, 0))
        except Exception as error:
            self._on_error(f"judge failed, so unclear: {error!r}")
            return UNCLEAR


def first_answer(ask: Callable[[str, str], Awaitable[str]], racers: int) -> Callable[[str, str], Awaitable[str]]:
    """The same question sent `racers` times at once; the first answer wins, the rest are
    cancelled. The owner's choice (16 Nov 2026): Flash-Lite took 1-9 s per request in its
    slow spells, on every key alike, and the 2 s judge timed out three times in a call. When
    one request is stuck another often is not. With a ClientRing under `ask`, each racer goes
    out on its own key. If every racer fails, the last error is raised (the judge then says
    "unclear", never yes)."""
    if racers < 1:
        raise ValueError("at least one racer")

    async def ask_first(question: str, words: str) -> str:
        tasks = [asyncio.ensure_future(ask(question, words)) for _ in range(racers)]
        error: BaseException | None = None
        try:
            for finished in asyncio.as_completed(tasks):
                try:
                    return await finished
                except Exception as exc:  # this racer failed: wait for the next one
                    error = exc
            raise error if error is not None else RuntimeError("no judge answer")
        finally:
            for task in tasks:
                task.cancel()  # the losers: their answers are not needed

    return ask_first


def gemini_ask(client: Any, model: str, system: str) -> Callable[[str, str], Awaitable[str]]:
    """The judge on Gemini Flash-Lite: one short answer, thinking minimal."""
    from google.genai import types

    config = types.GenerateContentConfig(system_instruction=system, temperature=0.0, max_output_tokens=5,
                                         thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL))

    async def ask(question: str, words: str) -> str:
        response = await client.aio.models.generate_content(
            model=model, contents=f"Question: {question}\nCaller's answer: {words}", config=config)
        return response.text or ""

    return ask
