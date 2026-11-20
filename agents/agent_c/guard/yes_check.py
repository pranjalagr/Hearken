"""Is the caller's answer a clear yes? Asked of TypeSafe's Jev before the one-word judge
(20 Nov 2026, the owner's idea).

Why: the judge gives one label per answer, so "Haan, sahi hai. Refund 180 aur rider ka number
chahiye" had to be "other", and the yes was lost (food_10, three times); it also read "Ek minute,
phir se bataiye" as "stop" (food_09). On 24 answers from the 19 Nov run (scratchpad
jev_yes_test.py), Jev at 0.7 got 23 right and the judge 18; Jev's one miss was "जी", a word the
ears made up from noise, which the judge took as a yes too.

Two questions in one request: a clear yes (alone, or with a separate request), and whether the
caller also asked for something else, so that part is answered after the change. No answer, or
too slow, is "no answer": the Guard falls back to the judge alone, so a call never waits on Jev.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from agents.common.ai_log import NO_LOG, AiLog, timed

JEV_URL = "https://api.typesafe.ai/v1/systemone"
HERE = Path(__file__).resolve().parent / "prompts"


@dataclass(frozen=True)
class YesCheckAnswer:
    yes: float  # the chance the caller gave a clear yes to the question
    also_asks: float  # the chance they also asked for something else in the same words


class YesCheck(Protocol):
    line: float  # a yes at or above this counts (settings.yaml yes_check.line)

    async def clear_yes(self, question: str, words: str) -> YesCheckAnswer | None: ...


class JevYesCheck:
    """The check on Jev: two "noul" questions (is this statement true?) about one answer."""

    def __init__(self, client: Any, model: str, limit_s: float, line: float, key: str = "",
                 log: AiLog | None = NO_LOG, on_error: Any = lambda why: None) -> None:
        self._client = client  # an httpx.AsyncClient with the key in its headers
        self._model = model
        self._limit_s = limit_s
        self.line = line
        self._key = key  # the key's .env name, for the request log; never the key
        self._log = log
        self._on_error = on_error
        # Read once, never during a call (a file read would block the loop).
        self._questions = {name: {"type": "noul", "instructions": (HERE / f"{name}.md").read_text(encoding="utf-8").strip()}
                           for name in ("clear_yes", "also_asks")}

    async def clear_yes(self, question: str, words: str) -> YesCheckAnswer | None:
        body = {"model": self._model, "state": {"robot_asked": question, "caller_said": words},
                "questions": self._questions}
        try:
            async with timed(self._log, "jev", self._model, self._key, "yes check") as req:
                response = await asyncio.wait_for(self._client.post(JEV_URL, json=body), self._limit_s)
                if response.is_error:
                    req.refused(response.status_code, response.text, response.headers)
                    raise RuntimeError(f"Jev said {response.status_code}: {response.text[:200]}")
                answers = response.json()["answers"]
                answer = YesCheckAnswer(float(answers["clear_yes"]["noul"]), float(answers["also_asks"]["noul"]))
                req.add({"probability": round(answer.yes, 3), "also_asks": round(answer.also_asks, 3)})
                return answer
        except Exception as error:
            self._on_error(f"yes check gave no answer, so the judge decides alone: {error!r}"[:300])
            return None
