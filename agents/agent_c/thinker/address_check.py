"""Is the address "decide" wrote down the one the caller gave? (19 Nov 2026, the owner's idea)

Before an address change is pressed, a decision model reads the caller's words and the
address, and gives the probability that it is the same address: same numbers and place names,
nothing left out, nothing added. Under the line in settings.yaml, the change isn't prepared.

Why a decision model, not code alone: code can see the order number inside an address
("45 30 House 74 ...", thinker/graph.py _holds_order_number), but not a missing part
("House 74, Sector 15" when the caller also said "Gurgaon"). On 52 cases from our calls
(scratchpad jev_address_test.py, 19 Nov 2026), TypeSafe's Jev caught 30 of 33 wrong addresses
under 0.5, every missing part among them, and refused 4 of 19 right ones (3 of them spellings
made up for the test that the ears never heard). It missed one with the order number in front
(0.81), which the code check catches: both run.

It never blocks a change by failing: no answer in time, or an error, is "no answer" (None),
and the change goes on as before the check existed; the yes-question still asks the caller.
One request, its own time limit, no retries (the project's rule for outside services).
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from agents.common.ai_log import NO_LOG, AiLog, timed

PROMPT = Path(__file__).parent / "prompts" / "address_check.md"
JEV_URL = "https://api.typesafe.ai/v1/systemone"


class AddressCheck(Protocol):
    async def same_address(self, caller_words: Sequence[str], address: str, order_number: str) -> float | None:
        """The probability (0-1) that `address` is the address the caller gave, or None when
        no answer came."""
        ...


class JevAddressCheck:
    """The check on TypeSafe's Jev, a "noul" question: is this statement true?"""

    def __init__(self, client: Any, model: str, limit_s: float, key: str = "", log: AiLog | None = NO_LOG,
                 on_error: Any = lambda why: None) -> None:
        self._client = client  # an httpx.AsyncClient with the key in its headers
        self._model = model
        self._limit_s = limit_s
        self._key = key  # the key's .env name, for the request log; never the key
        self._log = log
        self._on_error = on_error
        self._instruction = PROMPT.read_text(encoding="utf-8").strip()  # read once, never during a call

    async def same_address(self, caller_words: Sequence[str], address: str, order_number: str) -> float | None:
        body = {"model": self._model,
                "state": {"caller_said": list(caller_words), "order_number": order_number,
                          "robot_wrote_address": address},
                "questions": {"same_address": {"type": "noul", "instructions": self._instruction}}}
        try:
            async with timed(self._log, "jev", self._model, self._key, "address check") as req:
                response = await asyncio.wait_for(self._client.post(JEV_URL, json=body), self._limit_s)
                if response.is_error:
                    req.refused(response.status_code, response.text, response.headers)
                    raise RuntimeError(f"Jev said {response.status_code}: {response.text[:200]}")
                probability = float(response.json()["answers"]["same_address"]["noul"])
                req.add({"probability": round(probability, 3)})
                return probability
        except Exception as error:
            self._on_error(f"address check gave no answer, so the change goes on unchecked: {error!r}"[:300])
            return None
