"""One Thinker job with the real AI (Gemini Flash-Lite) on the step prompts, against the
real backend. Uses about 4 free Gemini requests, so it runs only when asked:

    set HEARKEN_REAL_AI=1 && .venv/Scripts/python.exe -m pytest tests/agents/test_thinker_graph_real_ai.py -s
"""

from __future__ import annotations

import os
import time

import pytest

from agents.agent_c.notebook import CALLER, CHECKED_NUMBER, ORDER_NUMBER, RESULT
from agents.agent_c.thinker.step_brain import GeminiStepBrain
from common.env import gemini_key, load_env
from common.google import GOOGLE_MIN_TIMEOUT_S, google_client
from packs.food.pack import FoodPack
from test_thinker_graph import run  # pytest puts this folder on the path

pytestmark = pytest.mark.skipif(os.environ.get("HEARKEN_REAL_AI") != "1", reason="uses real Gemini requests")


class Logged:
    """The real AI, keeping each step's answer so the test can show it."""

    def __init__(self, brain) -> None:
        self.brain, self.log = brain, []

    async def ask(self, step, facts):
        started = time.perf_counter()
        try:
            answer = await self.brain.ask(step, facts)
        except BaseException as error:  # shown even when the job fails
            print(step, "-> error after", round(time.perf_counter() - started, 1), "s:", repr(error)[:200])
            raise
        print(step, "->", answer, f"({round(time.perf_counter() - started, 1)} s)")
        self.log.append((step, answer))
        return answer


async def test_a_cancel_with_the_real_ai(open_call):
    load_env()
    ai = Logged(GeminiStepBrain(google_client(gemini_key(), GOOGLE_MIN_TIMEOUT_S), "gemini-3.5-flash-lite"))
    kit = await open_call(FoodPack())
    notebook, tools, handed = await run(kit, "Haan ji, mujhe apna order cancel karna hai", ai, [
        (None, CALLER, "Order number chaar paanch do ek hai"), (None, ORDER_NUMBER, "4521"),
        (RESULT, CALLER, "Nahi bas, thank you")], ["yes"], step_s=8.0)  # decision 30
    print("notes:", [(note.kind, note.text[:70]) for note in notebook.notes])
    assert next(n.text for n in notebook.notes if n.kind == CHECKED_NUMBER) == "4521"
    assert tools.calls == ["find_order", "prepare_cancel_order", "confirm"] and handed == []
    assert (await kit.one("SELECT status FROM orders WHERE order_no = 4521"))["status"] == "cancelled"
