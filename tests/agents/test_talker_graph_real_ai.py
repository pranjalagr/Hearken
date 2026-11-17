"""One whole call with every AI real: the Talker's Flash-Lite (streamed, with its tool
exits), Sarvam's number finder, the Thinker's Flash-Lite, and the real backend. About 8
Gemini requests and 1-2 Sarvam requests, so it runs only when asked:

    set HEARKEN_REAL_AI=1 && .venv/Scripts/python.exe -m pytest tests/agents/test_talker_graph_real_ai.py -s
"""

from __future__ import annotations

import os

import pytest

from agents.agent_c.notebook import NEED_YES, RESULT
from agents.agent_c.talker.number_finder import sarvam_find
from agents.agent_c.talker.talker_ai import GeminiTalkerAI
from agents.agent_c.thinker.step_brain import GeminiStepBrain
from common.env import gemini_key, load_env, sarvam_key
from common.google import GOOGLE_MIN_TIMEOUT_S, google_client
from common.sarvam import sarvam_client
from packs.food.pack import FoodPack
from test_talker_graph import whole_call  # pytest puts this folder on the path
from test_thinker_graph_real_ai import Logged

pytestmark = pytest.mark.skipif(os.environ.get("HEARKEN_REAL_AI") != "1", reason="uses real AI requests")


class LoggedTalker:
    """The real Talker AI, printing each step's exit and what it said."""

    def __init__(self, ai) -> None:
        self.ai = ai

    async def reply(self, step, facts, exits, speak):
        said = []

        async def keep(sentence):
            said.append(sentence)
            await speak(sentence)

        exit_ = await self.ai.reply(step, facts, exits, keep)
        print("talker", step, "->", exit_, said)
        return exit_


async def test_a_whole_cancel_call_with_every_ai_real(open_call):
    load_env()
    client = google_client(gemini_key(), GOOGLE_MIN_TIMEOUT_S)
    talker_ai = LoggedTalker(GeminiTalkerAI(client, "gemini-3.5-flash-lite"))
    thinker_ai = Logged(GeminiStepBrain(client, "gemini-3.5-flash-lite"))
    finder_client = sarvam_client(sarvam_key(), 5.0)
    find = sarvam_find(finder_client, "sarvam-105b-conversations")

    async def logged_find(words):
        found = await find(words)
        print("finder", repr(words), "->", found)
        return found

    kit = await open_call(FoodPack())
    try:
        call, tools = await whole_call(kit, talker_ai, thinker_ai, logged_find, [
            (None, "Haan ji, mujhe apna order cancel karna hai. Order number chaar paanch do ek hai."),
            (NEED_YES, "Haan, kar dijiye"), (RESULT, "Nahi bas, thank you")], step_s=8.0)
    finally:
        await finder_client.aclose()
    for said in call.voice.said:
        print("said:", said)
    assert tools.calls == ["find_order", "prepare_cancel_order", "confirm"]
    assert (await kit.one("SELECT status FROM orders WHERE order_no = 4521"))["status"] == "cancelled"
    assert call.ended == ["the caller said goodbye"]
