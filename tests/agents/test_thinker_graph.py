"""The Thinker graph, end to end against the real food backend (17 Nov 2026).

The AI is a script (no key, no quota), the caller is a script that writes notes when a
given note appears, and a stand-in Guard takes the yes: it gets a real yes-ticket and
presses the real confirm. So each test runs a whole job: the number, the lookup, the
decision, the plan, the tool, the yes, the result, and what the caller says after.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from agents.agent_c.notebook import (
    ANSWER,
    ASKS_USED_UP,
    CALLER,
    CHECKED_NUMBER,
    FACT,
    NEED_INFO,
    NEED_YES,
    ORDER_NUMBER,
    PLAN,
    REQUEST,
    RESULT,
    YES_ANSWER,
    Notebook,
)
from agents.agent_c.thinker.graph import SAVER, THINKER_GRAPH, ThinkerParts, ThinkerStuck, run_job
from engine.service.tool_catalog import build_catalog
from packs.food.pack import FoodPack


class ScriptedAI:
    """Answers each step from a list, in order, and keeps what it was asked."""

    def __init__(self, **answers: list[dict[str, Any]]) -> None:
        self.answers = {step: list(given) for step, given in answers.items()}
        self.asked: list[tuple[str, dict[str, Any]]] = []

    async def ask(self, step: str, facts: dict[str, Any]) -> dict[str, Any]:
        self.asked.append((step, facts))
        return self.answers[step].pop(0)


class BackendTools:
    """The real backend, counting every press."""

    def __init__(self, backend) -> None:
        self.backend = backend
        self.calls: list[str] = []

    async def call(self, name: str, inputs: dict[str, Any]):
        self.calls.append(name)
        return await self.backend.call_tool(name, inputs)


async def _after(notebook: Notebook, kind: str, seen: int) -> int:
    """Wait for the next note of this kind after `seen`; return its number."""
    while True:
        hit = next((note for note in notebook.after(seen) if note.kind == kind), None)
        if hit is not None:
            return hit.number
        await notebook.wait(1.0)


async def _caller(notebook: Notebook, script: list[tuple[str | None, str, str]]) -> None:
    """The caller and the Talker's code: when the named note appears, write the next one."""
    seen = 0
    for wait_for, kind, text in script:
        if wait_for is not None:
            seen = await _after(notebook, wait_for, seen)
        data = {"digits": text} if kind == ORDER_NUMBER else {}
        notebook.add(kind, text, 1, data)


async def _guard(kit, tools: BackendTools, notebook: Notebook, answers: list[Any]) -> None:
    """Stands in for the Guard: for each yes-question, the caller's answers in turn."""
    seen = 0
    while answers:
        seen = await _after(notebook, NEED_YES, seen)
        token = notebook.notes[seen - 1].data["data"]["token"]
        while answers:
            answer = answers.pop(0)
            if isinstance(answer, tuple):  # ("other", words): asked something instead
                notebook.add(YES_ANSWER, f"other: {answer[1]}", 1)
                await asyncio.sleep(0.01)
                continue
            if answer == "yes":
                notebook.add(YES_ANSWER, "yes: haan", 1)
                ticket = await kit.guard.issue_ticket(token)
                reply = await tools.call("confirm", {"token": token, "ticket": ticket})
                notebook.add(RESULT, f"done: {reply.status}", 1, {"outcome": "done"})
            else:
                notebook.add(YES_ANSWER, "no: nahi", 1)
                notebook.add(RESULT, "stopped", 1, {"outcome": "stopped"})
            break


async def run(kit, request: str, ai: ScriptedAI, caller: list, guard: list | None = None, step_s: float = 2.0):
    notebook = Notebook(max_notes=200)
    notebook.add(REQUEST, request, 1)
    tools = BackendTools(kit.backend)
    handed: list[str] = []

    async def hand_over(reason: str) -> None:
        handed.append(reason)

    parts = ThinkerParts(notebook, tools, ai, hand_over, build_catalog(FoodPack().actions(), kit.settings), step_s=step_s)
    helpers = [asyncio.create_task(_caller(notebook, caller)),
               asyncio.create_task(_guard(kit, tools, notebook, list(guard or [])))]
    try:
        await asyncio.wait_for(run_job(THINKER_GRAPH, parts, f"job-{id(notebook)}"), 10 + 10 * step_s)
    finally:
        for task in helpers:
            task.cancel()
    return notebook, tools, handed


def kinds(notebook: Notebook) -> list[str]:
    return [note.kind for note in notebook.notes]


def ok(number: str) -> dict[str, Any]:
    return {"exit": "ok", "number": number}


async def test_a_cancel_runs_from_the_number_to_the_result(open_call):
    kit = await open_call(FoodPack())
    ai = ScriptedAI(check_number=[ok("4521")], work_left=[{"exit": "done"}],
                    decide=[{"exit": "prepare", "tool": "prepare_cancel_order", "inputs": {}, "plan": "Cancel kar raha hoon."}])
    notebook, tools, handed = await run(kit, "Mujhe order cancel karna hai", ai,
                                        [(None, ORDER_NUMBER, "4521"), (RESULT, CALLER, "Nahi bas, thank you")], ["yes"])
    assert tools.calls == ["find_order", "prepare_cancel_order", "confirm"]  # each pressed once, despite waking
    said = [k for k in kinds(notebook) if k in (CHECKED_NUMBER, FACT, PLAN, NEED_YES, YES_ANSWER, RESULT)]
    assert said == [CHECKED_NUMBER, FACT, PLAN, NEED_YES, YES_ANSWER, RESULT]  # the plan before the tool
    assert (await kit.one("SELECT status FROM orders WHERE order_no = 4521"))["status"] == "cancelled"
    assert handed == [] and not SAVER.storage  # the job's saved steps are gone


async def test_not_found_waits_for_the_talker_to_ask_again(open_call):
    kit = await open_call(FoodPack())
    ai = ScriptedAI(check_number=[ok("4599"), ok("4521")], decide=[{"exit": "lookup"}], work_left=[{"exit": "done"}])
    notebook, tools, _ = await run(kit, "Order kahan hai?", ai, [
        (None, ORDER_NUMBER, "4599"), (FACT, ORDER_NUMBER, "4521"), (None, CALLER, "Theek hai, bas")])
    assert tools.calls == ["find_order", "find_order"]
    missed = next(note for note in notebook.notes if note.kind == FACT)
    assert missed.data["data"] == {"wrong_input": "order_no"}  # how the Talker knows to ask again


async def test_asks_used_up_hands_over(open_call):
    kit = await open_call(FoodPack())
    ai = ScriptedAI(check_number=[ok("4599")])
    _, tools, handed = await run(kit, "Cancel karna hai", ai, [(None, ORDER_NUMBER, "4599"), (FACT, ASKS_USED_UP, "")])
    assert tools.calls == ["find_order"] and len(handed) == 1


async def test_an_unclear_number_is_asked_again(open_call):
    kit = await open_call(FoodPack())
    ai = ScriptedAI(check_number=[{"exit": "unclear"}, ok("4521")], decide=[{"exit": "lookup"}],
                    work_left=[{"exit": "done"}])
    notebook, tools, _ = await run(kit, "Status batao", ai, [
        (None, ORDER_NUMBER, "4521"), (NEED_INFO, ORDER_NUMBER, "4521"), (FACT, CALLER, "bas")])
    assert NEED_INFO in kinds(notebook) and tools.calls == ["find_order"]


async def test_a_missing_detail_is_asked_then_the_change_goes_ahead(open_call):
    kit = await open_call(FoodPack())
    address = "Flat 2C, Rose Apartments, Model Town, Ludhiana"
    ai = ScriptedAI(check_number=[ok("4561")], work_left=[{"exit": "done"}], decide=[
        {"exit": "missing", "need": "new_address", "ask": "Naya address kya hai?"},
        {"exit": "prepare", "tool": "prepare_change_address", "inputs": {"new_address": address}, "plan": "Address badal raha hoon."}])
    notebook, tools, _ = await run(kit, "Address change karna hai", ai, [
        (None, ORDER_NUMBER, "4561"), (NEED_INFO, CALLER, address), (RESULT, CALLER, "Bas, thanks")], ["yes"])
    assert tools.calls == ["find_order", "prepare_change_address", "confirm"]
    assert (await kit.one("SELECT address FROM orders WHERE order_no = 4561"))["address"] == address


async def test_a_refund_names_the_item_and_the_backend_fills_the_price(open_call):
    kit = await open_call(FoodPack())
    ai = ScriptedAI(check_number=[ok("4512")], work_left=[{"exit": "done"}], decide=[
        {"exit": "prepare", "tool": "prepare_refund_order", "inputs": {"item": "Raita", "reason": "missing_item"},
         "plan": "Raita ka refund dekh raha hoon."}])
    await run(kit, "Raita missing tha", ai, [(None, ORDER_NUMBER, "4512"), (RESULT, CALLER, "bas")], ["yes"])
    assert (await kit.one("SELECT amount_paise FROM refunds WHERE order_no = 4512 AND at >= 0"))["amount_paise"] == 4000


async def test_a_wrong_item_goes_back_to_decide_with_the_reason(open_call):
    kit = await open_call(FoodPack())
    ai = ScriptedAI(check_number=[ok("4512")], work_left=[{"exit": "done"}], decide=[
        {"exit": "prepare", "tool": "prepare_refund_order", "inputs": {"item": "Pizza", "reason": "missing_item"}},
        {"exit": "prepare", "tool": "prepare_refund_order", "inputs": {"item": "Raita", "reason": "missing_item"}}])
    await run(kit, "Item missing tha", ai, [(None, ORDER_NUMBER, "4512"), (RESULT, CALLER, "bas")], ["yes"])
    second = [facts for step, facts in ai.asked if step == "decide"][1]
    assert second["last_refusal"] == "Order 4512 has no item 'Pizza'."


async def test_a_refusal_is_explained_and_the_caller_can_accept_it(open_call):
    kit = await open_call(FoodPack())
    ai = ScriptedAI(check_number=[ok("4512")], after_explain=[{"exit": "accepts"}],
                    explain=[{"exit": "note", "text": "Yeh order deliver ho chuka hai, cancel nahi ho sakta."}],
                    decide=[{"exit": "prepare", "tool": "prepare_cancel_order", "inputs": {}}])
    notebook, tools, handed = await run(kit, "Cancel karna hai", ai, [(None, ORDER_NUMBER, "4512"),
                                                                     (ANSWER, CALLER, "Theek hai")])
    assert tools.calls == ["find_order", "prepare_cancel_order"] and handed == []
    assert notebook.notes[-2].kind == ANSWER  # the explanation, then the caller's reply


async def test_no_tool_can_do_it_then_the_caller_wants_a_person(open_call):
    kit = await open_call(FoodPack())
    ai = ScriptedAI(check_number=[ok("4568")], after_explain=[{"exit": "person"}],
                    decide=[{"exit": "cannot", "reason": "the rider's phone number is not something this line gives"}],
                    explain=[{"exit": "note", "text": "Rider ka number main nahi de sakta. Person se connect karun?"}])
    _, tools, handed = await run(kit, "Rider ka number do", ai, [(None, ORDER_NUMBER, "4568"), (ANSWER, CALLER, "Haan karo")])
    assert tools.calls == ["find_order"] and handed == ["the caller asked for a person"]


async def test_a_person_asked_for_while_the_number_is_collected(open_call):
    kit = await open_call(FoodPack())
    ai = ScriptedAI(person=[{"exit": "person"}])
    _, tools, handed = await run(kit, "Cancel karna hai", ai, [(None, CALLER, "Mujhe kisi insaan se baat karni hai")])
    assert tools.calls == [] and handed == ["the caller asked for a person"]


async def test_an_other_answer_is_answered_then_the_yes_counts(open_call):
    kit = await open_call(FoodPack())
    ai = ScriptedAI(check_number=[ok("4521")], work_left=[{"exit": "done"}],
                    answer_other=[{"exit": "answer", "text": "Poore paise card par wapas aayenge."}],
                    decide=[{"exit": "prepare", "tool": "prepare_cancel_order", "inputs": {}, "plan": "Cancel kar raha hoon."}])
    notebook, tools, _ = await run(kit, "Cancel karna hai", ai, [(None, ORDER_NUMBER, "4521"), (RESULT, CALLER, "bas")],
                                   [("other", "kitna milega?"), "yes"])
    assert kinds(notebook).index(ANSWER) < kinds(notebook).index(RESULT)
    assert tools.calls == ["find_order", "prepare_cancel_order", "confirm"]


async def test_a_no_presses_no_confirm(open_call):
    kit = await open_call(FoodPack())
    ai = ScriptedAI(check_number=[ok("4521")], work_left=[{"exit": "done"}],
                    decide=[{"exit": "prepare", "tool": "prepare_cancel_order", "inputs": {}}])
    _, tools, _ = await run(kit, "Cancel karna hai", ai, [(None, ORDER_NUMBER, "4521"), (RESULT, CALLER, "bas")], ["no"])
    assert "confirm" not in tools.calls
    assert (await kit.one("SELECT status FROM orders WHERE order_no = 4521"))["status"] != "cancelled"


async def test_a_new_order_number_mid_job_starts_from_the_check(open_call):
    kit = await open_call(FoodPack())
    ai = ScriptedAI(check_number=[ok("4561"), ok("4530")], work_left=[{"exit": "done"}], decide=[
        {"exit": "missing", "need": "new_address", "ask": "Naya address?"}, {"exit": "lookup"}])
    _, tools, _ = await run(kit, "Address badalna hai", ai, [
        (None, ORDER_NUMBER, "4561"), (NEED_INFO, ORDER_NUMBER, "4530"), (FACT, CALLER, "bas")])
    assert tools.calls == ["find_order", "find_order"]


async def test_more_work_goes_back_to_decide_in_the_same_job(open_call):
    kit = await open_call(FoodPack())
    ai = ScriptedAI(check_number=[ok("4512")], work_left=[{"exit": "more"}, {"exit": "done"}], decide=[
        {"exit": "lookup"},
        {"exit": "prepare", "tool": "prepare_refund_order", "inputs": {"item": "Raita", "reason": "missing_item"}}])
    _, tools, _ = await run(kit, "Order kab aaya?", ai, [
        (None, ORDER_NUMBER, "4512"), (FACT, CALLER, "Aur raita missing tha"), (RESULT, CALLER, "bas")], ["yes"])
    assert tools.calls == ["find_order", "prepare_refund_order", "confirm"]


async def test_the_same_refused_try_three_times_hands_over(open_call):
    kit = await open_call(FoodPack())
    pizza = {"exit": "prepare", "tool": "prepare_refund_order", "inputs": {"item": "Pizza", "reason": "missing_item"}}
    ai = ScriptedAI(check_number=[ok("4512")], decide=[pizza, pizza, pizza])
    _, tools, handed = await run(kit, "Pizza missing", ai, [(None, ORDER_NUMBER, "4512")])
    assert tools.calls == ["find_order"] + ["prepare_refund_order"] * 3 and handed == ["the same step kept failing"]


async def test_a_tool_not_on_the_menu_is_never_pressed(open_call):
    kit = await open_call(FoodPack())
    ai = ScriptedAI(check_number=[ok("4521")], after_explain=[{"exit": "accepts"}],
                    decide=[{"exit": "prepare", "tool": "confirm", "inputs": {"token": "x"}}],
                    explain=[{"exit": "note", "text": "Yeh nahi ho sakta."}])
    _, tools, _ = await run(kit, "Kar do", ai, [(None, ORDER_NUMBER, "4521"), (ANSWER, CALLER, "ok")])
    assert tools.calls == ["find_order"]


async def test_an_answer_with_no_allowed_exit_twice_is_stuck(open_call):
    kit = await open_call(FoodPack())
    ai = ScriptedAI(check_number=[{"exit": "maybe"}, {"nothing": True}])
    with pytest.raises(ThinkerStuck):
        await run(kit, "Cancel", ai, [(None, ORDER_NUMBER, "4521")])
