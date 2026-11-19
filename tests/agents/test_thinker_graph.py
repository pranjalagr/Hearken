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
    ANSWERED,
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
    YES_DROPPED,
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
        self.handed_orders: list[int | None] = []

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

    async def hand_over(reason: str, order_no: int | None = None) -> None:
        handed.append(reason)
        tools.handed_orders.append(order_no)  # which order the person takes over

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


async def test_an_answer_with_no_allowed_exit_three_times_is_stuck(open_call):
    # "decide" is the one big step that may still get stuck (then the box hands over).
    kit = await open_call(FoodPack())
    ai = ScriptedAI(check_number=[ok("4521")], decide=[{"exit": "maybe"}, {"nothing": True}, {"exit": "?"}])
    with pytest.raises(ThinkerStuck):
        await run(kit, "Cancel", ai, [(None, ORDER_NUMBER, "4521")])


async def test_a_refused_change_is_never_pressed_again_with_the_same_inputs(open_call):
    # The owner's rule, 17 Nov 2026: whatever the AI decides, code never presses a refused
    # change again; it explains, and if the caller still insists, hands over.
    kit = await open_call(FoodPack())
    cancel = {"exit": "prepare", "tool": "prepare_cancel_order", "inputs": {}}
    ai = ScriptedAI(check_number=[ok("4512")], decide=[cancel, cancel], after_explain=[{"exit": "changed"}],
                    explain=[{"exit": "note", "text": "Deliver ho chuka hai, cancel nahi ho sakta."}])
    _, tools, handed = await run(kit, "Cancel karna hai", ai, [(None, ORDER_NUMBER, "4512"),
                                                             (ANSWER, CALLER, "Phir bhi cancel karo")])
    assert tools.calls == ["find_order", "prepare_cancel_order"]  # pressed once, never again
    assert handed == ["the caller still insists on something this line can't do"]


async def test_a_hand_over_carries_the_checked_order_number(open_call):
    # 17 Nov 2026, run 2: hand-overs were saved with no order, and the lab scored the right
    # ones in food_05 and food_06 as wrong changes.
    kit = await open_call(FoodPack())
    ai = ScriptedAI(check_number=[ok("4568")], after_explain=[{"exit": "person"}],
                    decide=[{"exit": "cannot", "reason": "no rider numbers on this line"}],
                    explain=[{"exit": "note", "text": "Rider ka number nahi de sakte. Person se connect karun?"}])
    _, tools, handed = await run(kit, "Rider ka number do", ai, [(None, ORDER_NUMBER, "4568"), (ANSWER, CALLER, "Haan")])
    assert handed == ["the caller asked for a person"] and tools.handed_orders == [4568]


async def test_a_new_request_during_the_yes_question_is_taken_up_after_it(open_call):
    # The user's choice, 17 Nov 2026 (food_10): "haan, aur rider ka number bhi chahiye" during
    # the yes-question becomes a request of its own once the yes moment ends.
    kit = await open_call(FoodPack())
    ai = ScriptedAI(check_number=[ok("4568")], after_explain=[{"exit": "person"}],
                    answer_other=[{"exit": "new_request", "text": "Ji, rider ka number bhi dekhta hoon, pehle refund.",
                                   "request": "the rider's phone number"}],
                    decide=[{"exit": "prepare", "tool": "prepare_refund_order",
                             "inputs": {"item": "Veg Momos", "reason": "missing_item"}},
                            {"exit": "cannot", "reason": "this line can't give the rider's number"}],
                    explain=[{"exit": "note", "text": "Rider ka number nahi de sakte. Person se connect karun?"}])
    # The caller's "connect me" comes after the refund's result (the answer during the question
    # is an ANSWER note too, so waiting for one would reply too early).
    notebook, tools, handed = await run(kit, "Momos nahi aaye", ai, [(None, ORDER_NUMBER, "4568"),
                                                                    (RESULT, CALLER, "Haan, connect karo")],
                                        [("other", "Haan, aur rider ka number bhi chahiye"), "yes"])
    assert tools.calls == ["find_order", "prepare_refund_order", "confirm"]
    assert (await kit.one("SELECT amount_paise FROM refunds WHERE order_no = 4568 AND at >= 0"))["amount_paise"] == 18000
    second_decide = [facts for step, facts in ai.asked if step == "decide"][1]
    assert second_decide["handle_now"] == "the rider's phone number"
    assert handed == ["the caller asked for a person"] and tools.handed_orders == [4568]


async def test_a_failed_person_check_never_crashes_the_job(open_call):
    # 17 Nov 2026, run 3, food_10: the AI gave no usable answer twice at the small "a person?"
    # check, and the whole job crashed. Now the caller just goes on giving the number.
    kit = await open_call(FoodPack())
    ai = ScriptedAI(person=[{"exit": "maybe"}, {"nothing": True}, {"exit": "?"}], check_number=[ok("4521")],
                    decide=[{"exit": "lookup"}], work_left=[{"exit": "done"}])
    _, tools, handed = await run(kit, "Status batao", ai, [(None, CALLER, "ये"), (None, ORDER_NUMBER, "4521"),
                                                          (FACT, CALLER, "bas")])
    assert tools.calls == ["find_order"] and handed == []


async def test_a_failed_number_check_asks_for_the_number_again(open_call):
    # 17 Nov 2026, run 3, food_02: a Gemini slow spell at the number check crashed the job.
    # Now the caller is asked for the number again, and the next try goes ahead.
    kit = await open_call(FoodPack())
    bad = {"exit": "maybe"}
    ai = ScriptedAI(check_number=[bad, bad, bad, ok("4521")], decide=[{"exit": "lookup"}], work_left=[{"exit": "done"}])
    notebook, tools, handed = await run(kit, "Status batao", ai, [
        (None, ORDER_NUMBER, "4521"), (NEED_INFO, ORDER_NUMBER, "4521"), (FACT, CALLER, "bas")])
    assert NEED_INFO in kinds(notebook) and tools.calls == ["find_order"] and handed == []


async def test_a_failed_anything_else_check_asks_again(open_call):
    # 17 Nov 2026, run 3, food_03: the job crashed reading the reply to "anything else?"
    # after the work was done. Now "anything else?" is asked again.
    kit = await open_call(FoodPack())
    bad = {"exit": "maybe"}
    ai = ScriptedAI(check_number=[ok("4521")], decide=[{"exit": "lookup"}],
                    work_left=[bad, bad, bad, {"exit": "done"}])
    notebook, tools, handed = await run(kit, "Status batao", ai, [
        (None, ORDER_NUMBER, "4521"), (FACT, CALLER, "hmm"), (ANSWERED, CALLER, "bas")])
    assert kinds(notebook).count(ANSWERED) == 2 and handed == []  # the lookup's, then the ask-again


async def test_each_thrown_away_try_is_logged_with_its_reason(open_call):
    # Run 3 only said "no allowed answer": a slow answer and a wrong one looked the same.
    kit = await open_call(FoodPack())

    class SlowOnce(ScriptedAI):
        async def ask(self, step, facts):
            if step == "check_number" and not self.asked:
                self.asked.append((step, facts))
                await asyncio.sleep(1.0)  # longer than the step's limit
            return await super().ask(step, facts)

    ai = SlowOnce(check_number=[{"exit": "maybe"}, ok("4521")], decide=[{"exit": "lookup"}],
                  work_left=[{"exit": "done"}])
    notebook = Notebook(max_notes=200)
    notebook.add(REQUEST, "Status batao", 1)
    failed: list[tuple[str, str]] = []

    async def hand_over(reason, order_no=None):
        pass

    parts = ThinkerParts(notebook, BackendTools(kit.backend), ai, hand_over,
                         build_catalog(FoodPack().actions(), kit.settings), step_s=0.3,
                         on_try_failed=lambda step, reason: failed.append((step, reason)))
    caller = asyncio.create_task(_caller(notebook, [(None, ORDER_NUMBER, "4521"), (FACT, CALLER, "bas")]))
    try:
        await asyncio.wait_for(run_job(THINKER_GRAPH, parts, f"job-{id(notebook)}"), 10)
    finally:
        caller.cancel()
    assert [step for step, _ in failed] == ["check_number", "check_number"]
    assert "no answer within" in failed[0][1] and "exit not allowed" in failed[1][1]


async def test_decide_goes_to_its_own_ai_and_every_other_step_to_the_default():
    from agents.agent_c.thinker.step_brain import StepRouter
    default, decide = ScriptedAI(person=[{"exit": "no"}]), ScriptedAI(decide=[{"exit": "lookup"}])
    router = StepRouter(default, {"decide": decide})
    assert await router.ask("person", {}) == {"exit": "no"}
    assert await router.ask("decide", {}) == {"exit": "lookup"}
    assert [s for s, _ in default.asked] == ["person"] and [s for s, _ in decide.asked] == ["decide"]


async def test_the_groq_brain_moves_to_the_next_key_when_one_is_refused():
    # The owner's rule (16 Nov 2026): the first working key, the next on failure.
    import httpx

    from agents.agent_c.thinker.step_brain import GroqStepBrain
    seen: list[str] = []

    def network(request: httpx.Request) -> httpx.Response:
        key = request.headers["Authorization"].removeprefix("Bearer ")
        seen.append(key)
        if key == "used-up":
            return httpx.Response(429, json={"error": {"message": "Rate limit reached ... on tokens per day"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"exit": "lookup"}'}}]})

    clients = [httpx.AsyncClient(base_url="https://groq.test", headers={"Authorization": f"Bearer {key}"},
                                 transport=httpx.MockTransport(network)) for key in ("used-up", "fresh")]
    switches: list[str] = []
    brain = GroqStepBrain(clients, "openai/gpt-oss-120b", on_switch=switches.append)
    assert await brain.ask("decide", {"caller_words": ["status batao"]}) == {"exit": "lookup"}
    assert await brain.ask("decide", {"caller_words": ["status batao"]}) == {"exit": "lookup"}
    assert seen == ["used-up", "fresh", "fresh"] and len(switches) == 1  # moved once, for good
    assert "used-up" not in switches[0] and "fresh" not in switches[0]  # never a key in the diary


# ---- every caller word reaches the Thinker, whatever its step (the owner's rule, 18 Nov 2026) ----

class TalkWhileDeciding(ScriptedAI):
    """A ScriptedAI where the caller speaks while some "decide" steps think: the Talker's
    code writes the turn and raises the job's version, as job_turn does."""

    def __init__(self, notebook: Notebook, version: dict[str, int], during_decide: list[str | None], **answers):
        super().__init__(**answers)
        self.notebook, self.version, self.during = notebook, version, list(during_decide)

    async def ask(self, step: str, facts: dict[str, Any]) -> dict[str, Any]:
        answer = await super().ask(step, facts)
        words = self.during.pop(0) if step == "decide" and self.during else None
        if words:
            self.version["now"] += 1
            self.notebook.add(CALLER, words, self.version["now"])
        return answer


async def _talker_and_guard(kit, tools: BackendTools, notebook: Notebook, version: dict[str, int]) -> None:
    """Stands in for the Talker and the Guard: a yes-question older than the newest turn is
    dropped and the Thinker told; a current one gets the caller's yes."""
    seen = 0
    while True:
        seen = await _after(notebook, NEED_YES, seen)
        note = notebook.notes[seen - 1]
        if note.version < version["now"]:
            notebook.add(YES_DROPPED, note.text, version["now"], {"token": note.data["data"]["token"]})
            continue
        token = note.data["data"]["token"]
        notebook.add(YES_ANSWER, "yes: haan", version["now"])
        reply = await tools.call("confirm", {"token": token, "ticket": await kit.guard.issue_ticket(token)})
        notebook.add(RESULT, f"done: {reply.status}", version["now"], {"outcome": "done"})
        return


async def run_talking(kit, ai_answers: dict[str, list], during_decide: list[str | None], redecide_limit: int = 2):
    notebook = Notebook(max_notes=200)
    notebook.add(REQUEST, "Address change karna hai", 1)
    version = {"now": 1}
    ai = TalkWhileDeciding(notebook, version, during_decide, **ai_answers)
    tools = BackendTools(kit.backend)
    handed: list[str] = []

    async def hand_over(reason: str, order_no: int | None = None) -> None:
        handed.append(reason)

    parts = ThinkerParts(notebook, tools, ai, hand_over, build_catalog(FoodPack().actions(), kit.settings),
                         version=lambda: version["now"], step_s=2.0, redecide_limit=redecide_limit)
    helpers = [asyncio.create_task(_caller(notebook, [(None, ORDER_NUMBER, "4561"), (RESULT, CALLER, "Bas, thanks")])),
               asyncio.create_task(_talker_and_guard(kit, tools, notebook, version))]
    try:
        await asyncio.wait_for(run_job(THINKER_GRAPH, parts, f"job-{id(notebook)}"), 20)
    finally:
        for task in helpers:
            task.cancel()
    return notebook, tools, handed, ai


def _change(address: str) -> dict[str, Any]:
    return {"exit": "prepare", "tool": "prepare_change_address", "inputs": {"new_address": address},
            "plan": "Address badal raha hoon."}


async def test_a_turn_during_decide_drops_the_question_and_the_thinker_decides_again(open_call):
    # 18 Nov 2026, food_14: "Gurgaon" came 1 s into "decide", and the address was asked and saved
    # as "House 74, Sector 15." The question worked out before it is dropped; "decide" runs
    # again, sees "Gurgaon", and the full address is asked and saved.
    kit = await open_call(FoodPack())
    full = "House 74, Sector 15, Gurgaon"
    notebook, tools, handed, ai = await run_talking(
        kit, {"check_number": [ok("4561")], "work_left": [{"exit": "done"}],
              "decide": [_change("House 74, Sector 15."), _change(full)]}, ["Gurgaon", None])
    decides = [facts for step, facts in ai.asked if step == "decide"]
    assert len(decides) == 2 and "Gurgaon" in decides[1]["caller_words"]
    assert tools.calls == ["find_order", "prepare_change_address", "prepare_change_address", "confirm"]
    assert [n.kind for n in notebook.notes].count(YES_DROPPED) == 1
    assert (await kit.one("SELECT address FROM orders WHERE order_no = 4561"))["address"] == full
    assert handed == []


async def test_a_caller_who_keeps_talking_gets_the_question_after_the_redecide_limit(open_call):
    # The limit: a caller who talks through every "decide" must not keep the robot deciding.
    # After 2 re-decides the question is asked anyway. The same change three times is not
    # "going round in circles" either (each dropped press is taken off that count).
    kit = await open_call(FoodPack())
    same = "Flat 2C, Rose Apartments, Model Town, Ludhiana"
    notebook, tools, handed, ai = await run_talking(
        kit, {"check_number": [ok("4561")], "work_left": [{"exit": "done"}],
              "decide": [_change(same), _change(same), _change(same)]}, ["aur", "haan", "suniye"])
    assert [step for step, _ in ai.asked].count("decide") == 3
    assert [n.kind for n in notebook.notes].count(YES_DROPPED) == 2
    assert tools.calls[-1] == "confirm" and handed == []
    assert (await kit.one("SELECT address FROM orders WHERE order_no = 4561"))["address"] == same


async def _guard_with_a_correction(kit, tools: BackendTools, notebook: Notebook, correction: str) -> None:
    """Stands in for the Guard: at the first yes-question the caller corrects it (the judge
    says "changed", the Guard closes it); the second gets a yes."""
    seen = await _after(notebook, NEED_YES, 0)
    notebook.add(CALLER, correction, 1)  # the caller's turn, which every job turn writes
    notebook.add(YES_ANSWER, f"changed: {correction}", 1)
    seen = await _after(notebook, NEED_YES, seen)
    token = notebook.notes[seen - 1].data["data"]["token"]
    notebook.add(YES_ANSWER, "yes: haan", 1)
    reply = await tools.call("confirm", {"token": token, "ticket": await kit.guard.issue_ticket(token)})
    notebook.add(RESULT, f"done: {reply.status}", 1, {"outcome": "done"})


async def test_a_correction_during_the_yes_question_goes_back_to_decide(open_call):
    # 18 Nov 2026, food_04: the question said "Flat do si" (the ears' mishearing of "Flat 2C").
    # A correction then makes the old question wrong; the Guard closes it ("changed"), and the
    # Thinker decides again with every caller word: the right address is asked and saved.
    kit = await open_call(FoodPack())
    wrong, right = "Flat do si, Rose Apartments, Model Town, Ludhiana", "Flat 2C, Rose Apartments, Model Town, Ludhiana"
    ai = ScriptedAI(check_number=[ok("4561")], work_left=[{"exit": "done"}], decide=[_change(wrong), _change(right)])
    notebook = Notebook(max_notes=200)
    notebook.add(REQUEST, "Address change karna hai", 1)
    tools = BackendTools(kit.backend)

    async def hand_over(reason: str, order_no: int | None = None) -> None:
        raise AssertionError(f"no hand-over expected: {reason}")

    parts = ThinkerParts(notebook, tools, ai, hand_over, build_catalog(FoodPack().actions(), kit.settings), step_s=2.0)
    helpers = [asyncio.create_task(_caller(notebook, [(None, ORDER_NUMBER, "4561"), (RESULT, CALLER, "Bas, thanks")])),
               asyncio.create_task(_guard_with_a_correction(kit, tools, notebook, "Flat 2C, Rose Apartments"))]
    try:
        await asyncio.wait_for(run_job(THINKER_GRAPH, parts, f"job-{id(notebook)}"), 20)
    finally:
        for task in helpers:
            task.cancel()
    decides = [facts for step, facts in ai.asked if step == "decide"]
    assert len(decides) == 2 and "Flat 2C, Rose Apartments" in decides[1]["caller_words"]
    assert "answer_other" not in [step for step, _ in ai.asked]  # never treated as a side question
    assert tools.calls == ["find_order", "prepare_change_address", "prepare_change_address", "confirm"]
    assert (await kit.one("SELECT address FROM orders WHERE order_no = 4561"))["address"] == right


# ---- the order number is never part of an address (19 Nov 2026) -----------------------------

def test_code_finds_the_order_number_inside_an_input():
    from agents.agent_c.thinker.graph import _holds_order_number

    assert _holds_order_number({"new_address": "4 5 6 1"}, "4561")  # food_04, 18 Nov 2026
    assert _holds_order_number({"new_address": "45 30 House 74, Sector 15, Gurgaon"}, "4530")  # food_14
    assert _holds_order_number({"new_address": "4561"}, "4561")
    assert not _holds_order_number({"new_address": "Flat 45, Sector 30, Indore"}, "4530")  # split by words
    assert not _holds_order_number({"new_address": "House 74, Sector 15, Gurgaon"}, "4530")
    assert not _holds_order_number({}, "4530") and not _holds_order_number({"new_address": "x 4561"}, "")


async def _number_then(kit, decide_answers, caller_after_ask=None):
    """A job where the caller gives the number as "4 5 6 1" (its turn written as every job
    turn is), then answers decide's question if it asks one."""
    notebook = Notebook(max_notes=200)
    notebook.add(REQUEST, "Hi, I need to change the delivery address of my order.", 1)
    ai = ScriptedAI(check_number=[ok("4561")], work_left=[{"exit": "done"}], decide=decide_answers)
    tools = BackendTools(kit.backend)

    async def hand_over(reason, order_no=None):
        raise AssertionError(f"no hand-over expected: {reason}")

    async def caller():
        notebook.add(CALLER, "4 5 6 1", 1)
        notebook.add(ORDER_NUMBER, "4561", 1, {"digits": "4561", "words": "4 5 6 1"})
        if caller_after_ask:
            await _after(notebook, NEED_INFO, 0)
            notebook.add(CALLER, caller_after_ask, 1)
        await _after(notebook, RESULT, 0)
        notebook.add(CALLER, "Bas, thanks", 1)

    parts = ThinkerParts(notebook, tools, ai, hand_over, build_catalog(FoodPack().actions(), kit.settings), step_s=2.0)
    helpers = [asyncio.create_task(caller()), asyncio.create_task(_guard(kit, tools, notebook, ["yes"]))]
    try:
        await asyncio.wait_for(run_job(THINKER_GRAPH, parts, f"job-{id(notebook)}"), 20)
    finally:
        for task in helpers:
            task.cancel()
    return notebook, tools, ai


async def test_the_order_number_as_the_address_is_sent_back_to_decide_with_the_reason(open_call):
    # 18 Nov 2026, food_04: "decide" prepared new_address "4 5 6 1" and the address became "4 5 6 1".
    kit = await open_call(FoodPack())
    right = "Flat 2C, Rose Apartments, Model Town, Ludhiana"
    notebook, tools, ai = await _number_then(kit, [
        _change("4 5 6 1"),
        {"exit": "missing", "need": "new_address", "ask": "Naya address kya hai?"},
        _change(right)], caller_after_ask=right)
    decides = [facts for step, facts in ai.asked if step == "decide"]
    assert "4 5 6 1 (this turn gave the order number)" in decides[0]["caller_words"]  # marked for the AI
    assert "order number 4561" in decides[1]["last_refusal"]  # sent back with the reason
    assert tools.calls == ["find_order", "prepare_change_address", "confirm"]  # "4 5 6 1" never prepared
    assert (await kit.one("SELECT address FROM orders WHERE order_no = 4561"))["address"] == right


async def test_the_order_number_in_an_input_twice_asks_the_caller_again(open_call):
    kit = await open_call(FoodPack())
    right = "Flat 2C, Rose Apartments, Model Town, Ludhiana"
    notebook, tools, ai = await _number_then(kit, [_change("4 5 6 1"), _change("45 61 Model Town"), _change(right)],
                                             caller_after_ask=right)
    asked = [note for note in notebook.notes if note.kind == NEED_INFO]
    assert [note.text for note in asked] == ["Kripya naya address ek baar phir bataiye."]
    assert tools.calls == ["find_order", "prepare_change_address", "confirm"]
    assert (await kit.one("SELECT address FROM orders WHERE order_no = 4561"))["address"] == right


# ---- the address check before pressing (19 Nov 2026) ----------------------------------------

class FakeCheck:
    """An address check that answers from a list, and keeps what it was asked."""

    def __init__(self, *answers):
        self.answers, self.asked = list(answers), []

    async def same_address(self, caller_words, address, order_number):
        self.asked.append((list(caller_words), address, order_number))
        return self.answers.pop(0)


async def _address_job(kit, decide_answers, check, caller_after_ask=None):
    """An address change where the caller gives 4561 and, if asked again, the address."""
    notebook = Notebook(max_notes=200)
    notebook.add(REQUEST, "Address change karna hai", 1)
    ai = ScriptedAI(check_number=[ok("4561")], work_left=[{"exit": "done"}], decide=decide_answers)
    tools = BackendTools(kit.backend)

    async def hand_over(reason, order_no=None):
        raise AssertionError(f"no hand-over expected: {reason}")

    async def caller():
        notebook.add(ORDER_NUMBER, "4561", 1, {"digits": "4561"})
        if caller_after_ask:
            await _after(notebook, NEED_INFO, 0)
            notebook.add(CALLER, caller_after_ask, 1)
        await _after(notebook, RESULT, 0)
        notebook.add(CALLER, "Bas, thanks", 1)

    parts = ThinkerParts(notebook, tools, ai, hand_over, build_catalog(FoodPack().actions(), kit.settings), step_s=2.0,
                         check_address=check, checked_inputs=("new_address",), address_line=0.5)
    helpers = [asyncio.create_task(caller()), asyncio.create_task(_guard(kit, tools, notebook, ["yes"]))]
    try:
        await asyncio.wait_for(run_job(THINKER_GRAPH, parts, f"job-{id(notebook)}"), 20)
    finally:
        for task in helpers:
            task.cancel()
    return notebook, tools, ai


async def test_an_address_the_check_doubts_is_sent_back_to_decide_and_never_pressed(open_call):
    # 18 Nov 2026, food_14: "House 74, Sector 15." was saved without the "Gurgaon" the caller said.
    kit = await open_call(FoodPack())
    full = "House 74, Sector 15, Gurgaon"
    check = FakeCheck(0.10, 0.95)
    notebook, tools, ai = await _address_job(kit, [_change("House 74, Sector 15"), _change(full)], check)
    decides = [facts for step, facts in ai.asked if step == "decide"]
    assert "not what the caller said" in decides[1]["last_refusal"] and "0.10" in decides[1]["last_refusal"]
    assert [address for _, address, _ in check.asked] == ["House 74, Sector 15", full]
    assert tools.calls == ["find_order", "prepare_change_address", "confirm"]  # the doubted one never pressed
    assert (await kit.one("SELECT address FROM orders WHERE order_no = 4561"))["address"] == full


async def test_an_address_doubted_twice_asks_the_caller_again(open_call):
    kit = await open_call(FoodPack())
    right = "Flat 2C, Rose Apartments, Model Town, Ludhiana"
    check = FakeCheck(0.10, 0.20, 0.90)
    notebook, tools, _ = await _address_job(kit, [_change("Flat 2C"), _change("Flat 2C, Rose"), _change(right)], check,
                                            caller_after_ask=right)
    assert [n.text for n in notebook.notes if n.kind == NEED_INFO] == ["Kripya naya address ek baar phir bataiye."]
    assert tools.calls == ["find_order", "prepare_change_address", "confirm"]
    assert (await kit.one("SELECT address FROM orders WHERE order_no = 4561"))["address"] == right


async def test_no_answer_from_the_check_lets_the_change_go_on(open_call):
    kit = await open_call(FoodPack())
    right = "Flat 2C, Rose Apartments, Model Town, Ludhiana"
    notebook, tools, _ = await _address_job(kit, [_change(right)], FakeCheck(None))
    assert tools.calls == ["find_order", "prepare_change_address", "confirm"]
    assert (await kit.one("SELECT address FROM orders WHERE order_no = 4561"))["address"] == right
