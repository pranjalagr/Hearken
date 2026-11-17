"""The Talker graph (17 Nov 2026): first alone, with a fake Thinker writing scripted notes;
then end to end, with the real Thinker graph and the real food backend.

The AIs are scripts (no key, no quota), the voice is a list of what was said, and the
finder is a script of tagged number words. The end-to-end tests use a stand-in Guard that
gets a real yes-ticket and presses the real confirm.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

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
    Notebook,
)
from agents.agent_c.talker.graph import TALKER_GRAPH, TalkerParts, TalkerRunner
from agents.agent_c.talker.talker_ai import REPLY
from agents.agent_c.thinker.graph import THINKER_GRAPH, ThinkerParts, run_job
from engine.contract.spoken import SpokenWords
from engine.service.tool_catalog import build_catalog
from packs.food.pack import FoodPack
from test_thinker_graph import BackendTools, ScriptedAI  # pytest puts this folder on the path

WORDS = SpokenWords({**FoodPack().spoken_words().groups,
                     **SpokenWords.from_yaml(Path("agents/agent_c/talker/words.yaml")).groups})
GREETING = "Namaste, Hearken Food se bol raha hoon. Bataiye?"


def line(group: str, key: str, **values: str) -> str:
    return WORDS.phrase(group, key, "hinglish").format(**values)


class Voice:
    """Keeps every line said, in order."""

    def __init__(self) -> None:
        self.said: list[str] = []

    async def say(self, text: str) -> None:
        self.said.append(text)


class ScriptedTalkerAI:
    """Answers each step from a list: (exit, sentences). The sentences go to the voice."""

    def __init__(self, **answers: list[tuple[str, list[str]]]) -> None:
        self.answers = {step: list(given) for step, given in answers.items()}
        self.asked: list[tuple[str, dict[str, Any]]] = []

    async def reply(self, step, facts, exits, speak):
        self.asked.append((step, facts))
        exit_, sentences = self.answers[step].pop(0)
        for sentence in sentences:
            await speak(sentence)
        return exit_


def finder(**by_words: list[dict[str, str]]):
    """A scripted number finder: the tagged number words for each turn."""
    async def find(words: str) -> list[dict[str, str]]:
        return by_words.get(words, [])
    return find


def order(words: str) -> list[dict[str, str]]:
    return [{"words": words, "tag": "order"}]


class FakeGuard:
    """Records what reaches the Guard."""

    def __init__(self) -> None:
        self.prepared_notes, self.answers, self.keys = [], [], []

    async def prepared(self, note) -> None:
        self.prepared_notes.append(note)

    async def answer(self, words, start_ms) -> None:
        self.answers.append(words)

    def key(self, digit, caller_ms) -> None:
        self.keys.append(digit)


class Call:
    """One call's Talker, with everything it touched kept for the test to check."""

    def __init__(self, ai, find=finder(), guard=None, keypad_s: float = 15.0) -> None:
        self.notebook = Notebook(max_notes=200)
        self.voice = Voice()
        self.guard = guard or FakeGuard()
        self.thinker_started = 0
        self.ended: list[str] = []

        def start_thinker() -> None:
            self.thinker_started += 1

        async def end_call(reason: str) -> None:
            self.ended.append(reason)

        self.parts = TalkerParts(self.notebook, self.voice, WORDS, ai, find, self.guard, start_thinker, end_call,
                                 GREETING, keypad_s=keypad_s)
        self.runner = TalkerRunner(TALKER_GRAPH, self.parts, f"call-{id(self)}")
        self.task: asyncio.Task | None = None

    async def start(self) -> None:
        self.task = asyncio.create_task(self.runner.run())
        await self.settle()

    async def settle(self) -> None:
        """Let the graph handle everything queued so far."""
        for _ in range(30):
            await asyncio.sleep(0.005)

    async def turn(self, words: str) -> None:
        self.runner.turn(words)
        await self.settle()

    async def note(self, kind: str, text: str, data: dict[str, Any] | None = None) -> None:
        self.notebook.add(kind, text, 1, data or {})
        await self.settle()

    async def finish(self) -> None:
        await asyncio.wait_for(self.task, 2)

    def kinds(self) -> list[str]:
        return [note.kind for note in self.notebook.notes]


def found(number: int, status: str = "preparing") -> dict[str, Any]:
    return {"tool": "find_order", "status": "ok", "data": {"order_no": number, "status": status}}


NOT_FOUND = {"tool": "find_order", "status": "refused", "data": {"wrong_input": "order_no"}}


# ---- the Talker alone -------------------------------------------------------------------

async def test_greets_replies_then_says_goodbye():
    call = Call(ScriptedTalkerAI(no_job=[(REPLY, ["Namaste ji, bataiye kya madad karun?"]), ("end_call", [])]))
    await call.start()
    await call.turn("Hello")
    await call.turn("Kuch nahi, thank you")
    await call.finish()
    assert call.voice.said == [GREETING, "Namaste ji, bataiye kya madad karun?", line("talk", "goodbye")]
    assert call.ended == ["the caller said goodbye"] and call.thinker_started == 0


async def test_a_job_opens_and_a_number_in_the_same_turn_is_heard():
    words = "Order 4521 cancel karna hai"
    call = Call(ScriptedTalkerAI(no_job=[("open_job", [])], collect=[("heard_number", [])]),
                finder(**{words: order("4521")}))
    await call.start()
    await call.turn(words)
    assert call.voice.said[1:] == [line("talk", "quick"), line("talk", "number_heard")]  # never the number
    assert call.kinds() == [REQUEST, ORDER_NUMBER] and call.notebook.notes[1].data["digits"] == "4521"
    assert call.thinker_started == 1


async def test_no_number_asks_for_it_then_hears_it_later():
    call = Call(ScriptedTalkerAI(no_job=[("open_job", [])], collect=[("heard_number", []), ("heard_number", [])]),
                finder(**{"chaar paanch do ek": order("chaar paanch do ek")}))
    await call.start()
    await call.turn("Mera order cancel karna hai")
    await call.turn("chaar paanch do ek")
    assert call.voice.said[2:] == [line("talk", "ask_number"), line("talk", "number_heard")]
    # The second turn reaches the Thinker too (decision 7), then the number.
    assert call.kinds() == [REQUEST, CALLER, ORDER_NUMBER] and call.notebook.notes[2].data["digits"] == "4521"


async def test_a_short_number_asks_for_the_full_one():
    call = Call(ScriptedTalkerAI(no_job=[("open_job", [])], collect=[("heard_number", [])]),
                finder(**{"chaar paanch": order("chaar paanch")}))
    await call.start()
    await call.turn("chaar paanch")
    assert call.voice.said[-1] == line("talk", "number_again_full") and ORDER_NUMBER not in call.kinds()


async def test_number_words_the_reader_cant_read_go_to_the_thinker():
    words = "forty five hundred twenty one"
    call = Call(ScriptedTalkerAI(no_job=[("open_job", [])], collect=[("heard_number", [])]),
                finder(**{words: [{"words": words, "tag": "unsure"}]}))
    await call.start()
    await call.turn(words)
    note = call.notebook.notes[-1]
    assert note.kind == ORDER_NUMBER and note.data == {"digits": "", "words": words}
    assert call.voice.said[-1] == line("talk", "number_heard")


async def test_the_thinkers_notes_are_said_the_agreed_way():
    call = Call(ScriptedTalkerAI(no_job=[("open_job", [])], collect=[("heard_number", [])]),
                finder(**{"4521": order("4521")}))
    await call.start()
    await call.turn("4521")
    await call.note(CHECKED_NUMBER, "4521", {"digits": "4521"})
    await call.note(FACT, "find_order: ok", found(4521))
    await call.note(PLAN, "Order cancel kar raha hoon.")
    await call.note(NEED_YES, "cancel order 4521", {"tool": "prepare_cancel_order", "data": {"token": "t"}})
    await call.note(RESULT, "cancel_order: done", {"action": "cancel_order", "outcome": "done"})
    assert call.voice.said[-5:] == [
        line("talk", "number_back", number="4521"),
        line("talk", "order_found", number="4521", status=WORDS.phrase("status", "preparing", "hinglish")),
        "Order cancel kar raha hoon.",
        line("result", "cancel_order_done"),
        line("talk", "anything_else")]
    assert len(call.guard.prepared_notes) == 1  # the yes moment went to the Guard


async def test_not_found_asks_again_then_digit_by_digit_then_the_keypad():
    call = Call(ScriptedTalkerAI(no_job=[("open_job", [])], collect=[("heard_number", [])] * 3),
                finder(**{"4599": order("4599"), "4598": order("4598")}))
    await call.start()
    await call.turn("4599")
    await call.note(FACT, "not found", NOT_FOUND)
    await call.turn("4598")
    await call.note(FACT, "not found", NOT_FOUND)
    await call.turn("4598")
    await call.note(FACT, "not found", NOT_FOUND)
    ladder = [said for said in call.voice.said if said in {line("ladder", k) for k in
                                                           ("number_again", "number_digits", "number_keypad")}]
    assert ladder == [line("ladder", "number_again"), line("ladder", "number_digits"), line("ladder", "number_keypad")]
    for digit in "4521":  # typed on the keypad
        call.runner.key(digit)
    await call.settle()
    typed = call.notebook.notes[-1]
    assert typed.kind == ORDER_NUMBER and typed.data == {"digits": "4521", "typed": True}


async def test_still_not_found_after_the_keypad_writes_asks_used_up():
    call = Call(ScriptedTalkerAI(no_job=[("open_job", [])], collect=[("heard_number", [])] * 3),
                finder(**{"4599": order("4599")}))
    await call.start()
    for _ in range(3):
        await call.turn("4599")
        await call.note(FACT, "not found", NOT_FOUND)
    for digit in "4599#":
        call.runner.key(digit)
    await call.settle()
    await call.note(FACT, "not found", NOT_FOUND)
    assert call.kinds()[-1] == ASKS_USED_UP  # the Thinker hands over on this note


async def test_nothing_typed_in_time_counts_as_a_miss():
    call = Call(ScriptedTalkerAI(no_job=[("open_job", [])], collect=[("heard_number", [])] * 3),
                finder(**{"4599": order("4599")}), keypad_s=0.05)
    await call.start()
    for _ in range(3):
        await call.turn("4599")
        await call.note(FACT, "not found", NOT_FOUND)
    await asyncio.sleep(0.1)
    await call.settle()
    assert call.kinds()[-1] == ASKS_USED_UP


async def test_a_made_up_number_is_never_said():
    call = Call(ScriptedTalkerAI(no_job=[("open_job", [])], collect=[(REPLY, ["Order number bataiye?"])]))
    await call.start()
    await call.turn("Cancel karna hai")
    await call.note(ANSWER, "Aapko 999 rupaye wapas milenge.")  # 999 is in no note and nothing the caller said
    assert call.voice.said[-1] == line("line", "safe")


async def test_an_ai_reply_is_cut_to_two_sentences():
    call = Call(ScriptedTalkerAI(no_job=[(REPLY, ["Ek.", "Do.", "Teen."])]))
    await call.start()
    await call.turn("Hello")
    assert call.voice.said[1:] == ["Ek.", "Do."]


async def test_an_answer_during_the_yes_moment_keeps_the_yes_moment():
    call = Call(ScriptedTalkerAI(no_job=[("open_job", [])], collect=[("heard_number", [])]),
                finder(**{"4521": order("4521")}))
    await call.start()
    await call.turn("4521")
    await call.note(NEED_YES, "cancel order 4521", {"tool": "prepare_cancel_order", "data": {"token": "t"}})
    await call.turn("Kitna milega?")
    await call.note(ANSWER, "Poore paise wapas aayenge.")
    await call.turn("Haan kar do")
    assert call.guard.answers == ["Kitna milega?", "Haan kar do"]  # both went to the Guard
    assert call.voice.said[-1] == "Poore paise wapas aayenge."


async def test_a_need_info_question_is_asked():
    call = Call(ScriptedTalkerAI(no_job=[("open_job", [])], collect=[("heard_number", [])]),
                finder(**{"4561": order("4561")}))
    await call.start()
    await call.turn("4561")
    await call.note(NEED_INFO, "Naya address kya hai?", {"need": "new_address"})
    assert call.voice.said[-1] == "Naya address kya hai?"


async def test_a_hand_over_says_its_line_and_ends_the_call():
    call = Call(ScriptedTalkerAI(no_job=[("open_job", [])], collect=[(REPLY, ["Order number?"])]))
    await call.start()
    await call.turn("Mujhe kisi insaan se baat karni hai")
    call.runner.hand_over("the caller asked for a person")
    await call.finish()
    assert call.voice.said[-1] == line("line", "hand_over") and call.ended == ["the caller asked for a person"]


async def test_after_a_job_closes_the_next_turn_is_read_with_no_job_open():
    call = Call(ScriptedTalkerAI(no_job=[("open_job", []), (REPLY, ["Ji bataiye."])],
                                 collect=[("heard_number", [])]), finder(**{"4521": order("4521")}))
    await call.start()
    await call.turn("4521")
    call.runner.job_closed(caller_done=False)
    await call.turn("Ek aur baat")
    assert call.voice.said[-1] == "Ji bataiye."


# ---- Talker and Thinker together, against the real backend ---------------------------------

class RealGuard:
    """Stands in for the Guard: a "haan" gets a real yes-ticket and the real confirm; a
    "nahi" stops it. Either way it writes the result, as the real Guard does."""

    def __init__(self, kit, tools, notebook) -> None:
        self.kit, self.tools, self.notebook, self.note = kit, tools, notebook, None

    async def prepared(self, note) -> None:
        self.note = note

    async def answer(self, words, start_ms) -> None:
        action = self.note.data["action"]
        if words.lower().startswith("haan"):
            token = self.note.data["data"]["token"]
            ticket = await self.kit.guard.issue_ticket(token)
            reply = await self.tools.call("confirm", {"token": token, "ticket": ticket})
            outcome = "done" if reply.status == "ok" else "unknown"
        else:
            outcome = "stopped"
        self.notebook.add(RESULT, f"{action}: {outcome}", 1, {"action": action, "outcome": outcome})

    def key(self, digit, caller_ms) -> None:
        pass


async def whole_call(kit, talker_ai, thinker_ai, find, turns: list[tuple[str | None, str]], step_s: float = 2.0):
    """One call: the Talker graph, and a Thinker graph for each job it opens, on the real
    backend. `turns`: (a note kind to wait for first, or None; then the caller's words)."""
    tools = BackendTools(kit.backend)
    holder: dict[str, Any] = {}

    def start_thinker() -> None:
        async def thinker() -> None:
            async def hand_over(reason: str) -> None:
                await tools.call("hand_over_to_human", {"reason": reason})
                call.runner.hand_over(reason)
            parts = ThinkerParts(call.notebook, tools, thinker_ai, hand_over,
                                 build_catalog(FoodPack().actions(), kit.settings), step_s=step_s)
            caller_done = await run_job(THINKER_GRAPH, parts, f"job-{id(parts)}")
            call.runner.job_closed(caller_done)
        holder["thinker"] = asyncio.create_task(thinker())

    call = Call(talker_ai, find)
    call.guard = RealGuard(kit, tools, call.notebook)
    call.parts.guard = call.guard
    call.parts.start_thinker = start_thinker
    await call.start()
    seen = 0
    for wait_for, words in turns:
        if wait_for is not None:
            for _ in range(int(300 * step_s)):  # bounded: a crashed graph must fail the test, not hang it
                if any(n.kind == wait_for for n in call.notebook.notes[seen:]):
                    break
                _raise_if_crashed(holder, call)
                await asyncio.sleep(0.01)
            else:
                raise AssertionError(f"no {wait_for} note came; notes: {call.kinds()}")
            seen = max(n.number for n in call.notebook.notes if n.kind == wait_for)
        await call.turn(words)
    await asyncio.wait_for(call.task, 5 * step_s)
    return call, tools


def _raise_if_crashed(holder: dict[str, Any], call: Call) -> None:
    """Stop the test with the real error if either graph died."""
    for task in (holder.get("thinker"), call.task):
        if task is not None and task.done() and task.exception() is not None:
            raise task.exception()


async def test_a_whole_cancel_call_end_to_end(open_call):
    kit = await open_call(FoodPack())
    opening = "Mujhe order cancel karna hai, number chaar paanch do ek"
    talker_ai = ScriptedTalkerAI(no_job=[("open_job", [])], collect=[("heard_number", [])],
                                 job_open=[(REPLY, ["Ji, theek hai."])])
    thinker_ai = ScriptedAI(check_number=[{"exit": "ok", "number": "4521"}], work_left=[{"exit": "done"}],
                            decide=[{"exit": "prepare", "tool": "prepare_cancel_order", "inputs": {},
                                     "plan": "Aapka order cancel kar raha hoon."}])
    call, tools = await whole_call(kit, talker_ai, thinker_ai, finder(**{opening: order("chaar paanch do ek")}),
                                   [(None, opening), (NEED_YES, "Haan, kar do"), (RESULT, "Nahi bas, thank you")])
    assert tools.calls == ["find_order", "prepare_cancel_order", "confirm"]
    assert (await kit.one("SELECT status FROM orders WHERE order_no = 4521"))["status"] == "cancelled"
    assert call.voice.said == [
        GREETING, line("talk", "quick"), line("talk", "number_heard"),
        line("talk", "number_back", number="4521"),
        line("talk", "order_found", number="4521", status=WORDS.phrase("status", "preparing", "hinglish")),
        "Aapka order cancel kar raha hoon.",
        line("result", "cancel_order_done"), line("talk", "anything_else"),
        "Ji, theek hai.", line("talk", "goodbye")]
    assert call.ended == ["the caller said goodbye"]


async def test_a_wrong_number_then_the_right_one_end_to_end(open_call):
    kit = await open_call(FoodPack())
    talker_ai = ScriptedTalkerAI(no_job=[("open_job", [])], collect=[("heard_number", [])] * 2,
                                 job_open=[(REPLY, ["Ji."])])
    # The second number is a caller turn while collecting: the Thinker checks it for "a person" first.
    thinker_ai = ScriptedAI(check_number=[{"exit": "ok", "number": "4599"}, {"exit": "ok", "number": "4521"}],
                            person=[{"exit": "no"}], decide=[{"exit": "lookup"}], work_left=[{"exit": "done"}])
    call, tools = await whole_call(kit, talker_ai, thinker_ai,
                                   finder(**{"Order 4599 kahan hai": order("4599"), "4521": order("4521")}),
                                   [(None, "Order 4599 kahan hai"), (FACT, "4521"), (CHECKED_NUMBER, "Bas, thanks")])
    assert tools.calls == ["find_order", "find_order"]
    assert line("ladder", "number_again") in call.voice.said
    assert call.voice.said[-1] == line("talk", "goodbye")


async def test_a_person_asked_for_mid_collect_hands_over_end_to_end(open_call):
    kit = await open_call(FoodPack())
    talker_ai = ScriptedTalkerAI(no_job=[("open_job", [])], collect=[(REPLY, ["Order number bataiye?"])] * 2)
    thinker_ai = ScriptedAI(person=[{"exit": "person"}])
    call, tools = await whole_call(kit, talker_ai, thinker_ai, finder(),
                                   [(None, "Cancel karna hai"), (None, "Mujhe insaan se baat karni hai")])
    assert tools.calls == ["hand_over_to_human"]
    assert call.voice.said[-1] == line("line", "hand_over") and call.ended == ["the caller asked for a person"]
