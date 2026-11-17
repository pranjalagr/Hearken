"""The Thinker graph: one job, from its first step to its end (decided 17 Nov 2026).

What it is, in plain words:
- A job opens. The graph waits for the Talker's order number, checks it, finds the order,
  decides what to do, writes its plan, presses the tool, waits for the caller's yes (the
  Guard takes it), and ends when nothing is left to do.
- In each step the AI may think and decide, but only among that step's own exits. Code
  checks the exit, presses every tool, and copies every backend answer into the notebook.
- The Thinker never talks to the caller. It writes notes; the Talker says them.
- When a step must wait (for the caller, or for the Guard), the graph pauses there. The
  next matching note wakes it at the same step, not at the start.

Two rules from the LangGraph test (17 Nov 2026):
- Build the graph once, when the program starts (`THINKER_GRAPH` below). Building it
  during a call froze the loop for over 100 ms with 20 calls.
- A step that presses a tool never also waits: on waking, LangGraph runs the paused step
  again from its first line, and a test step that did both pressed its tool twice.

Each call brings its own notebook, tools and AI in `ThinkerParts`; the graph holds none.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from agents.agent_c.notebook import (
    ANSWER,
    ANSWERED,
    ASKS_USED_UP,
    CALLER,
    CHECKED_NUMBER,
    NEED_INFO,
    ORDER_NUMBER,
    PLAN,
    REQUEST,
    RESULT,
    YES_ANSWER,
    Note,
    Notebook,
)
from agents.agent_c.thinker.brain import ToolCall
from agents.agent_c.thinker.copy_notes import note_for_reply
from agents.agent_c.thinker.step_brain import StepBrain
from common.retry import worth_retrying
from engine.service.replies import ERROR, OK, REFUSED, ToolReply
from engine.service.tool_catalog import CONFIRM, PREPARE_PREFIX, ToolSpec

FIND_ORDER = "find_order"
HAND_OVER = "hand_over_to_human"
NUMBER_LENGTH = 4  # the food pack's order numbers


class Tools(Protocol):
    async def call(self, name: str, inputs: dict[str, Any]) -> ToolReply: ...


@dataclass
class ThinkerParts:
    """Everything one call gives its Thinker graph. Nothing here is shared between calls."""

    notebook: Notebook
    tools: Tools  # the backend's tools
    brain: StepBrain  # the AI for each step
    # Presses the hand-over tool (with the order, when known) and ends the call.
    hand_over: Callable[[str, int | None], Awaitable[None]]
    menu: Sequence[ToolSpec]  # the tools the AI may choose from (the safe-mode catalog)
    version: Callable[[], int] = lambda: 1  # the job's version, for each note (decision 16)
    step_s: float = 8.0  # one AI step's time limit (decision 30)
    retries: int = 1  # an AI step or a failed lookup is tried once more (decisions 19, 30)
    same_call_limit: int = 3  # the same tool, inputs and answer this often: going round in circles
    # The call's language, for anything the caller will hear (the plan, a question, an
    # explanation). 17 Nov 2026: without it an explanation came out in English in a Hinglish call.
    language: Callable[[], str] = lambda: "hinglish"
    on_step: Callable[[str, dict[str, Any]], None] = lambda step, answer: None  # the diary: each AI answer
    step_version: int = 0  # the job's version when the running step began (set by _parts)
    # The note kinds the graph is paused for, or () while it runs a step: the robot reads
    # this to know the Thinker is working, for the waiting lines (decision 11).
    waiting_for: tuple[str, ...] = ()


class JobState(TypedDict, total=False):
    """What the graph remembers between steps for one job."""

    note: dict[str, Any]  # the note that woke the last waiting step
    number: str  # the checked order number
    order: dict[str, Any]  # find_order's answer
    choice: dict[str, Any]  # the AI's last decision in "decide"
    refusal: str  # the backend's last refusal, shown to the AI in "decide"
    asked: str  # what the caller was last asked, so the AI reading their reply knows what it answers
    refused: list[dict[str, Any]]  # every change the backend refused in this job: never tried again
    explained: int  # how many times this job explained that something can't be done
    pending: list[str]  # new requests said during the yes-question, taken up once it ends
    handle_now: str  # the pending request "decide" is working on, if any
    after: str  # which step reads the caller's next reply
    reason: str  # why the job hands over
    pressed: dict[str, int]  # how often each (tool, inputs, answer) happened


class ThinkerStuck(RuntimeError):
    """An AI step gave no allowed answer, even when asked again. The call box turns this
    into a "failed" note, and the Talker apologises and hands over (decision 14)."""


# ---- small helpers -------------------------------------------------------------------

def _parts(config: dict[str, Any]) -> ThinkerParts:
    """This call's own parts, which come in with every run of the graph. Each step calls
    this first, so it also notes the job's version as the step begins."""
    parts = config["configurable"]["parts"]
    parts.step_version = parts.version()
    return parts


def _write(parts: ThinkerParts, kind: str, text: str, data: dict[str, Any] | None = None) -> None:
    """Write one note to the notebook, for the Talker (and the log) to read. It carries the
    version from when its step began: if the caller spoke since, the Talker treats it as
    old news (decision 16)."""
    parts.notebook.add(kind, text, parts.step_version or parts.version(), data or {})


def _caller_words(parts: ThinkerParts) -> list[str]:
    """Everything the caller said in this call, in order: the job note and each turn.
    The AI always sees the whole call (decision 15)."""
    return [note.text for note in parts.notebook.notes if note.kind in (REQUEST, CALLER)]


def _note_dict(note: Note) -> dict[str, Any]:
    """A note as plain data, so the graph can keep it in its saved state."""
    return {"number": note.number, "kind": note.kind, "text": note.text, "data": dict(note.data)}


async def _think(parts: ThinkerParts, step: str, facts: dict[str, Any], exits: set[str]) -> dict[str, Any]:
    """Ask the AI one step's question. Only an answer with an allowed exit counts: anything
    else, a timeout or an error is asked once more, then the job is stuck."""
    for _ in range(1 + parts.retries):
        try:
            answer = await asyncio.wait_for(parts.brain.ask(step, facts), parts.step_s)
        except (TimeoutError, ValueError, KeyError):
            continue  # too slow, or not JSON: ask once more
        except Exception as error:
            # A busy or broken service is asked once more (common/retry.py's rules). A daily
            # limit fails at once: on 17 Nov 2026 a key's 500 free requests were used up, and
            # asking again only wastes time (the plan's rule).
            if worth_retrying(error) and "PerDay" not in str(error):
                continue
            raise ThinkerStuck(f"the AI service failed at step {step!r}: {error}") from error
        if answer.get("exit") in exits:
            parts.on_step(step, answer)  # every AI decision in the robot's diary
            return answer
    raise ThinkerStuck(f"the AI gave no allowed answer at step {step!r} (allowed: {sorted(exits)})")


async def _press(parts: ThinkerParts, name: str, inputs: dict[str, Any]) -> ToolReply:
    """Press one backend tool. A lookup or prepare that errors is tried once more (decision
    19); neither changes anything, so trying again is safe."""
    reply = await parts.tools.call(name, inputs)
    for _ in range(parts.retries):
        if reply.status != ERROR:
            break
        reply = await parts.tools.call(name, inputs)
    return reply


def _copy(parts: ThinkerParts, name: str, inputs: dict[str, Any], reply: ToolReply) -> None:
    """Code copies the backend's answer into the notebook, exactly as it came (decision 16)."""
    kind, text, data = note_for_reply(ToolCall(name, inputs), reply)
    _write(parts, kind, text, data)


def _menu(parts: ThinkerParts) -> list[dict[str, Any]]:
    """The tools the AI may pick in "decide", with what each needs. The order number,
    the hand-over and confirm are left out: code fills the number, the hand-over is its
    own exit, and only the Guard confirms."""
    menu = []
    for spec in parts.menu:
        if spec.name in (FIND_ORDER, HAND_OVER, CONFIRM):
            continue
        required = set(spec.input_schema.get("required", []))
        inputs = {}
        for name, schema in spec.input_schema.get("properties", {}).items():
            if name == "order_no":
                continue
            # The allowed values too: on 17 Nov 2026 the AI wrote a refund reason as "raita
            # missing", which no tool takes, because the menu showed only the description.
            inputs[name] = {"means": schema.get("description", ""), "needed": name in required,
                            **({"only": schema["enum"]} if "enum" in schema else {})}
        menu.append({"tool": spec.name, "does": spec.description, "inputs": inputs})
    return menu


# ---- the steps -------------------------------------------------------------------------
# Each step gets the job's state and this call's config, and returns what it changed.
# A step that waits holds nothing but the wait.

def wait_number(state: JobState, config) -> JobState:
    """Wait for the Talker's order-number note. Also wakes on the caller's turns (to catch
    "I want a person") and on the Talker's "asks used up" note."""
    return {"note": interrupt({"wait": [ORDER_NUMBER, ASKS_USED_UP, CALLER]})}


def after_wait_number(state: JobState) -> str:
    """Which way after waiting for the number."""
    kind = state["note"]["kind"]
    if kind == ORDER_NUMBER:
        return "check_number"
    if kind == ASKS_USED_UP:
        return "hand_over"
    return "person_check"


async def person_check(state: JobState, config) -> JobState:
    """The AI reads one caller turn said while the number is being collected: does the
    caller want a person? (Decided 17 Nov 2026: hand over at once, from any step.)"""
    parts = _parts(config)
    answer = await _think(parts, "person", {"caller_said": state["note"]["text"]}, {"person", "no"})
    if answer["exit"] == "person":
        return {"reason": "the caller asked for a person", "after": "hand_over"}
    return {"after": "wait_number"}


async def check_number(state: JobState, config) -> JobState:
    """The AI checks the Talker's number against the caller's own words, then code writes
    the checked number: the only number used for the lookup or said to the caller
    (decision 27, 17 Nov 2026)."""
    parts = _parts(config)
    if state["note"]["data"].get("typed"):
        # Typed on the keypad: it can't be misheard, and it isn't in the caller's words.
        number = str(state["note"]["data"]["digits"])
        _write(parts, CHECKED_NUMBER, number, {"digits": number, "typed": True})
        return {"number": number, "after": "find_order"}
    facts = {"reader_digits": state["note"]["data"].get("digits", ""), "caller_words": _caller_words(parts)}
    answer = await _think(parts, "check_number", facts, {"ok", "unclear"})
    number = str(answer.get("number", "")).strip()
    if answer["exit"] == "ok" and number.isdigit() and len(number) == NUMBER_LENGTH:
        _write(parts, CHECKED_NUMBER, number, {"digits": number})
        return {"number": number, "after": "find_order"}
    _write(parts, NEED_INFO, "Ask the caller to say the order number again.", {"need": "order_no"})
    return {"after": "wait_number"}


async def find_order(state: JobState, config) -> JobState:
    """Code looks the order up with the checked number and copies the answer. Not found:
    the Talker reads the copied note and asks again, so this goes back to waiting."""
    parts = _parts(config)
    inputs = {"order_no": int(state["number"])}
    reply = await _press(parts, FIND_ORDER, inputs)
    _copy(parts, FIND_ORDER, inputs, reply)
    if reply.status == OK:
        return {"order": dict(reply.data), "after": "decide"}
    if reply.status == REFUSED:
        return {"after": "wait_number"}
    return {"reason": f"the order lookup did not answer: {reply.message}", "after": "explain"}


async def decide(state: JobState, config) -> JobState:
    """The AI decides the request: answer from the order (a lookup), prepare a change with
    one of the menu's tools, ask for a missing detail, say no tool can do it, or hand
    over. It also writes its plan in words."""
    parts = _parts(config)
    facts = {"caller_words": _caller_words(parts), "order": state.get("order", {}), "menu": _menu(parts),
             "last_refusal": state.get("refusal", ""), "already_refused": state.get("refused", []),
             "already_done": [note.text for note in parts.notebook.notes if note.kind == RESULT],
             "handle_now": state.get("handle_now", ""),
             "language": parts.language()}
    answer = await _think(parts, "decide", facts, {"lookup", "prepare", "missing", "cannot", "person"})
    exit_ = answer["exit"]
    if exit_ == "prepare" and answer.get("tool") not in {item["tool"] for item in _menu(parts)}:
        # A tool that isn't on the menu is never pressed: treat it as "nothing can do it".
        exit_, answer = "cannot", {**answer, "reason": f"there is no tool {answer.get('tool')!r}"}
    if exit_ == "lookup":
        return {"choice": answer, "after": "complete"}
    if exit_ == "prepare":
        return {"choice": answer, "after": "write_plan"}
    if exit_ == "missing":
        _write(parts, NEED_INFO, str(answer.get("ask", "")), {"need": answer.get("need", "")})
        return {"choice": answer, "after": "wait_reply_decide"}  # the reply comes back here
    if exit_ == "person":
        return {"reason": "the caller asked for a person", "after": "hand_over"}
    return {"reason": str(answer.get("reason", "no tool can do this")), "after": "explain"}


def write_plan(state: JobState, config) -> JobState:
    """Code writes the AI's plan to the notebook before the tool is pressed. The Talker
    says it instead of the next waiting line (decision 16, 17 Nov 2026)."""
    parts = _parts(config)
    plan = str(state["choice"].get("plan", "")).strip()
    if plan:
        _write(parts, PLAN, plan)
    return {}


async def press_tool(state: JobState, config) -> JobState:
    """Code presses the tool the AI chose, with the checked order number (never one the AI
    typed), and copies the answer. Prepared: wait for the yes. Refused because of one
    input: back to decide with the reason. Refused otherwise: explain."""
    parts = _parts(config)
    choice = state["choice"]
    name = str(choice["tool"])
    inputs = {**{k: v for k, v in dict(choice.get("inputs") or {}).items() if k != "order_no"},
              "order_no": int(state["number"])}
    # The owner's rule (17 Nov 2026): a change the backend refused is never pressed again with
    # the same inputs, whatever the AI decides; its reason still holds, so it is explained
    # (or, if already explained, handed over). Only a refusal naming a wrong input invites a
    # fixed try. 17 Nov 2026, food_02: the same refused cancel was tried three more times.
    for old in state.get("refused", []):
        if old["tool"] == name and old.get("inputs") == inputs:
            return {"reason": old["reason"], "after": "explain"}
    reply = await _press(parts, name, inputs)
    _copy(parts, name, inputs, reply)
    # Decision 19: the same tool, inputs and answer again and again is going round in circles.
    key = json.dumps([name, inputs, reply.status, reply.message], sort_keys=True, ensure_ascii=False)
    pressed = {**state.get("pressed", {}), key: state.get("pressed", {}).get(key, 0) + 1}
    if pressed[key] >= parts.same_call_limit:
        return {"pressed": pressed, "reason": "the same step kept failing", "after": "hand_over"}
    # A change went ahead: the job moved on, so an earlier explanation no longer counts
    # towards "the caller still insists" (17 Nov 2026, food_12: an explanation from before
    # the cancel made the next, unrelated "can't" hand over at once).
    if reply.status == OK and name.startswith(PREPARE_PREFIX) and reply.data.get("token"):
        return {"pressed": pressed, "refusal": "", "explained": 0, "after": "wait_yes"}
    if reply.status == OK:
        return {"pressed": pressed, "refusal": "", "explained": 0, "after": "complete"}
    if reply.status == REFUSED and reply.data.get("wrong_input"):
        return {"pressed": pressed, "refusal": reply.message, "after": "decide"}
    refused = [*state.get("refused", []), {"tool": name, "inputs": inputs, "reason": reply.message}]
    return {"pressed": pressed, "refused": refused, "reason": reply.message, "after": "explain"}


def wait_yes(state: JobState, config) -> JobState:
    """Wait while the Guard asks the yes-question and takes the answer. It writes the
    caller's answer and, once known, the result (decisions 17, 20-23)."""
    return {"note": interrupt({"wait": [RESULT, YES_ANSWER]})}


def after_wait_yes(state: JobState) -> str:
    """The result: the change is finished, one way or another. An "other" answer ("kitna
    milega?"): the AI answers it, and the Guard asks again. A plain yes or no: keep waiting
    for the result."""
    note = state["note"]
    if note["kind"] == RESULT and note["data"].get("outcome") == "refused":
        return "refused_result"
    if note["kind"] == RESULT:
        return "complete"
    if note["text"].startswith("other:"):
        return "answer_other"
    return "wait_yes"


def refused_result(state: JobState, config) -> JobState:
    """The backend refused the change at confirm (the world moved during the question: the
    rider picked the order up). Remember it, and explain it with its reason, as a refusal at
    prepare is (decisions 8, 19). 17 Nov 2026, food_02: it was followed by "anything else?"
    with no reason, and the AI then tried the same cancel three more times."""
    _parts(config)
    data = state["note"]["data"]
    choice = state.get("choice", {})
    # The inputs it was prepared with, as press_tool built them, so the same change is matched.
    inputs = {**{k: v for k, v in dict(choice.get("inputs") or {}).items() if k != "order_no"},
              "order_no": int(state["number"])}
    refused = [*state.get("refused", []), {"tool": PREPARE_PREFIX + str(data.get("action", "")),
                                           "inputs": inputs, "reason": data.get("reason", "")}]
    return {"refused": refused, "reason": str(data.get("reason", ""))}


async def answer_other(state: JobState, config) -> JobState:
    """The caller said something other than yes or no to the yes-question. The AI either
    answers a question about the change, from the facts only, or sees a new request ("aur
    rider ka number bhi chahiye"): it says it will take it up next, and the request is kept
    for after the yes moment (the user's choice, 17 Nov 2026, after food_10 lost one). The
    Guard asks the yes again either way."""
    parts = _parts(config)
    said = state["note"]["text"].removeprefix("other:").strip()
    facts = {"caller_said": said, "order": state.get("order", {}), "change": state.get("choice", {}),
             "language": parts.language()}
    answer = await _think(parts, "answer_other", facts, {"answer", "new_request"})
    _write(parts, ANSWER, str(answer.get("text", "")))
    if answer["exit"] == "new_request":
        return {"pending": [*state.get("pending", []), str(answer.get("request") or said)]}
    return {}


def complete(state: JobState, config) -> JobState:
    """The request is answered or the change is finished: its note is already in the
    notebook (the found order, or the Guard's result). The Talker says it and asks
    "anything else?"; the caller's reply decides if work is left. A lookup has no result
    note, so code writes one: 17 Nov 2026, food_07, a status answer got no "anything else?",
    the caller's next "haan" was read as done, and the robot hung up on a caller who still
    wanted a cancel."""
    parts = _parts(config)
    pending = state.get("pending", [])
    if pending:
        # A request said during the yes-question comes next, in the same job (decision 7).
        return {"pending": pending[1:], "handle_now": pending[0], "after": "decide"}
    if state.get("choice", {}).get("exit") == "lookup":
        _write(parts, ANSWERED, "answered from the order")
    return {"handle_now": "", "after": "wait_reply_complete"}


async def explain(state: JobState, config) -> JobState:
    """The AI writes why it can't be done, in plain words, and offers a person (decisions
    8, 19). Only facts: the reason comes from the backend or from the AI's own decision."""
    parts = _parts(config)
    if state.get("explained", 0) >= 1:
        # Explained once already and the caller still insists: one hand-over (the plan's
        # "refused -> explain -> caller still insists -> hand over", food_05). 17 Nov 2026,
        # food_06: without it the robot explained eight times and ran out of requests.
        return {"reason": "the caller still insists on something this line can't do", "after": "hand_over"}
    answer = await _think(parts, "explain", {"reason": state.get("reason", ""), "order": state.get("order", {}),
                                             "language": parts.language()}, {"note"})
    text = str(answer.get("text", ""))
    _write(parts, ANSWER, text, {"reason": state.get("reason", "")})
    return {"asked": text, "explained": state.get("explained", 0) + 1, "after": "wait_reply_explain"}


def wait_reply(state: JobState, config) -> JobState:
    """Wait for the caller's next turn. A new order number also wakes it."""
    return {"note": interrupt({"wait": [CALLER, ORDER_NUMBER]})}


def after_wait_reply(state: JobState) -> str:
    """A new order number starts again from the check. A caller turn goes to the step
    that asked: decide (a missing detail), the explanation, or "anything else?"."""
    if state["note"]["kind"] == ORDER_NUMBER:
        return "check_number"
    return {"wait_reply_decide": "decide", "wait_reply_explain": "after_explain",
            "wait_reply_complete": "work_left"}[state["after"]]


async def after_explain(state: JobState, config) -> JobState:
    """The AI reads the caller's reply to the explanation: a person, accept it, or ask for
    something else."""
    parts = _parts(config)
    # What the robot said is part of the question: "haan" to "connect you to a person?" is
    # a person (17 Nov 2026, food_03: read without it as "accepts", and the call ended).
    answer = await _think(parts, "after_explain", {"robot_said": state.get("asked", ""),
                                                   "caller_said": state["note"]["text"]},
                          {"person", "accepts", "changed"})
    if answer["exit"] == "person":
        return {"reason": "the caller asked for a person", "after": "hand_over"}
    return {"after": "done" if answer["exit"] == "accepts" else "decide"}


async def work_left(state: JobState, config) -> JobState:
    """The AI reads the caller's answer to "anything else?": done, more for this order
    (the same job, decision 7), or a person."""
    parts = _parts(config)
    # What was already done in this call, so a "haan" or "ho gaya?" is never read as asking for
    # it again (17 Nov 2026, food_03: the refund was tried twice more after it was done).
    done = [note.text for note in parts.notebook.notes if note.kind == RESULT]
    answer = await _think(parts, "work_left", {"robot_asked": "Anything else I can help with?",
                                               "already_done": done, "caller_said": state["note"]["text"]},
                          {"done", "more", "person"})
    if answer["exit"] == "person":
        return {"reason": "the caller asked for a person", "after": "hand_over"}
    return {"after": "decide" if answer["exit"] == "more" else "done"}


async def hand_over(state: JobState, config) -> JobState:
    """Code presses the hand-over, with the checked order number when there is one, so the
    person taking over knows the order (17 Nov 2026, run 2: every hand-over was saved with no
    order, and the lab scored the right hand-overs in food_05 and food_06 as wrong changes).
    The call ends once its line has played."""
    number = state.get("number")
    await _parts(config).hand_over(state.get("reason", "the caller needs a person"), int(number) if number else None)
    return {}


def _go(state: JobState) -> str:
    """Follow the step's own choice of where to go next ("done" is the job's end)."""
    return END if state["after"] == "done" else state["after"]


# ---- building the graph --------------------------------------------------------------

# One checkpoint store for the program. Each job's saved steps are deleted when it ends.
SAVER = InMemorySaver()


def build_thinker_graph():
    """Join the steps with their allowed exits. Called once, at program start."""
    g = StateGraph(JobState)
    steps = [wait_number, person_check, check_number, find_order, decide, write_plan, press_tool, wait_yes,
             refused_result, answer_other, complete, explain, after_explain, work_left, hand_over]
    for step in steps:
        g.add_node(step.__name__, step)
    # Three waits for the caller's reply share one step, told apart by who asked.
    for name in ("wait_reply_decide", "wait_reply_explain", "wait_reply_complete"):
        g.add_node(name, wait_reply)

    g.add_edge(START, "wait_number")
    g.add_conditional_edges("wait_number", after_wait_number, ["check_number", "hand_over", "person_check"])
    g.add_conditional_edges("person_check", _go, ["hand_over", "wait_number"])
    g.add_conditional_edges("check_number", _go, ["find_order", "wait_number"])
    g.add_conditional_edges("find_order", _go, ["decide", "wait_number", "explain"])
    g.add_conditional_edges("decide", _go, ["complete", "write_plan", "wait_reply_decide", "hand_over", "explain"])
    g.add_edge("write_plan", "press_tool")
    g.add_conditional_edges("press_tool", _go, ["wait_yes", "complete", "decide", "explain", "hand_over"])
    g.add_conditional_edges("wait_yes", after_wait_yes, ["complete", "refused_result", "answer_other", "wait_yes"])
    g.add_edge("refused_result", "explain")
    g.add_edge("answer_other", "wait_yes")
    g.add_conditional_edges("complete", _go, ["decide", "wait_reply_complete"])
    g.add_conditional_edges("explain", _go, ["wait_reply_explain", "hand_over"])
    for name in ("wait_reply_decide", "wait_reply_explain", "wait_reply_complete"):
        # The wait's own name tells where the caller's reply goes.
        g.add_conditional_edges(name, _reply_router(name), ["check_number", "decide", "after_explain", "work_left"])
    g.add_conditional_edges("after_explain", _go, ["hand_over", "decide", END])
    g.add_conditional_edges("work_left", _go, ["hand_over", "decide", END])
    g.add_edge("hand_over", END)
    return g.compile(checkpointer=SAVER)


def _reply_router(name: str) -> Callable[[JobState], str]:
    """Where a caller's reply goes, for the wait named `name`."""
    def route(state: JobState) -> str:
        return after_wait_reply({**state, "after": name})
    return route


# ---- running one job -----------------------------------------------------------------

async def next_note(notebook: Notebook, kinds: Sequence[str], seen: int) -> tuple[Note, int]:
    """The first note of one of these kinds after note number `seen`, waiting for it if it
    hasn't come yet. Notes of other kinds are passed over (the Guard reads the caller's
    words during a yes; the Thinker's own notes need no answer)."""
    while True:
        for note in notebook.after(seen):
            seen = note.number
            if note.kind in kinds:
                return note, seen
        await notebook.wait(1.0)


async def run_job(graph, parts: ThinkerParts, job_id: str, seen: int = 0) -> bool:
    """Run one job's Thinker graph to its end. Each time it pauses, wait for the note it
    asked for and wake it with that note. `seen`: start reading after this note number.
    Returns True when the job ended because the caller had nothing more (so the Talker
    says goodbye), False when it ended in a hand-over."""
    config = {"configurable": {"thread_id": job_id, "parts": parts}, "recursion_limit": 200}
    try:
        result = await graph.ainvoke({}, config)
        while "__interrupt__" in result:
            kinds = result["__interrupt__"][0].value["wait"]
            parts.waiting_for = tuple(kinds)
            note, seen = await next_note(parts.notebook, kinds, seen)
            parts.waiting_for = ()
            result = await graph.ainvoke(Command(resume=_note_dict(note)), config)
        return result.get("after") == "done"
    finally:
        # The job is over: its saved steps go, so memory doesn't grow call after call.
        SAVER.delete_thread(job_id)


# Built once, when the program starts (the LangGraph test's rule).
THINKER_GRAPH = build_thinker_graph()
