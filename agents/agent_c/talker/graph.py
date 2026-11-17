"""The Talker graph: one per call, from the greeting to the hang-up (decided 17 Nov 2026).

What it is, in plain words:
- The Talker does all the talking. It greets, answers simple things itself, opens a job
  when real work is needed, collects the order number, says the Thinker's notes as they
  come, runs the ask-again steps when an order isn't found, hands the yes moment to the
  Guard, and says goodbye or hands over.
- It never touches the backend: the Thinker does. They meet only in the notebook.
- Where the words must be exact or instant, it says a fixed line (talker/words.yaml).
  Where the content changes, its AI writes the line. A number is only ever said from the
  Thinker's checked number, filled in by code (decision 27).
- Between events it waits at the step it reached; it never starts over.

Events that wake it: a finished caller turn, a new note, a key press, a time-out (the
keypad wait), a hand-over, and the job closing. `TalkerRunner` turns them into wake-ups.

Built once when the program starts (`TALKER_GRAPH`), like the Thinker graph; each call
brings its own parts in `TalkerParts`.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
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
    FACT,
    FAILED,
    NEED_INFO,
    NEED_YES,
    ORDER_NUMBER,
    PLAN,
    REQUEST,
    RESULT,
    Note,
    Notebook,
)
from agents.agent_c.talker.number_finder import (FOR_THINKER, HEARD, NO_NUMBER, SHORT, FindNumbers, Heard,
                                                  what_was_heard)
from agents.agent_c.talker.slots import stray_numbers
from agents.agent_c.talker.talker_ai import REPLY, TalkerAI
from engine.contract.spoken import SpokenWords

# The kinds of event.
TURN, NOTE, KEY, TIMEOUT, HAND_OVER, JOB_CLOSED = "turn", "note", "key", "timeout", "hand_over", "job_closed"
# Notes the Talker writes itself: never said back to the caller.
OWN_NOTES = (REQUEST, CALLER, ORDER_NUMBER, ASKS_USED_UP)
# Decision 27's ask-again steps after "not found", in order (lines in words.yaml's "ladder").
LADDER = ("number_again", "number_digits", "number_keypad")
KEYPAD_END = "#"
# Notes not said once the caller has spoken since they were worked out (decision 16). The
# yes-question, results, questions and failures are always said: skipping one would leave
# the job waiting for an answer the caller never heard.
OLD_NEWS_SKIPPED = (PLAN, ANSWER, CHECKED_NUMBER, FACT)


class Voice(Protocol):
    async def say(self, text: str, again_if_cut: bool = False) -> None:
        """Say one line to the caller. `again_if_cut`: if the caller cuts it off, say it once
        more (a result line must be heard, decision 7)."""
        ...


class Guard(Protocol):
    async def prepared(self, note: Note) -> None:
        """A change waits for a yes: start asking (the Guard owns the yes moment)."""
        ...

    async def answer(self, words: str, start_ms: int | None) -> None:
        """The caller's answer to the yes-question."""
        ...

    def key(self, digit: str, caller_ms: int) -> None:
        """A key pressed during the yes moment (the keypad yes/no)."""
        ...


@dataclass
class TalkerParts:
    """Everything one call gives its Talker graph. Nothing here is shared between calls."""

    notebook: Notebook
    voice: Voice
    words: SpokenWords  # the fixed lines (talker/words.yaml) and the pack's phrases
    ai: TalkerAI
    find_numbers: FindNumbers  # the number finder (Sarvam)
    guard: Guard
    start_thinker: Callable[[], None]  # starts this job's Thinker graph
    end_call: Callable[[str], Awaitable[None]]  # hangs up
    greeting: str  # the call's first line (the same for every robot in the lab)
    language: Callable[[], str] = lambda: "hinglish"  # the sticky call language (decision 10)
    version: Callable[[], int] = lambda: 1  # the job's version, for each note (decision 16)
    new_turn: Callable[[], None] = lambda: None  # a caller turn during a job: older notes become old news
    reply_s: float = 8.0  # one AI reply's time limit (decision 30's Thinker step; a starting guess)
    # The Thinker's plan goes here to be said in place of the next waiting line (17 Nov 2026);
    # None: said at once (no waiting lines, as in tests of the graph alone).
    hold_plan: Callable[[str], None] | None = None
    on_error: Callable[[str], None] = lambda text: None  # the call's diary, for errors the call survives
    # The call's diary, for each Talker AI decision and finder answer (17 Nov 2026: food_05 lost
    # an order number five times and the diary couldn't show which part missed it).
    on_step: Callable[[str, dict[str, Any]], None] = lambda step, details: None
    number_length: int = 4
    keypad_s: float = 15.0  # time to type the number on the keypad (a starting guess)
    max_sentences: int = 2  # decision 8: 1-2 sentences per reply


class TalkState(TypedDict, total=False):
    """What the Talker graph remembers between events, for the whole call."""

    event: dict[str, Any]  # the event that woke the last waiting step
    mode: str  # where the call is waiting: no job, collecting, job open, yes moment, keypad
    misses: int  # how many times the order wasn't found (the ask-again count)
    typed: str  # digits typed on the keypad so far
    after: str  # where to go next


# ---- small helpers -------------------------------------------------------------------

def _parts(config: dict[str, Any]) -> TalkerParts:
    """This call's own parts, which come in with every run of the graph."""
    return config["configurable"]["parts"]


def _line(parts: TalkerParts, group: str, key: str, **values: str) -> str:
    """A fixed line in the call's language, with code's values filled in."""
    return parts.words.phrase(group, key, parts.language()).format(**values)


async def _say_fixed(parts: TalkerParts, group: str, key: str, **values: str) -> None:
    """Say one fixed line."""
    await parts.voice.say(_line(parts, group, key, **values))


def _known_numbers(parts: TalkerParts) -> list[str]:
    """Every text a number may come from: the caller's words, the backend's answers (copied
    by code) and the Thinker's checked number. A line an AI wrote is not a source: a
    number in nothing else was made up (decision 8)."""
    sources = (REQUEST, CALLER, FACT, NEED_YES, RESULT, CHECKED_NUMBER, ORDER_NUMBER)
    return [note.text + " " + str(note.data) for note in parts.notebook.notes if note.kind in sources]


def _safe(parts: TalkerParts, text: str) -> str:
    """An AI-written line as it may be spoken: at most two sentences, and never with a
    number nobody gave. A made-up number makes the whole line the safe line instead."""
    sentences = [s for s in text.replace("\n", " ").split(". ") if s.strip()][: parts.max_sentences]
    text = ". ".join(sentences).strip()
    if stray_numbers(text, _known_numbers(parts)):
        return _line(parts, "line", "safe")
    return text


# An amount of money in a line: "₹380", "380 rupaye", "380 rupees".
_MONEY = re.compile(r"₹\s?(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s*(?:rupaye|rupees|rupay|rs\.?)", re.IGNORECASE)


def _amounts(text: str) -> set[str]:
    return {a or b for a, b in _MONEY.findall(text)}


def _money_ok(parts: TalkerParts, text: str) -> bool:
    """Every amount in a Thinker AI line is the change's own (its yes-question or result) or
    one the caller said. 17 Nov 2026, food_08 and food_13: an answer said "₹100" and "₹380",
    real amounts from the order (an old refund, the order total) said as this refund's, and
    the caller said no to a right refund. A prompt rule didn't hold, so code checks."""
    allowed: set[str] = set()
    for note in parts.notebook.notes:
        if note.kind in (NEED_YES, RESULT, REQUEST, CALLER):
            allowed |= _amounts(note.text + " " + str(note.data))
    return _amounts(text) <= allowed


async def _say_ai_text(parts: TalkerParts, text: str) -> None:
    """Say a line the Thinker's AI wrote (a plan, a question, an explanation), made safe:
    no made-up number, and no amount that isn't the change's own or the caller's."""
    if text.strip():
        await parts.voice.say(_safe(parts, text) if _money_ok(parts, text) else _line(parts, "line", "safe"))


def _write(parts: TalkerParts, kind: str, text: str, data: dict[str, Any] | None = None) -> None:
    """Write one note to the notebook, for the Thinker to read."""
    parts.notebook.add(kind, text, parts.version(), data or {})


def _caller_words(parts: TalkerParts) -> list[str]:
    """Everything the caller said in this call, in order."""
    return [note.text for note in parts.notebook.notes if note.kind in (REQUEST, CALLER)]


async def _ai_reply(parts: TalkerParts, step: str, facts: dict[str, Any], exits: tuple[str, ...]) -> str:
    """Let the Talker's AI answer a turn: its sentences go to the voice as they come,
    each made safe first. Returns its exit ("reply", or the tool it called).
    A failed or slow request is asked once more if nothing was said yet; after that the
    safe line plays, and the call goes on (decision 30)."""
    said = 0

    async def speak(sentence: str) -> None:
        nonlocal said
        if said < parts.max_sentences and sentence.strip():
            said += 1
            await parts.voice.say(_safe(parts, sentence))

    for _ in range(2):
        try:
            exit_ = await asyncio.wait_for(
                parts.ai.reply(step, {**facts, "language": parts.language()}, exits, speak), parts.reply_s)
            parts.on_step(step, {"exit": exit_})
            return exit_
        except Exception:
            if said:
                break  # part of the reply was said: asking again would repeat it
    if not said:
        await _say_fixed(parts, "line", "safe")
    return REPLY


# ---- the waiting steps ------------------------------------------------------------------
# Each holds nothing but the wait: on waking, LangGraph runs a paused step again from its
# first line, so anything that talks or writes lives in another step.

def make_wait(mode: str) -> Callable[[TalkState, Any], TalkState]:
    """A waiting step for one place in the call. It waits for the next event (a caller
    turn, a note, a key, a time-out, a hand-over or the job closing) and remembers where
    the call is, so a note said meanwhile returns the call to the same place. At the
    keypad it waits at most keypad_s."""
    def wait(state: TalkState, config) -> TalkState:
        timeout = _parts(config).keypad_s if mode == "wait_keypad" else None
        return {"event": interrupt({"timeout": timeout, "mode": mode}), "mode": mode}
    wait.__name__ = mode
    return wait


# ---- the talking steps ----------------------------------------------------------------

async def greet(state: TalkState, config) -> TalkState:
    """Say the greeting (a pre-made clip in the real robot, decision 13)."""
    parts = _parts(config)
    await parts.voice.say(parts.greeting)
    return {"misses": 0, "typed": ""}


async def no_job_turn(state: TalkState, config) -> TalkState:
    """No job is open. The AI answers the turn, and decides: open a job for real work (or
    a person), end the call, or just reply (decision 7)."""
    parts = _parts(config)
    exit_ = await _ai_reply(parts, "no_job", {"caller_words": _caller_words(parts) + [state["event"]["words"]]},
                            ("open_job", "end_call"))
    if exit_ == "open_job":
        return {"after": "open_job"}
    if exit_ == "end_call":
        return {"after": "goodbye"}
    # The turn is kept in the notebook too, so a job opened later sees the whole call.
    _write(parts, CALLER, state["event"]["words"])
    return {"after": "wait_no_job"}


async def open_job(state: TalkState, config) -> TalkState:
    """Code opens the job: writes the job note with the caller's words, starts the
    Thinker graph, and plays the quick line, never with a number (17 Nov 2026)."""
    parts = _parts(config)
    _write(parts, REQUEST, state["event"]["words"])
    parts.start_thinker()
    await _say_fixed(parts, "talk", "quick")
    return {"misses": 0}


async def collect_turn(state: TalkState, config) -> TalkState:
    """Read a turn for the order number. The AI says whether it holds one; only then the
    finder points at the number words and code reads the digits (decision 27, 17 Nov).
    A full number: a line with no number, and the order-number note for the Thinker."""
    parts = _parts(config)
    words = state["event"]["words"]
    if state.get("mode") in ("wait_collect", "wait_keypad"):
        parts.new_turn()
        _write(parts, CALLER, words)  # every turn in a job reaches the Thinker (decision 7)
    exit_ = await _ai_reply(parts, "collect", {"caller_said": words}, ("heard_number",))
    if exit_ != "heard_number":
        return {"after": "wait_collect"}  # the AI answered and asked for the number
    numbers: list = []
    try:
        numbers = await parts.find_numbers(words)
        heard = what_was_heard(numbers, words, parts.number_length)
        parts.on_step("number finder", {"words": words, "numbers": numbers, "heard": heard.what, "digits": heard.digits})
    except Exception as error:
        # The finder failed (no credit, a timeout): the call goes on. The Thinker's AI reads
        # the number from the caller's words instead, as for words the reader can't read
        # (17 Nov 2026: a finder's 402 crashed the Talker and ended food_02).
        parts.on_error(f"number finder failed: {error!r}"[:300])
        heard = Heard(FOR_THINKER)
    if heard.what == HEARD or heard.what == FOR_THINKER:
        # Unreadable number words go to the Thinker's AI as they are (15 Nov 2026).
        await _say_fixed(parts, "talk", "number_heard")
        _write(parts, ORDER_NUMBER, heard.digits or words, {"digits": heard.digits, "words": words})
        return {"after": "wait_job"}
    if heard.what == NO_NUMBER and numbers:
        # The AI heard an order number and the finder found number words but didn't tag
        # them as the order ("45 30" tagged "other", 17 Nov 2026, food_02): the Thinker's AI
        # reads them with the whole call, as for unreadable number words.
        await _say_fixed(parts, "talk", "number_heard")
        _write(parts, ORDER_NUMBER, words, {"digits": "", "words": words})
        return {"after": "wait_job"}
    if heard.what == SHORT:
        await _say_fixed(parts, "talk", "number_again_full")
    else:
        await _say_fixed(parts, "talk", "ask_number")
    return {"after": "wait_collect"}


async def job_turn(state: TalkState, config) -> TalkState:
    """A caller turn while the Thinker works. It goes to the Thinker (decision 7), and the
    AI answers as usual: says back what it understood, never an outcome (decision 18)."""
    parts = _parts(config)
    words = state["event"]["words"]
    parts.new_turn()
    _write(parts, CALLER, words)
    notes = [note.text for note in parts.notebook.notes if note.kind not in OWN_NOTES][-6:]
    await _ai_reply(parts, "job_open", {"caller_said": words, "notes": notes}, ())
    return {}


async def yes_turn(state: TalkState, config) -> TalkState:
    """A caller turn during the yes moment: the Guard reads it (decisions 20-23). It is
    also written for the Thinker, as every turn in a job is (decision 7)."""
    parts = _parts(config)
    parts.new_turn()
    _write(parts, CALLER, state["event"]["words"])
    await parts.guard.answer(state["event"]["words"], state["event"].get("start_ms"))
    return {}


async def yes_key(state: TalkState, config) -> TalkState:
    """A key during the yes moment ("1 for yes, 2 for no"): the Guard reads it."""
    _parts(config).guard.key(state["event"]["digit"], state["event"].get("caller_ms", 0))
    return {}


async def say_note(state: TalkState, config) -> TalkState:
    """A new note from the Thinker, the backend or the Guard: say it the agreed way
    (fixed line or AI line, 17 Nov 2026), and choose where the call goes next."""
    parts = _parts(config)
    note = state["event"]["note"]
    kind, data = note["kind"], note["data"]
    if kind != PLAN and parts.hold_plan is not None:
        # The plan's work has moved on (an answer, a question, a result): a plan not yet
        # said is stale (17 Nov 2026, food_03: it came after the explanation of a refusal).
        parts.hold_plan("")
    if kind in OLD_NEWS_SKIPPED and note.get("version", 0) < parts.version():
        # The caller spoke after this was worked out: it may no longer hold (decision 16).
        return {"after": state.get("mode", "wait_job")}
    if kind == CHECKED_NUMBER:
        # The only number the caller ever hears: the Thinker's checked one (decision 27).
        await _say_fixed(parts, "talk", "number_back", number=note["text"])
        return {"after": "wait_job"}
    if kind == FACT and data.get("tool") == "find_order":
        if data.get("status") == "ok":
            fields = data.get("data", {})
            status = parts.words.phrase("status", str(fields.get("status")), parts.language())
            await _say_fixed(parts, "talk", "order_found", number=str(fields.get("order_no")), status=status)
            return {"misses": 0, "after": "wait_job"}
        if data.get("data", {}).get("wrong_input") == "order_no":
            return {"after": "ask_again"}
        return {"after": state.get("mode", "wait_job")}  # the lookup failed: the Thinker explains
    if kind == NEED_INFO and data.get("need") == "order_no":
        # The Thinker couldn't tell the number from the words: ask for it again.
        await _say_fixed(parts, "talk", "number_again_full")
        return {"after": "wait_collect"}
    if kind == PLAN and parts.hold_plan is not None:
        # Said in place of the next waiting line, keeping their pace (decided 17 Nov 2026).
        parts.hold_plan(_safe(parts, note["text"]))
        return {"after": state.get("mode", "wait_job")}
    if kind in (NEED_INFO, PLAN, ANSWER):
        # Written by the Thinker's AI: a question, its plan, or an explanation. The call
        # stays where it was (an answer during the yes moment keeps the yes moment).
        await _say_ai_text(parts, note["text"])
        return {"after": state.get("mode", "wait_job")}
    if kind == NEED_YES:
        await parts.guard.prepared(_as_note(note))
        return {"after": "wait_yes"}
    if kind == RESULT:
        # The Guard's result: a fixed line per action and outcome, then "anything else?".
        action, outcome = data.get("action", ""), data.get("outcome", "")
        if f"{action}_{outcome}" in parts.words.values("result"):
            # Said again if the caller cuts it off: a result must be heard (decision 7, 17 Nov).
            await parts.voice.say(_line(parts, "result", f"{action}_{outcome}"), again_if_cut=True)
        if outcome in ("done", "stopped"):
            # Only when the change is settled. A refused one gets its reason and the offer of a
            # person next (the Thinker explains); an unknown one is still being checked.
            await _say_fixed(parts, "talk", "anything_else")
        return {"after": "wait_job"}
    if kind == ANSWERED:
        # A lookup's answer is said (the found order); then, as after any result, the fixed
        # "anything else?" (17 Nov 2026), so the caller's reply really answers it.
        await _say_fixed(parts, "talk", "anything_else")
        return {"after": "wait_job"}
    if kind == FAILED:
        await _say_fixed(parts, "talk", "sorry")  # the hand-over follows (decision 14)
        return {"after": "wait_job"}
    return {"after": state.get("mode", "wait_job")}  # nothing to say for this note


async def ask_again(state: TalkState, config) -> TalkState:
    """The order wasn't found. Code counts the misses and says the next ask-again line:
    again, then digit by digit, then the keypad (decision 27). Past the keypad, code
    writes "asks used up", and the Thinker hands over (17 Nov 2026)."""
    parts = _parts(config)
    misses = state.get("misses", 0) + 1
    if misses > len(LADDER):
        _write(parts, ASKS_USED_UP, "still not found after asking again, digit by digit and by keypad")
        return {"misses": misses, "after": "wait_job"}
    await _say_fixed(parts, "ladder", LADDER[misses - 1])
    keypad = LADDER[misses - 1] == "number_keypad"
    return {"misses": misses, "typed": "", "after": "wait_keypad" if keypad else "wait_collect"}


async def keypad_key(state: TalkState, config) -> TalkState:
    """A key while the caller types the number. Done after four digits or "#": a typed
    number can't be misheard, so it goes to the Thinker marked as typed."""
    parts = _parts(config)
    digit = state["event"]["digit"]
    typed = state.get("typed", "") + (digit if digit.isdigit() else "")
    if digit != KEYPAD_END and len(typed) < parts.number_length:
        return {"typed": typed, "after": "wait_keypad"}
    if not typed:
        return {"typed": "", "after": "wait_keypad"}
    await _say_fixed(parts, "talk", "number_heard")
    _write(parts, ORDER_NUMBER, typed, {"digits": typed, "typed": True})
    return {"typed": "", "after": "wait_job"}


async def keypad_timeout(state: TalkState, config) -> TalkState:
    """Nothing typed in time: that counts as one more miss."""
    return {"after": "ask_again"}


async def goodbye(state: TalkState, config) -> TalkState:
    """The caller is done: say goodbye and hang up."""
    parts = _parts(config)
    _drop_plan(parts)
    await _say_fixed(parts, "talk", "goodbye")
    await parts.end_call("the caller said goodbye")
    return {}


async def hand_over(state: TalkState, config) -> TalkState:
    """The Thinker handed over: say the hand-over line, then end the call (decision 13)."""
    parts = _parts(config)
    _drop_plan(parts)
    await _say_fixed(parts, "line", "hand_over")
    await parts.end_call(str(state["event"].get("reason", "handed over")))
    return {}


def _drop_plan(parts: TalkerParts) -> None:
    """The call is ending: a plan not yet said must never come after the last line
    (17 Nov 2026, food_02: "Mai aapka order cancel kar raha hoon" after the hand-over line)."""
    if parts.hold_plan is not None:
        parts.hold_plan("")


def job_closed(state: TalkState, config) -> TalkState:
    """The Thinker's job is over: the call goes on with no job open."""
    return {"misses": 0, "typed": ""}


def _as_note(note: dict[str, Any]) -> Note:
    """A saved note back as a Note, for the Guard."""
    return Note(note["number"], note["kind"], note["text"], 1, note["data"])


# ---- where each event goes, by where the call is -----------------------------------------

def _route(mode: str):
    """The router for one waiting step: send the event to the step that handles it here."""
    def route(state: TalkState) -> str:
        event = state["event"]
        kind = event["kind"]
        if kind == HAND_OVER:
            return "hand_over"
        if kind == JOB_CLOSED:
            return "goodbye" if event.get("caller_done") else "job_closed"
        if kind == NOTE:
            return "say_note" if event["note"]["kind"] not in OWN_NOTES else mode
        if mode == "wait_no_job":
            return "no_job_turn" if kind == TURN else mode
        if mode == "wait_collect":
            return "collect_turn" if kind == TURN else mode
        if mode == "wait_keypad":
            return {KEY: "keypad_key", TIMEOUT: "keypad_timeout", TURN: "collect_turn"}.get(kind, mode)
        if mode == "wait_yes":
            return {TURN: "yes_turn", KEY: "yes_key"}.get(kind, mode)
        return "job_turn" if kind == TURN else mode  # wait_job
    return route


def _go(state: TalkState) -> str:
    """Follow the step's own choice of where to go next ("done" ends the call)."""
    return END if state.get("after") == "done" else state["after"]


# ---- building the graph -----------------------------------------------------------------

SAVER = InMemorySaver()  # one per program; each call's saved steps are deleted when it ends
WAITS = ("wait_no_job", "wait_collect", "wait_job", "wait_yes")
HANDLERS = ["open_job", "no_job_turn", "collect_turn", "job_turn", "yes_turn", "yes_key", "say_note", "ask_again",
            "keypad_key", "keypad_timeout", "goodbye", "hand_over", "job_closed"]


def build_talker_graph():
    """Join the steps with their allowed exits. Called once, at program start."""
    g = StateGraph(TalkState)
    for step in (greet, open_job, no_job_turn, collect_turn, job_turn, yes_turn, yes_key, say_note, ask_again,
                 keypad_key, keypad_timeout, goodbye, hand_over, job_closed):
        g.add_node(step.__name__, step)
    # The five places the call can wait.
    for name in (*WAITS, "wait_keypad"):
        g.add_node(name, make_wait(name))
    every = [*WAITS, "wait_keypad", *HANDLERS]
    g.add_edge(START, "greet")
    g.add_edge("greet", "wait_no_job")
    for name in (*WAITS, "wait_keypad"):
        g.add_conditional_edges(name, _route(name), every)
    for step in ("no_job_turn", "collect_turn", "say_note", "ask_again", "keypad_key", "keypad_timeout"):
        g.add_conditional_edges(step, _go, every)
    g.add_edge("open_job", "collect_turn")  # the opening turn may already hold the number
    g.add_edge("job_turn", "wait_job")
    g.add_edge("yes_turn", "wait_yes")
    g.add_edge("yes_key", "wait_yes")
    g.add_edge("job_closed", "wait_no_job")
    g.add_edge("goodbye", END)
    g.add_edge("hand_over", END)
    return g.compile(checkpointer=SAVER)


# ---- running one call ---------------------------------------------------------------------

class TalkerRunner:
    """Feeds one call's events to its Talker graph. The robot (and tests) call turn(),
    key(), hand_over() and job_closed(); new notes are picked up from the notebook."""

    def __init__(self, graph, parts: TalkerParts, call_id: str) -> None:
        self._graph = graph
        self._parts = parts
        self._config = {"configurable": {"thread_id": call_id, "parts": parts}, "recursion_limit": 500}
        self._events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._seen = 0  # the last note number already turned into an event
        self.mode = ""  # where the call waits now (wait_no_job, wait_collect, wait_job, ...)

    def turn(self, words: str, start_ms: int | None = None) -> None:
        """A finished caller turn."""
        self._events.put_nowait({"kind": TURN, "words": words, "start_ms": start_ms})

    def key(self, digit: str, caller_ms: int = 0) -> None:
        """A key the caller pressed."""
        self._events.put_nowait({"kind": KEY, "digit": digit, "caller_ms": caller_ms})

    def hand_over(self, reason: str) -> None:
        """The Thinker pressed the hand-over: say its line and end the call."""
        self._events.put_nowait({"kind": HAND_OVER, "reason": reason})

    def job_closed(self, caller_done: bool) -> None:
        """The Thinker's job is over. `caller_done`: the caller said they need nothing more,
        so the call ends with goodbye (decision 13)."""
        self._events.put_nowait({"kind": JOB_CLOSED, "caller_done": caller_done})

    async def run(self) -> None:
        """Run the call's Talker graph until it ends (goodbye or hand-over)."""
        try:
            result = await self._graph.ainvoke({}, self._config)
            while "__interrupt__" in result:
                paused = result["__interrupt__"][0].value
                self.mode = paused.get("mode", "")
                event = await self._next(paused.get("timeout"))
                result = await self._graph.ainvoke(Command(resume=event), self._config)
        finally:
            SAVER.delete_thread(self._config["configurable"]["thread_id"])

    async def _next(self, timeout_s: float | None) -> dict[str, Any]:
        """The next event: a queued one, else a new note, else wait for either (or the time-out)."""
        loop = asyncio.get_running_loop()
        deadline = None if timeout_s is None else loop.time() + timeout_s
        while True:
            if not self._events.empty():
                return self._events.get_nowait()
            for note in self._parts.notebook.after(self._seen):
                self._seen = note.number
                if note.kind not in OWN_NOTES:
                    return {"kind": NOTE, "note": {"number": note.number, "kind": note.kind, "text": note.text,
                                                   "version": note.version, "data": dict(note.data)}}
            left = None if deadline is None else deadline - loop.time()
            if left is not None and left <= 0:
                return {"kind": TIMEOUT}
            # Wake on whichever comes first: a queued event or a new note.
            get = asyncio.ensure_future(self._events.get())
            bell = asyncio.ensure_future(self._parts.notebook.wait(1.0))
            done, pending = await asyncio.wait({get, bell}, timeout=left, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            if get in done:
                return get.result()


# Built once, when the program starts.
TALKER_GRAPH = build_talker_graph()
