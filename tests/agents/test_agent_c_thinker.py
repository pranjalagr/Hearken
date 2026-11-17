"""Agent C's notebook, jobs and Thinker (build step 2): code copies every backend
answer into notes, versions hold stale notes, and a job closes only on the five checks."""

import asyncio

import pytest

from agents.agent_c.desk import PASSED_ON, STARTED, JobDesk
from agents.agent_c.job import ANSWER_GIVEN, CHECKS, CLOSED, ENDED, OPEN, Job, StepMark, Turns, failing_checks
from agents.agent_c.live_retry import try_live
from agents.agent_c.notebook import (ANSWER, CALLER, FACT, FAILED, NEED_INFO, NEED_YES, REQUEST, STILL_GOOD, Notebook,
                                     NotebookFull)
from agents.agent_c.robot import CountedBrain, thinker_tools
from agents.agent_c.talker.feed import NoteFeed
from agents.agent_c.thinker.brain import GeminiThinker, Step, ToolCall
from agents.agent_c.thinker.loop import THINKER_FAILED, Thinker, ThinkerLimits
from agents.common.diary import Diary
from agents.common.pipecat_brain_cap import BrainCap
from agents.common.tools import BackendTools
from engine.service.tool_catalog import ToolSpec
from packs.food.pack import FoodPack

LIMITS = ThinkerLimits(step_s=0.5, step_retries=1, lookup_s=0.5, idle_wait_s=0.05)
RAHUL = {"order_no": 4521}


class ListSink:
    def __init__(self) -> None:
        self.records: list[dict] = []

    def write(self, record: dict) -> None:
        self.records.append(record)


def diary() -> Diary:
    return Diary("c1", ListSink())


class ScriptedBrain:
    """Plays fixed steps; records what each step was given."""

    def __init__(self, *steps):
        self.steps = list(steps)
        self.seen: list[tuple[list[str], list]] = []

    async def step(self, news, results):
        self.seen.append((list(news), list(results)))
        step = self.steps.pop(0)
        if isinstance(step, BaseException):
            raise step
        if step == "hang":
            await asyncio.sleep(10)
        return step


def desk_with(notebook=None, turns=None):
    notebook = notebook or Notebook(200)
    turns = turns or Turns()
    started = []
    desk = JobDesk(notebook, turns, lambda job, d: started.append(job), diary())
    return desk, notebook, turns, started


# ---- the notebook ---------------------------------------------------------------


async def test_notes_are_numbered_in_order_and_a_full_notebook_refuses_more():
    book = Notebook(2)
    first = book.add(FACT, "a", 1)
    second = book.add(ANSWER, "b", 1)
    assert (first.number, second.number, book.newest) == (1, 2, 2)
    assert book.last((FACT,)) == first and book.after(1) == [second]
    with pytest.raises(NotebookFull):
        book.add(FACT, "c", 1)
    with pytest.raises(ValueError):
        Notebook(5).add("guess", "x", 1)


async def test_the_bell_wakes_a_waiter_and_every_wait_has_a_time_limit():
    book = Notebook(10)
    assert await book.wait(0.01) is False
    waiter = asyncio.create_task(book.wait(1.0))
    await asyncio.sleep(0)
    book.add(FACT, "found", 1)
    assert await waiter is True
    assert await book.wait(0.01) is False  # an old ring is never seen by a late waiter


# ---- the five close checks --------------------------------------------------------


def test_each_close_check_fails_on_its_own():
    book, turns, job = Notebook(50), Turns(), Job(1, "cancel")
    turns.finish()
    mark = StepMark(read_turn=1, begun=1, version=1)
    assert failing_checks(job, mark, turns, book, reply_spoken=True) == []
    assert failing_checks(job, StepMark(0, 1, 1), turns, book, True) == [CHECKS[0]]
    turns.begin()  # the caller started talking after the step began
    assert CHECKS[1] in failing_checks(job, mark, turns, book, True)
    turns2 = Turns()
    turns2.finish()
    book.add(NEED_INFO, "order number?", 1)
    assert failing_checks(job, mark, turns2, book, True) == [CHECKS[2]]
    book.add(CALLER, "4521", 1)
    assert failing_checks(job, mark, turns2, book, True) == []
    job.change_sent()
    assert failing_checks(job, mark, turns2, book, True) == [CHECKS[3]]
    job.change_answered()
    assert failing_checks(job, mark, turns2, book, False) == [CHECKS[4]]


# ---- the job desk -----------------------------------------------------------------


def test_only_one_job_at_a_time_and_a_second_request_goes_to_it():
    desk, book, _, started = desk_with()
    assert desk.open_job("cancel 4521") == STARTED
    assert desk.open_job("also the address") == PASSED_ON
    assert len(started) == 1 and [n.kind for n in book.notes] == [REQUEST, REQUEST]


def test_every_caller_turn_during_a_job_goes_to_the_thinker_and_bumps_the_version():
    desk, book, turns, _ = desk_with()
    desk.caller_finished("hello")  # no job: nothing for the Thinker
    assert book.notes == ()
    desk.open_job("cancel 4521")
    desk.caller_began()
    desk.caller_finished("haan haan")  # no word list: even this goes
    assert desk.job.version == 2 and book.notes[-1].kind == CALLER and book.notes[-1].version == 2


def test_a_job_closes_after_two_clean_stops_and_waits_for_the_reply_to_be_spoken():
    desk, book, turns, _ = desk_with()
    desk.open_job("where is 4521")
    reply = book.add(ANSWER, "found it", 1)
    desk.stopped(desk.mark(), reply)
    assert desk.job.state == OPEN  # check 5: not yet spoken
    desk.reply_spoken(reply.number)
    assert desk.job.state == ANSWER_GIVEN
    desk.caller_finished("theek hai")  # still goes to the Thinker
    desk.stopped(desk.mark(), None)
    assert desk.job.state == CLOSED


def test_a_turn_that_came_during_the_step_keeps_the_job_open():
    desk, book, turns, _ = desk_with()
    desk.open_job("where is 4521")
    mark = desk.mark()
    desk.caller_finished("aur address bhi")
    desk.stopped(mark, None)
    assert desk.job.state == OPEN


def test_a_crash_ends_the_job_at_once_with_a_failed_note():
    desk, book, _, _ = desk_with()
    desk.open_job("cancel")
    desk.on_crash("thinker", RuntimeError("boom"))
    assert desk.job.state == ENDED and book.notes[-1].kind == FAILED


# ---- the Thinker ------------------------------------------------------------------


async def run_thinker(kit, brain, desk=None, book=None):
    desk, book, turns, _ = desk_with(book) if desk is None else (desk, book, None, None)
    if desk.job is None:
        desk.open_job("cancel order 4521, Rahul Verma")
    tools = BackendTools(kit.backend)
    allowed = {spec.name for spec in thinker_tools(await tools.catalog())}
    prepared = []

    async def on_prepared(note):
        prepared.append(note)

    desk.on_prepared = on_prepared
    thinker = Thinker(desk.job, book, brain, tools, allowed, desk, LIMITS, diary())
    task = asyncio.create_task(thinker.run())
    return desk, book, task, prepared


async def stop_thinker(desk, task):
    desk.end_job("test over")
    await asyncio.wait_for(task, 1.0)


async def test_backend_answers_are_copied_into_notes_and_a_prepare_becomes_need_yes(open_call):
    kit = await open_call(FoodPack())
    brain = ScriptedBrain(Step((ToolCall("find_order", RAHUL),)), Step((ToolCall("prepare_cancel_order", RAHUL),)),
                          Step(text="Need: the caller's yes"))
    desk, book, task, prepared = await run_thinker(kit, brain)
    await asyncio.sleep(0.1)
    fact = book.last((FACT,))
    assert fact.data["tool"] == "find_order" and fact.data["data"]["order_no"] == 4521  # the backend's own answer
    yes = book.last((NEED_YES,))
    assert yes.data["action"] == "cancel_order" and yes.data["data"]["token"] and prepared == [yes]
    assert brain.seen[0][0] == ["Request: cancel order 4521, Rahul Verma"]
    assert brain.seen[1][1][0][1]["status"] == "ok"  # the step after a tool reads its answer
    assert book.last((NEED_INFO,)).text == "the caller's yes"
    await stop_thinker(desk, task)


async def test_the_thinker_can_never_press_confirm(open_call):
    kit = await open_call(FoodPack())
    token = (await kit.tool("prepare_cancel_order", **RAHUL)).data["token"]
    brain = ScriptedBrain(Step((ToolCall("confirm", {"token": token}),)), Step(text="done"))
    desk, book, task, _ = await run_thinker(kit, brain)
    await asyncio.sleep(0.1)
    assert brain.seen[1][1][0][1]["status"] == "error"
    assert (await kit.one("SELECT status FROM orders WHERE order_no = 4521"))["status"] == "preparing"
    await stop_thinker(desk, task)


def test_agent_c_refuses_a_mode_without_confirm():
    with pytest.raises(ValueError):
        thinker_tools([ToolSpec("cancel_order", "", {})])


async def test_a_hung_step_is_retried_once_then_a_failed_note_ends_the_job(open_call):
    kit = await open_call(FoodPack())
    brain = ScriptedBrain("hang", "hang")
    desk, book, task, _ = await run_thinker(kit, brain)
    await asyncio.wait_for(task, 3.0)
    assert len(brain.seen) == 2 and book.notes[-1].kind == FAILED and desk.job.end_reason == THINKER_FAILED


async def test_the_callers_new_words_reach_the_next_step_and_still_good_is_a_note(open_call):
    kit = await open_call(FoodPack())
    brain = ScriptedBrain(Step(text="Order 4521 is being prepared."), Step(text="still good"))
    desk, book, task, _ = await run_thinker(kit, brain)
    await asyncio.sleep(0.1)
    desk.caller_finished("acha, aur kitna time?")
    await asyncio.sleep(0.2)
    assert brain.seen[1][0] == ["Caller: acha, aur kitna time?"]
    assert book.last((STILL_GOOD,)).version == 2
    await stop_thinker(desk, task)


# ---- the note feed ----------------------------------------------------------------


async def test_the_feed_holds_old_notes_until_still_good_and_never_speaks_over_the_caller():
    desk, book, turns, _ = desk_with()
    told = []

    async def push(text):
        told.append(text)

    feed = NoteFeed(book, desk, turns, push, 0.05)
    desk.open_job("where is 4521")
    book.add(FACT, "found 4521", 1)
    turns.begin()
    await feed.give_new_notes()
    assert told == []  # the caller is talking
    desk.caller_finished("aur address?")  # version 2: the fact is now old
    await feed.give_new_notes()
    assert told == []
    book.add(STILL_GOOD, "still good", 2)
    await feed.give_new_notes()
    assert told == ["NOTES: found 4521"]


async def test_newer_work_replaces_held_notes_and_a_fully_spoken_reply_is_reported():
    desk, book, turns, _ = desk_with()
    told = []

    async def push(text):
        told.append(text)

    feed = NoteFeed(book, desk, turns, push, 0.05)
    desk.open_job("where is 4521")
    book.add(FACT, "old", 1)
    desk.caller_finished("nahi, 4512")
    answer = book.add(ANSWER, "found 4512", 2)
    await feed.give_new_notes()
    assert told == ["NOTES: found 4512"]
    desk.stopped(desk.mark(), answer)
    feed.speech_ended(cut=False)  # no Talker reply yet: not the answer's speech
    assert desk.job.state == OPEN
    feed.talker_replied()
    feed.speech_ended(cut=True)  # cut off: not fully spoken
    assert desk.job.state == OPEN
    feed.talker_replied()
    feed.speech_ended(cut=False)
    assert desk.job.state == ANSWER_GIVEN


# ---- the real brain's shape, the retries and the cap -------------------------------


async def test_try_live_retries_a_timeout_once_but_never_a_bad_request():
    calls = []

    async def slow():
        calls.append(1)
        await asyncio.sleep(1)

    with pytest.raises(TimeoutError):
        await try_live(slow, 0.02, 1)
    assert len(calls) == 2

    async def bad():
        calls.append(2)
        raise ValueError("bad key")

    with pytest.raises(ValueError):
        await try_live(bad, 0.5, 1)
    assert calls.count(2) == 1


async def test_thinker_requests_count_against_the_calls_brain_cap():
    reached = []

    async def on_reached(n):
        reached.append(n)

    cap = BrainCap(2, on_reached)
    brain = CountedBrain(ScriptedBrain(Step(text="a"), Step(text="b")), cap)
    await brain.step([], [])
    await brain.step([], [])
    assert reached == [2]


async def test_the_gemini_thinker_sends_tools_without_additionalproperties_and_keeps_history_only_on_success():
    from google.genai import types

    class Models:
        def __init__(self):
            self.asked = []
            self.fail = True

        async def generate_content(self, model, contents, config):
            self.asked.append((model, list(contents), config))
            if self.fail:
                self.fail = False
                raise TimeoutError()
            call = types.Part(function_call=types.FunctionCall(name="find_order", args={"order_no": 4521}))
            return types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(role="model", parts=[call]))])

    models = Models()
    client = type("Client", (), {"aio": type("Aio", (), {"models": models})()})()
    spec = ToolSpec("find_order", "Look up.", {"type": "object", "properties": {"order_no": {"type": "integer"}},
                                               "required": ["order_no"], "additionalProperties": False})
    brain = GeminiThinker(client, "flash-lite", "system", [spec])
    with pytest.raises(TimeoutError):
        await brain.step(["Request: where is 4521"], [])
    step = await brain.step(["Request: where is 4521"], [])
    assert step.calls == (ToolCall("find_order", {"order_no": 4521}),)
    first, second = models.asked
    assert len(first[1]) == len(second[1]) == 1  # the failed try left no trace
    declared = second[2].tools[0].function_declarations[0]
    assert "additionalProperties" not in declared.parameters_json_schema
