"""Agent C's limits (build step 6): code stops a Thinker going round in circles, follows the
retry table, and hands over at most once per call."""

import asyncio

from agents.agent_c.desk import JobDesk
from agents.agent_c.guard.guard import DONE, UNKNOWN_OUTCOME, Guard, GuardSettings
from agents.agent_c.guard.judge import NO, STOP, YES
from agents.agent_c.hand_over import HANDED_OVER, HandOver
from agents.agent_c.job import ENDED, Turns
from agents.agent_c.notebook import FACT, NEED_YES, RESULT, Notebook
from agents.agent_c.robot import WORDS, thinker_tools
from agents.agent_c.talker.language import StickyLanguage
from agents.agent_c.thinker.brain import Step, ToolCall
from agents.agent_c.thinker.copy_notes import note_for_reply
from agents.agent_c.thinker.loop import GOING_ROUND, Thinker, ThinkerLimits
from agents.common.call_box import CallBox, CallBoxSettings
from agents.common.diary import Diary
from agents.common.tools import BackendTools
from engine.contract.spoken import SpokenWords
from engine.service.replies import ERROR, OK, ToolReply
from packs.food.pack import FoodPack

RAHUL = {"order_no": 4521}
HAND_OVER = "hand_over_to_human"
LIMITS = ThinkerLimits(step_s=0.5, step_retries=0, lookup_s=0.5, idle_wait_s=0.05, lookup_retries=1, same_call_limit=3)
SETTINGS = GuardSettings(reask_wait_s=0.05, vad_start_s=0.2, stop_words={"ruko": STOP, "nahi": NO}, change_retries=2,
                         expired_reprepares=1)


class ListSink:
    def __init__(self) -> None:
        self.records: list[dict] = []

    def write(self, record: dict) -> None:
        self.records.append(record)


def diary():
    return Diary("c1", ListSink())


class ScriptedBrain:
    def __init__(self, *steps):
        self.steps = list(steps)
        self.seen = []

    async def step(self, news, results):
        self.seen.append((list(news), list(results)))
        return self.steps.pop(0) if self.steps else Step(text="")


class CountingTools:
    """The real backend, with a count of every call and an optional failure plan."""

    def __init__(self, backend, fail=None):
        self.inner = BackendTools(backend)
        self.calls = []
        self.fail = fail or (lambda tool, n: None)

    async def call(self, tool, inputs):
        self.calls.append(tool)
        return self.fail(tool, self.calls.count(tool)) or await self.inner.call(tool, inputs)

    async def catalog(self):
        return await self.inner.catalog()


def desk_and_hand_over(tools):
    book = Notebook(200)
    desk = JobDesk(book, Turns(), lambda job, d: None, diary())
    lines = []

    async def say_and_end(text):
        lines.append(text)

    hand_over = HandOver(tools, HAND_OVER, desk, say_and_end, lambda: "connecting you", diary())
    desk.on_hand_over = hand_over.now
    return desk, book, hand_over, lines


# ---- one hand-over --------------------------------------------------------------------


async def test_a_call_hands_over_once_says_so_once_and_the_job_ends(open_call):
    kit = await open_call(FoodPack())
    tools = CountingTools(kit.backend)
    desk, _, hand_over, lines = desk_and_hand_over(tools)
    desk.open_job("cancel")
    first = await hand_over.now("caller wants a human", 4521)
    second = await hand_over.now("again")
    assert first.status == OK and second.status == ERROR and tools.calls == [HAND_OVER]
    assert lines == ["connecting you"] and desk.job.state == ENDED and desk.job.end_reason == HANDED_OVER


async def run(kit, brain, tools):
    desk, book, hand_over, lines = desk_and_hand_over(tools)
    desk.open_job("cancel order 4521, Rahul Verma")
    allowed = {spec.name for spec in thinker_tools(await tools.catalog())}
    thinker = Thinker(desk.job, book, brain, tools, allowed, desk, LIMITS, diary(), hand_over_tool=HAND_OVER)
    task = asyncio.create_task(thinker.run())
    await asyncio.sleep(0.2)
    task.cancel()
    return desk, book, hand_over, lines


async def test_the_same_tool_inputs_and_answer_three_times_in_a_row_hands_over_once(open_call):
    kit = await open_call(FoodPack())
    tools = CountingTools(kit.backend)
    same = Step((ToolCall("find_order", RAHUL),))
    desk, _, hand_over, lines = await run(kit, ScriptedBrain(same, same, same, same), tools)
    assert tools.calls == ["find_order"] * 3 + [HAND_OVER]
    assert desk.job.end_reason == GOING_ROUND and hand_over.done and len(lines) == 1


async def test_the_thinkers_own_hand_over_goes_through_code_and_ends_the_job(open_call):
    kit = await open_call(FoodPack())
    tools = CountingTools(kit.backend)
    press = Step((ToolCall(HAND_OVER, {"reason": "caller asked"}),))
    desk, _, hand_over, lines = await run(kit, ScriptedBrain(press, press), tools)
    assert tools.calls == [HAND_OVER] and hand_over.reason == "caller asked" and lines == ["connecting you"]
    assert desk.job.end_reason == HANDED_OVER  # so the Thinker never presses it again


async def test_a_failed_lookup_is_tried_once_more_but_a_refusal_never(open_call):
    kit = await open_call(FoodPack())
    tools = CountingTools(kit.backend, fail=lambda tool, n: ToolReply(ERROR, "timed out") if n == 1 else None)
    brain = ScriptedBrain(Step((ToolCall("find_order", RAHUL),)), Step((ToolCall("find_order", {"order_no": 4599}),)),  # no such order: refused
                          Step(text="done"))
    await run(kit, brain, tools)
    assert tools.calls == ["find_order", "find_order", "find_order"]  # the error retried, the refusal not
    assert brain.seen[1][1][0][1]["status"] == OK and brain.seen[2][1][0][1]["status"] == "refused"


# ---- the Guard's retry table -------------------------------------------------------------


class YesJudge:
    async def label(self, question, words):
        return YES


async def guard_rig(kit, tools):
    desk, book, hand_over, lines = desk_and_hand_over(tools)
    desk.open_job("cancel 4521")
    said, marks = [], []

    async def say(text):
        said.append(text)

    async def send_mark(name):
        marks.append(name)

    handed = []

    async def hand_over_now(why):
        handed.append(why)

    guard = Guard(kit.guard, tools, CallBox(CallBoxSettings(change_wait_s=1.0)), book, desk, YesJudge(),
                  SpokenWords.from_yaml(WORDS), StickyLanguage(4), say, send_mark, SETTINGS, diary(), hand_over=hand_over_now)
    call = ToolCall("prepare_cancel_order", RAHUL)
    kind, text, data = note_for_reply(call, await kit.tool(call.name, **call.inputs))
    await guard.prepared(book.add(kind, text, 1, data))

    async def yes(at_ms):
        guard.speech_ended(False, said[-1])
        await asyncio.sleep(0.01)
        guard.played(marks[-1], at_ms)
        guard.turn_began(at_ms + 300)
        guard.turn_finished("haan")
        await asyncio.sleep(0.2)

    return guard, book, said, handed, yes


async def test_a_confirm_whose_answer_was_lost_is_sent_again_with_the_same_token_and_never_done_twice(open_call):
    kit = await open_call(FoodPack())

    class LosesFirst(CountingTools):
        async def call(self, tool, inputs):
            self.calls.append(tool)
            reply = await self.inner.call(tool, inputs)
            if tool == "confirm" and self.calls.count("confirm") == 1:
                raise ConnectionError("line broke after the save")
            return reply

    tools = LosesFirst(kit.backend)
    guard, book, _, handed, yes = await guard_rig(kit, tools)
    await yes(5000)
    assert tools.calls.count("confirm") == 2  # the second answered "already done"
    [result] = [note for note in book.notes if note.kind == RESULT]
    assert result.data["outcome"] == DONE and handed == []
    assert len(await kit.rows("SELECT * FROM refunds WHERE order_no = 4521")) <= 1


async def test_no_answer_after_every_retry_is_unknown_never_done_and_hands_over(open_call):
    kit = await open_call(FoodPack())

    class NeverAnswers(CountingTools):
        async def call(self, tool, inputs):
            self.calls.append(tool)
            if tool == "confirm":
                raise ConnectionError("no line")
            return await self.inner.call(tool, inputs)

    tools = NeverAnswers(kit.backend)
    guard, book, _, handed, yes = await guard_rig(kit, tools)
    await yes(5000)
    assert tools.calls.count("confirm") == 3  # one try and two retries (decision 19)
    [result] = [note for note in book.notes if note.kind == RESULT]
    assert result.data["outcome"] == UNKNOWN_OUTCOME and handed == ["the change's result is unknown"]


async def test_an_expired_token_is_prepared_again_and_asked_again(open_call):
    kit = await open_call(FoodPack())
    tools = CountingTools(kit.backend)
    guard, book, said, handed, yes = await guard_rig(kit, tools)
    kit.clock.advance(10_000)  # the prepared change runs out while the caller thinks
    await yes(5000)
    assert tools.calls.count("prepare_cancel_order") == 1 and len(said) == 2 and said[1] == said[0]
    assert (await kit.one("SELECT status FROM orders WHERE order_no = 4521"))["status"] == "preparing"
    await yes(9000)
    assert (await kit.one("SELECT status FROM orders WHERE order_no = 4521"))["status"] == "cancelled"
    assert [note.kind for note in book.notes].count(NEED_YES) == 1 and FACT not in [n.kind for n in book.notes]
