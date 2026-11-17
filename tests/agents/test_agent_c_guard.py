"""Agent C's Guard (build step 4): code plays the backend's yes-question, a yes counts
only after the "played" receipt and with the judge's "yes", code presses confirm
shielded, and the outcome comes only from the backend's answer."""

import asyncio

import pytest

from agents.agent_c.call_job import CallJob
from agents.agent_c.guard.guard import DONE, STOPPED, Guard, GuardSettings
from agents.agent_c.guard.judge import NO, OTHER, STOP, UNCLEAR, YES, TimedJudge, read_label, word_list_label
from agents.agent_c.notebook import NEED_YES, RESULT, YES_ANSWER, Notebook
from agents.agent_c.robot import WORDS
from agents.agent_c.talker.language import StickyLanguage
from agents.agent_c.thinker.brain import ToolCall
from agents.agent_c.thinker.copy_notes import note_for_reply
from agents.common.call_box import CallBox, CallBoxSettings
from agents.common.diary import Diary
from agents.common.tools import BackendTools
from engine.contract.spoken import SpokenWords
from engine.lab_controls.normal_delays import NormalDelays
from packs.food.pack import FoodPack

RAHUL = {"order_no": 4521}
STOP_WORDS = {"ruko": STOP, "mat": STOP, "wait": STOP, "stop": STOP, "nahi": NO}
SETTINGS = GuardSettings(reask_wait_s=0.05, vad_start_s=0.2, stop_words=STOP_WORDS)


class ListSink:
    def __init__(self) -> None:
        self.records: list[dict] = []

    def write(self, record: dict) -> None:
        self.records.append(record)


class FakeJudge:
    def __init__(self, *labels):
        self.labels = list(labels)
        self.asked = []

    async def label(self, question, words):
        self.asked.append(words)
        return self.labels.pop(0)


class Rig:
    """One call's Guard on a real backend, with a fake voice and wire."""

    def __init__(self, kit, judge):
        self.kit = kit
        self.book = Notebook(200)
        self.desk = CallJob()  # the job's version and its changes in flight (17 Nov 2026)
        self.said, self.marks, self.sink = [], [], ListSink()
        self.box = CallBox(CallBoxSettings(change_wait_s=5.0))

        async def say(text):
            self.said.append(text)

        async def send_mark(name):
            self.marks.append(name)

        self.handed_over = []

        async def hand_over(why):
            self.handed_over.append(why)

        self.guard = Guard(kit.guard, BackendTools(kit.backend), self.box, self.book, self.desk, judge,
                           SpokenWords.from_yaml(WORDS), StickyLanguage(4), say, send_mark, SETTINGS,
                           Diary("c1", self.sink), hand_over=hand_over)

    async def prepare(self):
        call = ToolCall("prepare_cancel_order", RAHUL)
        reply = await self.kit.tool(call.name, **call.inputs)
        kind, text, data = note_for_reply(call, reply)
        assert kind == NEED_YES
        await self.guard.prepared(self.book.add(kind, text, self.desk.job.version, data))

    async def question_played(self, at_ms=5000):
        self.guard.speech_ended(False, self.said[-1])
        await asyncio.sleep(0.01)
        self.guard.played(self.marks[-1], at_ms)

    async def caller_says(self, words, started_ms):
        self.guard.turn_began(started_ms + 200)  # the voice detector speaks up 0.2 s late
        self.guard.turn_finished(words)
        await asyncio.sleep(0.05)

    async def order_status(self):
        return (await self.kit.one("SELECT status FROM orders WHERE order_no = 4521"))["status"]

    def results(self):
        return [note for note in self.book.notes if note.kind == RESULT]


async def test_a_real_yes_after_the_question_played_confirms_and_the_result_is_the_backends(open_call):
    kit = await open_call(FoodPack())
    rig = Rig(kit, FakeJudge(YES))
    await rig.prepare()
    assert rig.said[0].startswith("Order 4521 cancel") and rig.said[0].endswith("Haan ya nahi?")  # the backend's words
    await rig.question_played(at_ms=5000)
    await rig.caller_says("haan kar do", started_ms=5100)
    assert await rig.order_status() == "cancelled"
    [result] = rig.results()
    assert result.data["outcome"] == DONE and rig.desk.job.changes_running == 0
    assert rig.book.last((YES_ANSWER,)).text == "yes: haan kar do"
    moment = [r for r in rig.sink.records if r.get("what") == "yes moment"][0]
    assert (moment["label"], moment["early"], moment["played_ms"], moment["start_ms"]) == (YES, False, 5000, 5100)


async def test_haan_haan_while_the_robot_is_still_asking_never_counts(open_call):
    kit = await open_call(FoodPack())
    rig = Rig(kit, FakeJudge(YES))
    await rig.prepare()
    await rig.caller_says("haan haan", started_ms=1000)  # no "played" receipt yet
    await asyncio.sleep(0.1)
    assert await rig.order_status() == "preparing" and rig.results() == []
    assert rig.said[-1] == "Haan ya nahi?"  # the caller went quiet: asked again, short


async def test_a_haan_that_began_before_the_question_ended_never_counts(open_call):
    kit = await open_call(FoodPack())
    rig = Rig(kit, FakeJudge(YES, YES))
    await rig.prepare()
    await rig.question_played(at_ms=5000)
    await rig.caller_says("haan", started_ms=4800)
    await asyncio.sleep(0.1)
    assert await rig.order_status() == "preparing"
    await rig.question_played(at_ms=7000)
    await rig.caller_says("haan", started_ms=7300)
    assert await rig.order_status() == "cancelled"


async def test_nahi_or_ruko_needs_only_the_word_list_and_nothing_is_sent(open_call):
    kit = await open_call(FoodPack())
    judge = FakeJudge()
    rig = Rig(kit, judge)
    await rig.prepare()
    await rig.question_played()
    await rig.caller_says("nahi rehne do", started_ms=5200)
    assert judge.asked == [] and await rig.order_status() == "preparing"
    [result] = rig.results()
    assert result.data["outcome"] == STOPPED and rig.guard.moment is None


async def test_yes_but_with_a_condition_is_other_answered_and_asked_again_uncounted(open_call):
    kit = await open_call(FoodPack())
    rig = Rig(kit, FakeJudge(OTHER))
    await rig.prepare()
    await rig.question_played()
    await rig.caller_says("haan but refund card par chahiye", started_ms=5200)
    await asyncio.sleep(0.1)
    assert rig.book.last((YES_ANSWER,)).text.startswith("other:")
    assert rig.said[-1] == rig.said[0]  # the full question again: "other" does not climb the ladder
    assert await rig.order_status() == "preparing"


async def test_unclear_climbs_the_ladder_to_the_keypad_and_then_a_human(open_call):
    kit = await open_call(FoodPack())
    rig = Rig(kit, FakeJudge(UNCLEAR, UNCLEAR, UNCLEAR))
    await rig.prepare()
    for asked in ("Maaf kijiye, sunai nahi diya. Haan ya nahi?", "Haan ke liye 1 dabaiye, nahi ke liye 2."):
        await rig.question_played()
        await rig.caller_says("hmm", started_ms=5200)
        await asyncio.sleep(0.1)
        assert rig.said[-1] == asked
    await rig.question_played()
    await rig.caller_says("hmm", started_ms=5200)
    assert rig.results()[0].data["outcome"] == STOPPED
    assert rig.handed_over == ["no clear yes or no, even on the keypad"]  # build step 6: the one hand-over


async def test_a_keypad_1_after_the_question_played_is_a_yes(open_call):
    kit = await open_call(FoodPack())
    rig = Rig(kit, FakeJudge())
    await rig.prepare()
    await rig.question_played(at_ms=5000)
    rig.guard.key("1", 4000)  # pressed before it played: ignored
    await asyncio.sleep(0.05)
    assert await rig.order_status() == "preparing"
    rig.guard.key("1", 5600)
    await asyncio.sleep(0.1)
    assert await rig.order_status() == "cancelled"


async def test_ruko_right_after_the_yes_stops_the_ticket_before_the_save(open_call):
    # The backend waits 0.8 s before its save; a "ruko" heard meanwhile cancels the ticket.
    kit = await open_call(FoodPack(), real_sleep=True, delays=NormalDelays(lookup_s=0.0, change_s=0.8))
    rig = Rig(kit, FakeJudge(YES))
    await rig.prepare()
    await rig.question_played()
    await rig.caller_says("haan", started_ms=5200)
    await asyncio.sleep(0.1)
    rig.guard.partial("ruko ruko")
    await asyncio.sleep(1.2)
    assert await rig.order_status() == "preparing"
    [result] = rig.results()
    assert result.data["outcome"] == STOPPED


async def test_a_question_cut_off_by_the_caller_is_asked_again(open_call):
    kit = await open_call(FoodPack())
    rig = Rig(kit, FakeJudge())
    await rig.prepare()
    rig.guard.speech_ended(True, "Order 4521")
    await asyncio.sleep(0.1)
    assert len(rig.said) == 2 and rig.marks == []


def test_the_word_list_and_the_judges_one_word():
    assert word_list_label("nahi, ruko", STOP_WORDS) == STOP
    assert word_list_label("Nahi.", STOP_WORDS) == NO
    assert word_list_label("haan theek hai", STOP_WORDS) is None
    assert (read_label(" Yes."), read_label("maybe yes"), read_label("")) == (YES, UNCLEAR, UNCLEAR)


async def test_a_slow_judge_is_unclear_never_yes_and_counts_as_a_request():
    requests, errors = [], []

    async def slow(question, words):
        await asyncio.sleep(1)
        return "yes"

    async def counted():
        requests.append(1)

    judge = TimedJudge(slow, 0.05, counted, errors.append)
    assert await judge.label("Haan ya nahi?", "haan") == UNCLEAR
    assert requests == [1] and errors


@pytest.mark.parametrize("module", ["agents.agent_c.thinker.graph", "agents.agent_c.thinker.step_brain",
                                    "agents.agent_c.talker.graph", "agents.agent_c.talker.talker_ai"])
def test_no_ai_facing_part_can_reach_a_ticket(module):
    import importlib

    names = vars(importlib.import_module(module))
    assert "GuardDoor" not in names and "issue_ticket" not in str(names.get("__doc__", ""))


async def test_a_question_the_voice_speaks_in_two_pieces_is_still_marked_once_its_end_plays(open_call):
    # 16 Nov 2026, food_02: the voice said "... wapas kar doon?" and then "Haan ya nahi?" as
    # two pieces; each was checked alone, no mark was ever sent, and a real yes was early.
    kit = await open_call(FoodPack())
    rig = Rig(kit, FakeJudge(YES))
    await rig.prepare()
    question = rig.said[-1]
    first, last = question.rsplit("Haan ya nahi?", 1)[0], "Haan ya nahi?"
    rig.guard.speech_ended(False, first)
    assert rig.marks == []  # not the end yet
    rig.guard.speech_ended(False, last)
    await asyncio.sleep(0.01)
    assert len(rig.marks) == 1
    rig.guard.played(rig.marks[-1], 5000)
    await rig.caller_says("haan cancel kar dijiye", started_ms=5100)
    assert await rig.order_status() == "cancelled"


async def test_the_judge_asks_several_keys_at_once_and_takes_the_first_answer():
    # The owner's choice, 16 Nov 2026: Flash-Lite took 1-9 s per request in a slow spell.
    from agents.agent_c.guard.judge import first_answer

    delays = iter([5.0, 0.01, 5.0])  # one of three answers quickly
    started, finished = [], []

    async def ask(question, words):
        delay = next(delays)
        started.append(delay)
        await asyncio.sleep(delay)
        finished.append(delay)
        return "yes"

    assert await asyncio.wait_for(first_answer(ask, 3)("q", "haan"), 1) == "yes"
    await asyncio.sleep(0.05)
    assert sorted(started) == [0.01, 5.0, 5.0] and finished == [0.01]  # the slow two were cancelled

    async def broken(question, words):
        raise RuntimeError("429")

    with pytest.raises(RuntimeError):
        await first_answer(broken, 2)("q", "haan")  # every racer failed: unclear, never yes
