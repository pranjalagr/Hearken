"""The real pretend caller in whole lab calls: the real puppet robot as its own program,
the real phone line, director, recorder and scorer. Only the outside services are
fakes, so no key or request is spent."""

import asyncio
import json
from pathlib import Path

import pytest

from engine.tasks.task_file import load_tasks
from lab.caller.brain.brain import Brain
from lab.caller.chain import ChainCaller, ChainParts
from lab.caller.report_card import report_on_run
from lab.caller.settings import load_caller_settings
from lab.health.run_health import puppet_plan
from lab.runner.batch import make_run_dir, run_batch
from lab.runner.run_call import FINE, LAB_PROBLEM, PuppetRobot
from lab.scorer.score import read_score
from lab.settings import load_settings
from lab.synthetic_voice import is_speech, synthetic_line
from packs.food.pack import FoodPack

PRACTICE = Path(__file__).resolve().parents[2] / "packs" / "food" / "tasks" / "practice"
TASKS = {task.id: task for task in load_tasks(PRACTICE)}
LAB = load_settings().with_changes(robot_no_audio_s=3.0, robot_change_wait_s=5.0, reshuffle_rounds=200, calls_at_once=2)
SETTINGS = load_caller_settings()
PERFECT = PuppetRobot("perfect")


def answer(line, hindi, action="none", order=0, done=False):
    return {"line": line, "hindi_letters": hindi, "language": "hinglish", "agrees_to_action": action,
            "agrees_to_order": order, "press_key": "", "done": done}


class FakeVoice:
    async def pieces(self, text):
        return synthetic_line(text)


class FakeEars:
    """Hears the robot stop talking, and reports that it heard words: the puppet says
    beeps here, so there's nothing to transcribe."""

    name, rate = "fake", 8000

    def __init__(self, delay_chunks=3):
        self._quiet = 0
        self._spoke = False
        self._delay = delay_chunks
        self._turns = ["namaste bataiye", "order number aur naam", "kya cancel kar doon", "ho gaya dhanyavaad", "kuch aur", "achha phir"]

    async def stream(self, pieces, heard):
        async for chunk in pieces:
            speech = is_speech(chunk[:320]) or is_speech(chunk[-320:])
            if speech:
                self._spoke, self._quiet = True, 0
            elif self._spoke:
                self._quiet += 1
                if self._quiet >= self._delay:
                    self._spoke = False
                    # Different words each turn: the same words three times means a stuck robot (C12).
                    heard.final(self._turns.pop(0) if self._turns else "bas itna hi")


def maker(*answers, broken=None):
    remaining = list(answers)

    async def ask(system, messages):
        if broken is not None:
            raise broken
        return remaining.pop(0)

    def make(plan, on_event):
        parts = ChainParts(Brain(plan.task.caller, ask), FakeVoice(), FakeEars(), describe={"brain": "fake", "ears": "fake"})
        return ChainCaller(parts, plan.task, on_event, SETTINGS)

    return make


async def one_call(tmp_path, task_id, make):
    from dataclasses import replace

    plan = replace(puppet_plan(TASKS[task_id], PERFECT), caller=make)
    run_dir = make_run_dir(tmp_path, "chain")
    batch = await run_batch([plan], FoodPack(), run_dir, LAB)
    return batch, run_dir


def events_of(folder):
    return [json.loads(line) for line in (Path(folder) / "events.jsonl").read_text(encoding="utf-8").splitlines()]


@pytest.fixture(scope="module")
def cancel_call(tmp_path_factory):
    # One shared call: it takes half a minute. Its own loop, since it runs before any test.
    return asyncio.run(one_call(tmp_path_factory.mktemp("cancel"), "food_01_cancel_prepaid", maker(
        answer("4521, Rahul Verma.", "4521, राहुल वर्मा।"),
        answer("Haan, kar do.", "हां, कर दो।", "cancel_order", 4521),
        answer("Theek hai, thank you.", "ठीक है, थैंक यू।", done=True),
    )))


def test_a_whole_call_succeeds_and_is_scored_like_any_other(cancel_call):
    batch, _run_dir = cancel_call
    outcome = batch.outcomes[0]
    assert outcome.label == FINE, outcome.why
    score = read_score(Path(outcome.folder))
    assert score.records_right and not score.harms and score.success


def test_the_real_yes_and_every_line_reach_the_log(cancel_call):
    batch, _ = cancel_call
    events = events_of(batch.outcomes[0].folder)
    lines = [e for e in events if e["event"] == "caller.line"]
    assert [e["text"] for e in lines][:3] == ["Hello, mera order cancel karna hai.", "4521, Rahul Verma.", "Haan, kar do."]
    assert lines[2]["agrees"] == ["cancel_order", 4521]
    # The log keeps what was really read out, numbers in the task's style.
    assert lines[1]["spoken"] == "चार पांच दो एक, राहुल वर्मा।"
    assert [e for e in events if e["event"] == "caller.heard"]
    assert [e for e in events if e["event"] == "caller.thought"]


def test_what_the_caller_was_is_saved_with_the_call(cancel_call):
    batch, _ = cancel_call
    about = json.loads((Path(batch.outcomes[0].folder) / "caller.json").read_text(encoding="utf-8"))
    assert about["brain"] == "fake" and about["ears"] == "fake"


async def test_the_report_card_reads_the_run(cancel_call):
    _, run_dir = cancel_call
    report = await report_on_run(run_dir, SETTINGS, LAB.sound_level)
    assert report.calls == 1 and report.interruptions == 0 and report.caller_bugs == 0 and report.passes
    assert report.median_reply_gap_s is not None and (run_dir / "caller_report.md").exists()


async def test_a_race_line_hits_its_moment_with_the_chain_caller(tmp_path):
    batch, _ = await one_call(tmp_path, "food_07_ruko_while_asking", maker(
        answer("Haan, order 4521, Rahul Verma.", "हां, ऑर्डर 4521, राहुल वर्मा।"),
        answer("Haan, cancel kar do.", "हां, कैंसिल कर दो।"),
        answer("Theek hai.", "ठीक है।"),
        answer("Theek hai, thank you.", "ठीक है, थैंक यू।", done=True),
        answer("Thank you.", "थैंक यू।"),
        answer("Thank you.", "थैंक यू।"),
    ))
    outcome = batch.outcomes[-1]
    assert outcome.label == FINE, outcome.why
    assert outcome.race["hit"] is True, outcome.race
    assert read_score(Path(outcome.folder)).records_right


async def test_a_broken_brain_is_a_lab_problem_never_the_robots_fault(tmp_path):
    settings_no_rerun = LAB.with_changes(max_reruns=0)
    from dataclasses import replace

    plan = replace(puppet_plan(TASKS["food_01_cancel_prepaid"], PERFECT), caller=maker(broken=RuntimeError("limit reached")))
    batch = await run_batch([plan], FoodPack(), make_run_dir(tmp_path, "broken"), settings_no_rerun)
    outcome = batch.outcomes[0]
    assert outcome.label == LAB_PROBLEM and "limit reached" in outcome.why
    assert "kept breaking" in batch.stopped


async def test_a_brain_too_slow_to_answer_is_a_lab_problem_never_the_robots_fault(tmp_path):
    # Real calls on 4 Nov 2026: the puppet went on without answers and was marked with
    # harms for acting on no yes, on calls labelled fine. None of that may be counted.
    from dataclasses import replace

    async def ask(system, messages):
        await asyncio.sleep(3600)

    def make(plan, on_event):
        parts = ChainParts(Brain(plan.task.caller, ask), FakeVoice(), FakeEars(), describe={"brain": "fake", "ears": "fake"})
        return ChainCaller(parts, plan.task, on_event, replace(SETTINGS, reply_stuck_s=2.0))

    plan = replace(puppet_plan(TASKS["food_01_cancel_prepaid"], PERFECT), caller=make)
    run_dir = make_run_dir(tmp_path, "slow")
    batch = await run_batch([plan], FoodPack(), run_dir, LAB.with_changes(max_reruns=0))
    outcome = batch.outcomes[0]
    assert outcome.label == LAB_PROBLEM and "no reply" in outcome.why
    assert "kept breaking" in batch.stopped and not batch.scores
    assert (await report_on_run(run_dir, SETTINGS, LAB.sound_level)).calls == 0  # cut short: says nothing of the caller


def test_a_call_that_never_ended_is_only_the_robots_fault_if_the_lab_kept_running():
    from lab.runner.run_call import ROBOT_FAILURE, CallOutcome, _label_never_ended

    def label(late_ms):
        outcome = CallOutcome("c", "t", "r", "safe", 1, 1, FINE, "")
        _label_never_ended(outcome, late_ms, LAB)
        return outcome.label

    assert label(LAB.freeze_limit_ms) == ROBOT_FAILURE
    assert label(2_554_364.0) == LAB_PROBLEM  # the laptop was in standby


async def test_a_caller_that_cannot_start_is_a_lab_problem_and_never_dials(tmp_path):
    from dataclasses import replace

    class NoVoice(FakeVoice):
        async def pieces(self, text):
            raise RuntimeError("no voice credit")

    def make(plan, on_event):
        parts = ChainParts(Brain(plan.task.caller, None), NoVoice(), FakeEars())
        return ChainCaller(parts, plan.task, on_event, SETTINGS)

    plan = replace(puppet_plan(TASKS["food_01_cancel_prepaid"], PERFECT), caller=make)
    settings = LAB.with_changes(max_reruns=0, robot_no_audio_s=1.0, robot_change_wait_s=1.0)
    batch = await run_batch([plan], FoodPack(), make_run_dir(tmp_path, "nostart"), settings)
    outcome = batch.outcomes[0]
    assert outcome.label == LAB_PROBLEM and "no voice credit" in outcome.why
    assert not [e for e in events_of(outcome.folder) if e["event"] == "caller.line"]
