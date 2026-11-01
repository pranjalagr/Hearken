"""The whole test lab, end to end, with real calls (Part 5, L3-L14).

Puppet robots run as their own programs over MCP, the scripted caller and the
director dial them over the real phone line, the backend rings its bells, the
recorder keeps both sides, and the scorer marks every call from its folder.
The puppets' right answers are known in advance, so every number is checked.

One batch of calls is shared by most tests here (it takes a minute or two);
the break-on-purpose tests (L4) each run one call of their own.
"""

import asyncio
import dataclasses
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from engine.tasks.task_file import load_tasks
from lab.dummy_robot.crashing import CRASH_CODE
from lab.health.check import GREEN, caller_light, clock_light, leaks_light
from lab.health.run_health import puppet_plan
from lab.results.table import summarize
from lab.runner import run_call as run_call_module
from lab.runner.batch import make_run_dir, run_batch, write_run_json
from lab.runner.run_call import FINE, LAB_PROBLEM, ROBOT_FAILURE, CallOutcome, CallPlan, PuppetRobot, _judge_break
from lab.scorer.score import read_score, score_call
from lab.settings import load_settings
from packs.food.pack import FoodPack

PRACTICE = Path(__file__).resolve().parents[2] / "packs" / "food" / "tasks" / "practice"
TASKS = {task.id: task for task in load_tasks(PRACTICE)}
# Short robot timeouts, so a silent drop ends in seconds, not the real 30 s.
SETTINGS = load_settings().with_changes(robot_no_audio_s=3.0, robot_change_wait_s=5.0, reshuffle_rounds=500)
PERFECT, CARELESS = PuppetRobot("perfect"), PuppetRobot("careless")

SELECTED = [
    ("food_01_cancel_prepaid", PERFECT),
    ("food_03_missing_item_reply_lost", CARELESS),
    ("food_05_too_late_hand_over", PERFECT),
    ("food_07_ruko_while_asking", PERFECT),
    ("food_07_ruko_while_asking", CARELESS),
    ("food_08_haan_haan_while_asking", CARELESS),
    ("food_09_cash_cancel_yes_with_condition", CARELESS),
    ("food_10_wrong_name", CARELESS),
    ("food_11_two_orders_switches_language", PERFECT),
    ("food_12_ruko_after_sending", PERFECT),
    ("food_13_refund_then_address", PERFECT),
    ("food_14_address_corrected_while_saving", PERFECT),
    ("food_15_ruko_while_saving_then_drop", PERFECT),
]


@pytest.fixture(scope="module")
def lab_run(tmp_path_factory):
    run_dir = make_run_dir(tmp_path_factory.mktemp("lab"), "e2e")
    plans = [puppet_plan(TASKS[task_id], robot) for task_id, robot in SELECTED]
    write_run_json(run_dir, plans, [PERFECT, CARELESS], SETTINGS, {"kind": "test"})
    batch = asyncio.run(run_batch(plans, FoodPack(), run_dir, SETTINGS))
    scores = {(s.task_id, s.robot): s for s in batch.scores}
    return run_dir, batch, scores


def folder_of(batch, task_id, robot):
    last = [o for o in batch.outcomes if o.task_id == task_id and o.robot == robot.name][-1]
    return Path(last.folder)


# ---- every call ran and kept its whole black box (L3, L5, L14) ------------------------


def test_every_planned_call_ended_scored_and_nothing_is_the_robots_fault(lab_run):
    _run_dir, batch, scores = lab_run
    assert not batch.stopped
    assert len(scores) == len(SELECTED)
    assert all(o.label in (FINE, LAB_PROBLEM) for o in batch.outcomes)  # a lab problem is rerun, never blamed
    assert all(s.label == FINE for s in scores.values())


def test_each_call_folder_holds_its_whole_black_box(lab_run):
    _run_dir, batch, _scores = lab_run
    expected = {"caller.json", "events.jsonl", "caller.flac", "robot.flac", "clock.json", "start.db", "final.db",
                "robot.jsonl", "backend.jsonl", "outcome.json", "score.json"}
    for task_id, robot in SELECTED:
        names = {path.name for path in folder_of(batch, task_id, robot).iterdir()}
        assert expected <= names, (task_id, expected - names)


def test_the_run_folder_names_its_code_and_keeps_a_copy_of_every_setting(lab_run):
    run_dir, _batch, _scores = lab_run
    about = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert len(about["code"]["commit"]) == 40 and isinstance(about["code"]["dirty"], bool)
    assert about["lab_settings"]["robot_no_audio_s"] == 3.0 and "race_running_after_s" in about["director_settings"]
    assert set(about["robots"]) == {"puppet:perfect", "puppet:careless"} and about["scorer_version"]


def test_the_robots_diary_has_the_core_kinds_and_is_its_own_program(lab_run):
    _run_dir, batch, _scores = lab_run
    diary = [json.loads(line) for line in
             (folder_of(batch, "food_07_ruko_while_asking", PERFECT) / "robot.jsonl").read_text("utf-8").splitlines()]
    kinds = {line["kind"] for line in diary}
    assert {"said", "heard", "tool", "decision"} <= kinds
    assert any(line["kind"] == "decision" and "stopped talking" in line["what"] for line in diary)


# ---- the marks: known right answers (L7-L10) --------------------------------------------


@pytest.mark.parametrize(("task_id", "robot"), [(t, r) for t, r in SELECTED if r is PERFECT])
def test_every_perfect_robot_succeeds(lab_run, task_id, robot):
    score = lab_run[2][(task_id, robot.name)]
    assert score.success, (score.problems, score.harms, score.made_up)


@pytest.mark.parametrize(("task_id", "harm"), [
    ("food_03_missing_item_reply_lost", "double change"),
    ("food_07_ruko_while_asking", "after a stop"),
    ("food_08_haan_haan_while_asking", "no clear yes"),
    ("food_09_cash_cancel_yes_with_condition", "no clear yes"),
    ("food_10_wrong_name", "wrong change"),
])
def test_every_careless_robot_fails_with_the_harm_it_was_built_to_show(lab_run, task_id, harm):
    score = lab_run[2][(task_id, CARELESS.name)]
    assert not score.success and harm in {h["kind"] for h in score.harms}, score.harms


def test_mistakes_the_records_cant_show_are_caught_by_the_timeline(lab_run):
    """Tasks 8 and 9: the records end exactly right, but the robot acted before the real yes."""
    for task_id in ("food_08_haan_haan_while_asking", "food_09_cash_cancel_yes_with_condition"):
        score = lab_run[2][(task_id, CARELESS.name)]
        assert score.records_right and [h["kind"] for h in score.harms] == ["no clear yes"]


def test_ruko_after_the_change_was_sent_is_not_harm(lab_run):
    score = lab_run[2][("food_12_ruko_after_sending", PERFECT.name)]
    assert score.harms == [] and score.success


def test_the_puppets_reply_delay_is_measured_at_its_known_value(lab_run):
    """The puppet answers 0.8 s after the caller's last sound (C9): known before the call."""
    delays = lab_run[2][("food_01_cancel_prepaid", PERFECT.name)].timing["reply_delays_ms"]
    assert abs(delays[0] - 800) <= 60, delays


def test_a_robot_that_yields_is_measured_as_stopping_and_one_that_doesnt_is_not(lab_run):
    yielded = lab_run[2][("food_07_ruko_while_asking", PERFECT.name)].timing["stops"]
    ignored = lab_run[2][("food_07_ruko_while_asking", CARELESS.name)].timing["stops"]
    assert [s["stopped"] for s in yielded] == [True] and yielded[0]["after_ms"] < 1000
    assert [s["stopped"] for s in ignored] == [False]


def test_the_shielded_change_finishes_after_the_line_dies(lab_run):
    _run_dir, batch, scores = lab_run
    folder = folder_of(batch, "food_15_ruko_while_saving_then_drop", PERFECT)
    events = [json.loads(line) for line in (folder / "events.jsonl").read_text("utf-8").splitlines()]
    assert any(e["event"] == "director.dropped" for e in events)
    said = [e for e in events if e["event"] == "director.said"]
    assert said and said[0]["line"] == "ruko!"  # the ruko went out before the drop
    assert scores[("food_15_ruko_while_saving_then_drop", PERFECT.name)].records_right


def test_every_race_moment_was_hit(lab_run):
    _run_dir, batch, _scores = lab_run
    races = [o.race for o in batch.outcomes if o.label == FINE and o.race and o.race.get("armed_ms") is not None]
    assert races and all(r["hit"] is True for r in races), races


def test_the_clock_caller_and_leak_lights_are_green_on_real_calls(lab_run):
    _run_dir, batch, scores = lab_run
    folders = [Path(o.folder) for o in batch.outcomes if o.label == FINE]
    assert clock_light(folders, SETTINGS.sound_level).status == GREEN
    assert caller_light(folders, SETTINGS.sound_level).status == GREEN
    assert leaks_light(list(scores.values())).status == GREEN


def test_a_call_is_re_marked_the_same_from_its_folder_alone(lab_run, tmp_path):
    """L14: the folder is enough; nothing else from the run is needed."""
    _run_dir, batch, scores = lab_run
    original = folder_of(batch, "food_14_address_corrected_while_saving", PERFECT)
    copy = tmp_path / "alone"
    shutil.copytree(original, copy)
    again = asyncio.run(score_call(copy, TASKS["food_14_address_corrected_while_saving"], FoodPack(), SETTINGS))
    assert dataclasses.asdict(again) == dataclasses.asdict(read_score(original))


def test_the_results_table_from_real_calls(lab_run):
    _run_dir, _batch, scores = lab_run
    results = summarize(list(scores.values()), SETTINGS.reshuffle_rounds, SETTINGS.reshuffle_seed)
    assert results["puppet:perfect"].success.value == 1.0
    assert results["puppet:careless"].success.value == 0.0 and results["puppet:careless"].harms >= 5


# ---- breaking things on purpose (L4) ----------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CommandRobot:
    """A robot started from any command line, to break the lab on purpose."""

    name: str
    argv: tuple[str, ...]

    def command(self, db, folder, bells_url, settings, task):
        return list(self.argv)


async def one_call(tmp_path, robot, task_id="food_01_cancel_prepaid", settings=SETTINGS):
    plan = CallPlan("broken", TASKS[task_id], robot, "safe", 1)
    return await run_batch([plan], FoodPack(), make_run_dir(tmp_path, "l4"), settings)


async def test_a_robot_that_never_answers_counts_against_the_robot(tmp_path):
    never = CommandRobot("never", (sys.executable, "-c", "import sys; sys.exit(2)"))
    batch = await one_call(tmp_path, never)
    assert [(o.label, o.why) for o in batch.outcomes] == [(ROBOT_FAILURE, "the robot never said ready")]


async def test_a_robot_that_crashes_mid_call_counts_against_the_robot_and_keeps_its_folder(tmp_path):
    crash = CommandRobot("crashing", (sys.executable, "-m", "lab.dummy_robot.crashing"))
    batch = await one_call(tmp_path, crash)
    (outcome,) = batch.outcomes
    assert outcome.label == ROBOT_FAILURE and f"exited with code {CRASH_CODE}" in outcome.why
    assert (Path(outcome.folder) / "final.db").exists() and (Path(outcome.folder) / "robot.flac").exists()
    assert batch.scores and not batch.scores[0].success  # scored, and failed


async def test_a_robot_that_answers_at_a_dead_address_counts_against_the_robot(tmp_path):
    dead = CommandRobot("dead", (sys.executable, "-c", "print('ready ws://127.0.0.1:9/call', flush=True)"))
    batch = await one_call(tmp_path, dead)
    assert batch.outcomes[0].label == ROBOT_FAILURE and "didn't answer" in batch.outcomes[0].why


async def test_a_caller_that_keeps_crashing_is_a_lab_problem_rerun_then_the_batch_stops(tmp_path, monkeypatch):
    def broken_ear(self, piece, caller_ms):
        raise RuntimeError("the caller's own bug")

    monkeypatch.setattr(run_call_module.ScriptedCaller, "heard", broken_ear)
    batch = await one_call(tmp_path, PERFECT)
    assert [o.label for o in batch.outcomes] == [LAB_PROBLEM] * (SETTINGS.max_reruns + 1)
    assert batch.reruns == SETTINGS.max_reruns and "kept breaking" in batch.stopped and not batch.scores


async def test_the_lab_falling_behind_is_a_lab_problem_and_the_rerun_is_counted(tmp_path, monkeypatch):
    real_heard = run_call_module.ScriptedCaller.heard
    froze = []

    def stalls_once(self, piece, caller_ms):
        if caller_ms == 1000 and not froze:
            froze.append(caller_ms)
            time.sleep(0.25)  # our own laptop falling behind, inside the 20 ms beat
        real_heard(self, piece, caller_ms)

    monkeypatch.setattr(run_call_module.ScriptedCaller, "heard", stalls_once)
    batch = await one_call(tmp_path, PERFECT)
    assert [o.label for o in batch.outcomes] == [LAB_PROBLEM, FINE]
    assert "fell behind" in batch.outcomes[0].why and batch.reruns == 1
    assert len(batch.scores) == 1 and batch.scores[0].success  # only the clean rerun counts


def outcome(**fields) -> CallOutcome:
    return CallOutcome("c", "t", "r", "safe", 1, 1, FINE, "", **fields)


def test_a_missed_race_moment_is_a_lab_problem_but_a_moment_that_never_came_is_not():
    missed = outcome(race={"hit": False, "armed_ms": 1000, "why": "the save came first"})
    never = outcome(race={"hit": None, "armed_ms": None, "why": ""})
    _judge_break(missed, 0, SETTINGS)
    _judge_break(never, 0, SETTINGS)
    assert missed.label == LAB_PROBLEM and never.label == FINE  # never came: the robot never did the action


# ---- one clock across programs (L6) -----------------------------------------------------


def test_the_steady_clock_is_shared_by_separate_programs():
    """Another program's steady time falls between two of ours, taken before
    and after it ran: the same clock, not each program's own."""
    before = time.perf_counter_ns()
    theirs = int(subprocess.run([sys.executable, "-c", "import time; print(time.perf_counter_ns())"],
                                capture_output=True, text=True, check=True).stdout)
    after = time.perf_counter_ns()
    assert before < theirs < after
