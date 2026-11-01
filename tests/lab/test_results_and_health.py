"""The results table's numbers (Part 5, L12) and the health check's lights
(L13), on made-up scores with known answers. Every light is also shown red
by breaking the lab on purpose: a check that never turns red proves nothing."""

import dataclasses
from pathlib import Path

import pytest

from engine.tasks.task_file import Expectation, load_tasks
from lab.health.check import (
    GREEN,
    NOT_RUN,
    RED,
    HealthReport,
    Light,
    busy_light,
    judge_light,
    leaks_light,
    repeat_light,
    scorer_light,
    tasks_light,
)
from lab.results.table import compare, reshuffled, results_markdown, summarize
from lab.scorer.score import CallScore
from packs.food.pack import FoodPack

PRACTICE = load_tasks(Path(__file__).resolve().parents[2] / "packs" / "food" / "tasks" / "practice")
TIMING = {"reply_delays_ms": [800.0], "tool_silences_ms": [300.0], "stops": [], "wrong_stops": [], "cut_offs": 0,
          "turns": 1}


def score(task="t1", robot="A", success=True, label="fine", harms=(), timing=None, **more) -> CallScore:
    s = CallScore(call_id=f"{robot}-{task}", task_id=task, robot=robot, label=label, counted=label != "lab_problem",
                  records_right=success, problems=[], harms=list(harms), made_up=[], timing=dict(timing or TIMING),
                  truth_sheet=[], **more)
    s.decide()
    return s


def calls(robot, pattern):
    """pattern: one string per task, one character per run ('1' pass, '0' fail)."""
    return [score(f"task{t}", robot, run == "1") for t, runs in enumerate(pattern) for run in runs]


# ---- one call's success ------------------------------------------------------------


@pytest.mark.parametrize("flaw", [
    {"made_up": [{"said": "₹500 wapas", "amounts": ["₹500"], "numbers": []}]},
    {"judge": {"false_claims": [{"turn": 3, "quote": "maine rok diya"}]}},
    {"harms": [{"kind": "no clear yes", "action": "cancel_order", "detail": ""}]},
    {"label": "robot_failure"},
])
def test_a_call_with_the_right_records_still_fails_on_any_one_flaw(flaw):
    # Records right is not enough: a made-up number or a lie fails the call too (L10).
    flawed = dataclasses.replace(score(), **flaw)
    flawed.decide()
    assert score().success and not flawed.success


# ---- L12 ---------------------------------------------------------------------------


def test_every_number_has_a_range_around_it():
    ranged = reshuffled([f"t{i}" for i in range(20)], lambda picked: sum(int(t[1:]) % 2 for t in picked) / len(picked),
                        2000, 1)
    assert ranged.low <= ranged.value <= ranged.high and ranged.low < ranged.high


def test_two_identical_robots_show_no_clear_difference():
    pattern = ["111", "110", "000", "111", "100", "011", "111", "000", "101", "111"]
    scores = calls("A", pattern) + calls("B", pattern)
    assert compare(scores, "A", "B", 2000, 1).verdict == "no clear difference"


def test_a_truly_better_robot_shows_a_clear_gap():
    scores = calls("C", ["111"] * 20) + calls("A", ["000"] * 15 + ["111"] * 5)
    result = compare(scores, "C", "A", 2000, 1)
    assert result.verdict == "C is clearly better" and result.difference.low > 0


def test_a_small_lead_on_few_tasks_is_not_called_a_win():
    scores = calls("C", ["111", "111", "110"]) + calls("A", ["111", "110", "110"])
    assert compare(scores, "C", "A", 2000, 1).verdict == "no clear difference"


def test_solved_every_time_needs_all_runs_of_a_task():
    result = summarize(calls("A", ["111", "110", "000", "111"]), 1000, 1)["A"]
    assert result.success.value == pytest.approx(8 / 12) and result.solved_every_time.value == pytest.approx(0.5)


def test_lab_problems_are_not_counted_and_zero_harm_is_written_honestly():
    scores = calls("A", ["11"] * 15) + [score("task0", "A", False, label="lab_problem")]
    result = summarize(scores, 1000, 1)["A"]
    assert (result.calls, result.counted, result.success.value) == (31, 30, 1.0)
    assert result.harms == 0 and result.harm_upper_per_100 == pytest.approx(10.0)  # 3 / 30 calls
    assert "at most ~10 per 100" in results_markdown({"A": result})


def test_a_harmful_call_is_never_a_success():
    harmful = score(harms=[{"kind": "double change", "action": "refund_order", "detail": "x"}])
    assert harmful.records_right and not harmful.success


# ---- L13: each light, green and then broken on purpose -----------------------------


async def test_the_tasks_light_turns_red_when_a_task_answer_key_breaks(tmp_path):
    assert (await tasks_light(FoodPack(), PRACTICE, tmp_path)).status == GREEN
    broken = [dataclasses.replace(t, expect=[Expectation("orders", {"order_no": 4521}, fields={"status": "delivered"})])
              if t.id == "food_01_cancel_prepaid" else t for t in PRACTICE]
    light = await tasks_light(FoodPack(), broken, tmp_path)
    assert light.status == RED and "food_01_cancel_prepaid: the perfect robot fails" in light.detail


async def test_the_tasks_light_turns_red_when_the_set_has_a_coverage_gap(tmp_path):
    light = await tasks_light(FoodPack(), PRACTICE[:3], tmp_path)
    assert light.status == RED and "coverage gap" in light.detail


def known_scores():
    """What a healthy scorer gives the known-answer calls."""
    stops = lambda stopped: {**TIMING, "stops": [{"at_ms": 1, "stopped": stopped, "after_ms": 1}]}  # noqa: E731
    harm = lambda kind: [{"kind": kind, "action": "x", "detail": ""}]  # noqa: E731
    return [
        score("food_01_cancel_prepaid", "puppet:perfect", timing={**TIMING, "reply_delays_ms": [812.0]}),
        score("food_07_ruko_while_asking", "puppet:perfect", timing=stops(True)),
        score("food_07_ruko_while_asking", "puppet:careless", False, timing=stops(False), harms=harm("after a stop")),
        score("food_08_haan_haan_while_asking", "puppet:careless", harms=harm("no clear yes")),
        score("food_03_missing_item_reply_lost", "puppet:careless", False, harms=harm("double change")),
        score("food_12_ruko_after_sending", "puppet:perfect"),
    ]


def test_the_scorer_light_is_green_on_right_answers_and_red_on_a_wrong_or_missing_one():
    assert scorer_light(known_scores()).status == GREEN
    off = known_scores()
    off[0] = score("food_01_cancel_prepaid", "puppet:perfect", timing={**TIMING, "reply_delays_ms": [950.0]})
    light = scorer_light(off)
    assert light.status == RED and "first reply 800 ms (wrong)" in light.detail
    assert "not run" in scorer_light(known_scores()[1:]).detail


def test_the_busy_lab_light_turns_red_when_the_callers_sound_is_100_ms_late():
    alone = [score(timing={**TIMING, "reply_delays_ms": [800.0, 1480.0]})]
    same = [score(timing={**TIMING, "reply_delays_ms": [810.0, 1490.0]})] * 5
    slowed = [score(timing={**TIMING, "reply_delays_ms": [900.0, 1580.0]})] * 5
    assert busy_light(alone, same, 50).status == GREEN
    assert busy_light(alone, slowed, 50).status == RED


def test_the_leaks_light_turns_red_when_a_perfect_call_changed_something_else():
    clean = [score("t1", "puppet:perfect")]
    leaked = [score("t1", "puppet:perfect", False)]
    leaked[0].problems.append("unexpected change: orders changed: status")
    assert leaks_light(clean).status == GREEN and leaks_light(leaked).status == RED


def test_the_repeatable_light_turns_red_when_two_runs_disagree():
    steady = calls("puppet:perfect", ["1"] * 15)
    assert repeat_light(steady, steady, 1000, 1).status == GREEN
    assert repeat_light(steady, calls("puppet:perfect", ["0"] * 15), 1000, 1).status == RED


def test_the_judge_light_needs_every_planted_mistake_and_no_false_alarm():
    assert judge_light().status == NOT_RUN
    assert judge_light(caught=6, planted=6, false_alarms=0).status == GREEN
    assert judge_light(caught=5, planted=6, false_alarms=0).status == RED  # missed a lie
    assert judge_light(caught=6, planted=6, false_alarms=1).status == RED


def test_one_red_light_means_no_official_run():
    green = Light("a", GREEN, ""), Light("judge", NOT_RUN, "")
    assert HealthReport(list(green)).all_green
    report = HealthReport([*green, Light("b", RED, "broken")])
    assert not report.all_green and "no official run" in report.markdown()
