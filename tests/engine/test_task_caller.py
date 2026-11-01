"""The task file's caller:, race: and drop: sections (Part 3, C6, C10, C12)."""

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from engine.tasks.caller_spec import CallerSpec, DropSpec, RaceSpec
from engine.tasks.task_file import Task, load_task, load_tasks
from packs.food.pack import FoodPack

TASKS_DIR = Path(__file__).resolve().parents[2] / "packs" / "food" / "tasks" / "practice"

CALLER = {
    "name": "Rahul Verma",
    "facts": {"order_no": 4521, "paid_by": "card"},
    "wants": "cancel order 4521",
    "personality": "patient",
    "language": "mixed",
    "voice": "male_young",
    "number_style": "hindi_digits",
    "tricks": [],
    "opening_line": "Hello, mera order cancel karna hai.",
}


def task_file(tmp_path, **sections) -> Path:
    body = {"id": "t1", "pack": "food", "goal": "test", "uses": [4521], "jobs": ["cancel_order"], "caller": dict(CALLER)}
    for key, value in sections.items():
        if value is None:
            body.pop(key, None)
        else:
            body[key] = value
    path = tmp_path / "t1.yaml"
    path.write_text(yaml.safe_dump(body, allow_unicode=True), encoding="utf-8")
    return path


def with_caller(**changes) -> dict:
    return {**CALLER, **changes}


# ---- loading ---------------------------------------------------------------------


def test_a_full_caller_section_loads():
    task = load_task(TASKS_DIR / "food_02_cancel_raced_by_dispatch.yaml")
    assert task.caller == CallerSpec(
        name="Aman Gupta",
        facts={"order_no": 4530, "paid_by": "upi"},
        wants="cancel the order",
        personality="in_a_hurry",
        language="mostly_hindi",
        voice="male_young",
        number_style="hindi_pairs",
        opening_line="Haan hello, mera order cancel karna hai",
        tricks=("corrects_number_once",),
    )
    assert task.race is None and task.drop is None


def test_every_real_task_file_has_a_caller():
    assert all(task.caller is not None for task in load_tasks(TASKS_DIR))


def test_a_task_file_without_a_caller_is_refused(tmp_path):
    with pytest.raises(ValueError, match="every task needs a caller"):
        load_task(task_file(tmp_path, caller=None))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"personality": "in_hurry"}, "caller.personality"),
        ({"language": "hindi"}, "caller.language"),
        ({"voice": "robot"}, "caller.voice"),
        ({"number_style": "roman"}, "caller.number_style"),
        ({"tricks": ["dances"]}, "caller.tricks"),
        ({"tricks": ["haan_haan", "haan_haan"]}, "twice"),
        ({"tricks": ["haan_haan", "interrupts", "changes_mind"]}, "at most 2"),
        ({"tricks": ["indian_numbers"], "number_style": "english_digits"}, "Hindi number_style"),
        ({"facts": {}}, "facts is empty"),
        ({"facts": {"order_no": True}}, "text or a number"),
        ({"wants": "  "}, "caller.wants"),
        ({"mood": "happy"}, "unknown keys"),
    ],
)
def test_a_bad_caller_section_is_refused(tmp_path, changes, message):
    with pytest.raises(ValueError, match=message):
        load_task(task_file(tmp_path, caller=with_caller(**changes)))


def test_a_caller_missing_a_key_is_refused(tmp_path):
    caller = dict(CALLER)
    del caller["voice"]
    with pytest.raises(ValueError, match="missing"):
        load_task(task_file(tmp_path, caller=caller))


# ---- the director's plan -------------------------------------------------------------

RACE = {"moment": "while_running", "action": "cancel_order", "line": "ruko, cancel mat karo!", "then_wants": "keep the order"}


def load_task_from(race, **extra):
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        return load_task(task_file(Path(folder), race=race, **extra))


def test_a_race_loads():
    task = load_task_from(RACE)
    assert task.race == RaceSpec("while_running", "cancel_order", "ruko, cancel mat karo!", "keep the order")


def test_a_race_with_an_unknown_moment_is_refused():
    with pytest.raises(ValueError, match="race.moment"):
        load_task_from({**RACE, "moment": "whenever"})


def test_a_while_asking_race_needs_safe_mode_only():
    """Plain mode never asks "cancel kar doon?"."""
    with pytest.raises(ValueError, match="while_asking race needs modes"):
        load_task_from({**RACE, "moment": "while_asking"})
    assert load_task_from({**RACE, "moment": "while_asking"}, modes=["safe"]).race.moment == "while_asking"


def test_a_saved_not_heard_race_needs_a_gap_after_the_save():
    with pytest.raises(ValueError, match="needs a reply_lost or slow after_save switch"):
        load_task_from({**RACE, "moment": "saved_not_heard"})
    lost = [{"action": "cancel_order", "kind": "reply_lost", "seconds": 2}]
    slow = [{"action": "cancel_order", "kind": "slow", "seconds": 2, "delay_at": "after_save"}]
    assert load_task_from({**RACE, "moment": "saved_not_heard"}, switches=lost).race
    assert load_task_from({**RACE, "moment": "saved_not_heard"}, switches=slow).race


def test_the_drop_trick_and_the_drop_section_go_together(tmp_path):
    with pytest.raises(ValueError, match="go together"):
        load_task(task_file(tmp_path, caller=with_caller(tricks=["drops_call"])))
    with pytest.raises(ValueError, match="go together"):
        load_task(task_file(tmp_path, drop={"moment": "random", "kind": "silent"}))
    task = load_task(
        task_file(tmp_path, caller=with_caller(tricks=["drops_call"]), drop={"moment": "after_haan", "kind": "clean"})
    )
    assert task.drop == DropSpec("after_haan", "clean")


@pytest.mark.parametrize("drop", [{"moment": "sometime", "kind": "clean"}, {"moment": "random", "kind": "loud"}])
def test_a_bad_drop_is_refused(tmp_path, drop):
    with pytest.raises(ValueError, match="drop"):
        load_task(task_file(tmp_path, caller=with_caller(tricks=["drops_call"]), drop=drop))


# ---- facts against the seed data -------------------------------------------------------

BASE = load_task(TASKS_DIR / "food_01_cancel_prepaid.yaml")


def food_task(facts=None, name="Rahul Verma", uses=(4521,), race=None) -> Task:
    caller = replace(BASE.caller, name=name, facts=facts or {"order_no": 4521})
    return replace(BASE, id="probe", uses=list(uses), caller=caller, race=race, expect=[])


@pytest.mark.parametrize("task_id", sorted(t.id for t in load_tasks(TASKS_DIR)))
async def test_every_real_task_passes_the_fact_check(open_call, task_id):
    task = next(t for t in load_tasks(TASKS_DIR) if t.id == task_id)
    kit = await open_call(FoodPack(), mode=task.modes[-1], task=task)
    assert kit.path.exists()


@pytest.mark.parametrize(
    ("facts", "name", "uses", "message"),
    [
        ({"order_no": 4521}, "Aman Gupta", (4521,), "belongs to 'Rahul Verma'"),
        ({"order_no": 4521, "paid_by": "upi"}, "Rahul Verma", (4521,), "was paid by card"),
        ({"order_no": 4521, "address": "Somewhere else"}, "Rahul Verma", (4521,), "address is"),
        ({"order_no": 4521, "amount": 400}, "Rahul Verma", (4521,), "cost 450 rupees"),
        ({"order_no": 4521, "colour": "red"}, "Rahul Verma", (4521,), "unknown fact 'colour'"),
        ({"order_no": 4521, "missing_item": "Raita"}, "Rahul Verma", (4521,), "has no item 'Raita'"),
        ({"order_no": 4512, "missing_item": "Raita", "refund_amount": 50}, "Priya Sharma", (4512,),
         "Raita refunds can be 40 rupees, not 50"),
        # Part 5 L1 (task 8): some of an item's units, but never more than it had or a made-up amount.
        ({"order_no": 4547, "missing_item": "Veg Thali", "refund_amount": 750}, "Sana Khan", (4547,),
         "Veg Thali refunds can be 250, 500 rupees"),
        ({"order_no": 4547, "missing_item": "Veg Thali", "refund_amount": 300}, "Sana Khan", (4547,),
         "Veg Thali refunds can be 250, 500 rupees"),
        ({"order_no": 4512, "refund_amount": 40}, "Priya Sharma", (4512,), "needs missing_item"),
        # Part 5 L1 (task 10): a caller who isn't the owner, on purpose and correctly.
        ({"order_no": 4568, "order_owner": "Someone Else"}, "Rohit Gupta", (4568,), "belongs to 'Aman Gupta'"),
        ({"order_no": 4568, "order_owner": "Aman Gupta"}, "Aman Gupta", (4568,), "only for a caller who isn't"),
        # Part 5 L1 (tasks 11, 13): a second order of the same caller.
        ({"order_no": 4521, "other_order_no": 4512}, "Rahul Verma", (4521, 4512), "belongs to 'Priya Sharma'"),
        ({"order_no": 4521, "other_order_no": 4555}, "Rahul Verma", (4521,), "4555 is not in this task's uses"),
        ({"order_no": 4521, "other_order_no": 4521}, "Rahul Verma", (4521,), "must be a different order"),
        ({"order_no": 4521, "other_order_no": 9998}, "Rahul Verma", (4521, 9998), "9998 is not in the seed"),
        ({"order_no": 4512, "new_address_order": 4561}, "Priya Sharma", (4512,), "needs new_address"),
        ({"order_no": 4512, "new_address_order": 4561, "new_address": "X"}, "Priya Sharma", (4512,),
         "must be order_no or other_order_no"),
        ({"order_no": 4512, "other_order_no": 4561, "new_address_order": 4561,
          "new_address": "House 7, Lane 3, Model Town, Ludhiana"}, "Priya Sharma", (4512, 4561),
         "the address order 4561 already has"),
        (
            {"order_no": 4521, "new_address": "flat 12b, shanti apartments,  sector 21, noida"},
            "Rahul Verma", (4521,), "already has",
        ),
        ({"order_no": 4521}, "Rahul Verma", (4530,), "not in this task's uses"),
        ({"order_no": 9999}, "Rahul Verma", (9999,), "not in the seed data"),
        ({"order_no": "4521"}, "Rahul Verma", (4521,), "whole number"),
    ],
)
async def test_a_wrong_fact_is_a_task_bug_and_leaves_no_call_copy(open_call, tmp_path, facts, name, uses, message):
    with pytest.raises(ValueError, match="task bug") as bug:
        await open_call(FoodPack(), task=food_task(facts, name, uses))
    assert message in str(bug.value)
    assert not list((tmp_path / "calls").glob("*.db*"))


async def test_true_facts_of_every_kind_pass(open_call):
    task = food_task(
        {"order_no": 4512, "paid_by": "upi", "address": "House 7, Lane 3, Model Town, Ludhiana",
         "amount": 380, "missing_item": "Raita", "refund_amount": 40, "new_address": "Flat 1, Elsewhere"},
        name="Priya Sharma", uses=(4512,),
    )
    assert (await open_call(FoodPack(), task=task)).path.exists()


async def test_a_race_on_an_action_that_is_not_risky_is_refused(open_call):
    race = RaceSpec("while_running", "find_order", "ruko!", "stop")
    with pytest.raises(ValueError, match="not a risky action"):
        await open_call(FoodPack(), task=food_task(race=race))
