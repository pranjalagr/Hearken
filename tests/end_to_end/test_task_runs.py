"""End to end through tasks: load each task file, set up its call, drive it
with a puppet's tool calls, then let the task checker mark it (Part 5, L1, L2, L7).

The puppets stand in for the real AI agents of Part 2. For every practice
task, a perfect robot must pass and a careless robot, making the exact
mistake the task is built to catch, must fail; otherwise the task is toothless.
"""

from pathlib import Path

import pytest

from engine.storage.database import SqliteDatabase
from engine.tasks.coverage import count_coverage
from engine.tasks.records import business_tables, take_snapshot
from engine.tasks.task_checker import check_task
from engine.tasks.task_file import Expectation, Forbidden, Task, load_task, load_tasks
from lab.dummy_robot.robot import UseTool
from lab.puppets.answer_key import check_answer_key, play_tools
from lab.puppets.food import PUPPETS, Careless, NotByScript, script_for
from packs.food.pack import FoodPack

TASKS_DIR = Path(__file__).resolve().parents[2] / "packs" / "food" / "tasks" / "practice"
TASKS = {task.id: task for task in load_tasks(TASKS_DIR)}
CARELESS = {task_id: p.careless for task_id, p in PUPPETS.items() if isinstance(p.careless, Careless)}


async def checked(kit, task, before=None):
    db = await SqliteDatabase.open(kit.path)
    try:
        return await check_task(db, task, before)
    finally:
        await db.close()


async def play(kit, steps):
    return await play_tools(steps, kit.backend.call_tool, kit.guard.issue_ticket, kit.clock.advance)


def test_15_practice_tasks_and_ids_match_file_names():
    files = sorted(TASKS_DIR.glob("*.yaml"))
    assert len(files) == 15
    for path in files:
        assert load_task(path).id == path.stem


def test_every_task_has_puppets_and_only_three_cannot_have_a_careless_script():
    """A new task must get a perfect and a careless robot, or say why it can't."""
    assert set(PUPPETS) == set(TASKS)
    not_by_script = {task_id for task_id, p in PUPPETS.items() if isinstance(p.careless, NotByScript)}
    assert not_by_script == {
        "food_02_cancel_raced_by_dispatch",  # the shop refuses: honesty is what's tested
        "food_12_ruko_after_sending",  # already sent: honesty is what's tested
        "food_15_ruko_while_saving_then_drop",  # the call box's shield, tested by its own tests
    }


@pytest.mark.parametrize("task_id", sorted(TASKS))
async def test_the_perfect_robot_passes_every_task(tmp_path, task_id):
    steps, mode = script_for(task_id, "perfect")
    verdict = await check_answer_key(FoodPack(), TASKS[task_id], steps, mode, tmp_path)
    assert verdict.passed, verdict.problems


@pytest.mark.parametrize("task_id", sorted(t for t, c in CARELESS.items() if c.caught_by == "records"))
async def test_the_careless_robot_fails_every_task_the_records_can_show(tmp_path, task_id):
    steps, mode = script_for(task_id, "careless")
    verdict = await check_answer_key(FoodPack(), TASKS[task_id], steps, mode, tmp_path)
    assert not verdict.passed, f"toothless: {CARELESS[task_id].mistake}"


@pytest.mark.parametrize("task_id", sorted(t for t, c in CARELESS.items() if c.caught_by == "harm"))
async def test_some_mistakes_leave_the_same_records_and_need_the_calls_timeline(tmp_path, task_id):
    """Acting on "haan haan" or "haan, lekin..." ends with the same records as
    the right yes: only the call's timeline shows it (L8, tested in the lab runs)."""
    steps, mode = script_for(task_id, "careless")
    verdict = await check_answer_key(FoodPack(), TASKS[task_id], steps, mode, tmp_path)
    assert verdict.passed and verdict.changes


async def test_the_practice_set_covers_every_job_moment_trick_and_ending(tmp_path):
    """L1: a robot that fails at one of these can't pass the set unnoticed."""
    do_nothing = set()
    risky = {action.name for action in FoodPack().actions() if action.risky}
    for task_id, task in TASKS.items():
        steps, mode = script_for(task_id, "perfect")
        verdict = await check_answer_key(FoodPack(), task, steps, mode, tmp_path)
        # The robot changed nothing about the order or the money: no risky
        # action happened. Lookups and hand-overs aren't such changes, and the
        # task's own timeline (task 2's rider) isn't the robot's doing.
        if not risky & set(verdict.acted):
            do_nothing.add(task_id)
    coverage = count_coverage(list(TASKS.values()), do_nothing)
    actions = [action.name for action in FoodPack().actions()]
    assert coverage.gaps(actions) == [], coverage.grid()
    assert sorted(do_nothing) == [
        "food_02_cancel_raced_by_dispatch", "food_05_too_late_hand_over", "food_06_status_out_of_scope",
        "food_07_ruko_while_asking", "food_10_wrong_name",
    ]


def test_a_gap_in_coverage_is_named():
    coverage = count_coverage([TASKS["food_01_cancel_prepaid"]], set())
    gaps = coverage.gaps([action.name for action in FoodPack().actions()])
    assert "job cancel_order: 1 < 3" in gaps and "race moment while_asking: 0 < 2" in gaps
    assert "do-nothing endings: 0 < 3" in gaps and "English callers: 0 < 2" in gaps


# ---- the task checker (L7) --------------------------------------------------------


async def test_checker_fails_a_call_where_nothing_happened(open_call):
    task = TASKS["food_01_cancel_prepaid"]
    kit = await open_call(FoodPack(), task=task)
    verdict = await checked(kit, task)
    assert verdict.problems == [
        "orders {'order_no': 4521}: status is 'preparing', expected 'cancelled'",
        "refunds {'order_no': 4521}: expected 1 row, found 0",
    ]


async def test_checker_catches_a_forbidden_action(open_call):
    task = Task(
        id="no_cancel", pack="food", goal="x", uses=[4521], modes=("plain",),
        expect=[Expectation("orders", {"order_no": 4521}, count=1)],
        forbid=[Forbidden("cancel_order")],
    )
    kit = await open_call(FoodPack(), mode="plain")
    await kit.tool("cancel_order", order_no=4521, customer_name="Rahul")
    verdict = await checked(kit, task)
    assert verdict.problems == ["forbidden: cancel_order happened 1 time(s)"]


async def snapshot(kit):
    db = await SqliteDatabase.open(kit.path)
    try:
        return await take_snapshot(db, business_tables(FoodPack()))
    finally:
        await db.close()


async def test_a_sloppy_robot_that_also_changes_another_order_fails(open_call):
    """L7 check 3: the right job plus one extra change is not a pass."""
    task = TASKS["food_01_cancel_prepaid"]
    kit = await open_call(FoodPack(), task=task)
    before = await snapshot(kit)
    steps, _ = script_for(task.id, "perfect")
    extra = [UseTool("prepare_change_address", {"order_no": 4561, "customer_name": "Priya Sharma",
                                                "new_address": "Somewhere new"}),
             UseTool("confirm", approve=True)]
    await play(kit, steps + extra)
    verdict = await checked(kit, task, before)
    assert verdict.problems == [
        "unexpected change: orders changed: address 'House 7, Lane 3, Model Town, Ludhiana' -> 'Somewhere new'"
    ]


async def test_an_extra_change_to_a_row_the_task_names_still_fails(open_call):
    """The task names order 4521's status; changing its address too is not covered by that."""
    task = TASKS["food_01_cancel_prepaid"]
    kit = await open_call(FoodPack(), task=task)
    before = await snapshot(kit)
    extra = [UseTool("prepare_change_address", {"order_no": 4521, "customer_name": "Rahul Verma",
                                                "new_address": "Somewhere new"}),
             UseTool("confirm", approve=True)]
    steps, _ = script_for(task.id, "perfect")
    await play(kit, extra + steps)
    verdict = await checked(kit, task, before)
    assert len(verdict.problems) == 1 and "address 'Flat 12B" in verdict.problems[0]


async def test_a_new_row_nobody_expected_fails(open_call):
    task = TASKS["food_01_cancel_prepaid"]
    kit = await open_call(FoodPack(), task=task)
    before = await snapshot(kit)
    steps, _ = script_for(task.id, "perfect")
    await play(kit, steps + [UseTool("hand_over_to_human", {"reason": "not needed", "order_no": 4521})])
    verdict = await checked(kit, task, before)
    assert verdict.problems == [f"unexpected change: handoffs added: {verdict.changes[-1].after}"]


async def test_a_task_refuses_a_mode_it_is_not_meant_for(open_call):
    with pytest.raises(ValueError, match="not meant for plain mode"):
        await open_call(FoodPack(), mode="plain", task=TASKS["food_02_cancel_raced_by_dispatch"])


async def test_same_task_twice_gives_exactly_the_same_database(open_call):
    """Same task = same result, every run: the pretend clock and fixed seed make
    runs repeatable, so a difference between agents is never just luck."""
    task = TASKS["food_03_missing_item_reply_lost"]
    dumps = []
    for _ in range(2):
        kit = await open_call(FoodPack(), task=task)
        steps, _ = script_for(task.id, "perfect")
        await play(kit, steps)
        tables = [row["name"] for row in await kit.rows(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name != 'call_info' ORDER BY name"
        )]
        dumps.append({table: await kit.rows(f"SELECT * FROM {table}") for table in tables})
    assert dumps[0] == dumps[1]
