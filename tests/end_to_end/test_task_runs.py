"""End to end through tasks: load each task file, set up its call, drive it
with a scripted agent, then let the task checker mark it.

The scripted agents stand in for the real AI agents of Part 2. A "careful"
agent does the right thing; a "careless" plain agent retries after a timeout.
"""

from pathlib import Path

import pytest

from engine.storage.database import SqliteDatabase
from engine.tasks.task_checker import check_task
from engine.tasks.task_file import Expectation, Forbidden, Task, load_task, load_tasks
from packs.food.pack import FoodPack

TASKS_DIR = Path(__file__).resolve().parents[2] / "packs" / "food" / "tasks"
TASKS = {task.id: task for task in load_tasks(TASKS_DIR)}


async def careful_safe_agent(kit, task_id):
    """What a correct agent does in safe mode for each sample task."""
    async def yes_then_confirm(prepare_tool, **inputs):
        prepared = await kit.tool(prepare_tool, **inputs)
        if prepared.status != "ok":
            return prepared
        kit.clock.advance(6)  # the agent reads the summary; the caller says a clear "haan"
        ticket = await kit.guard.issue_ticket(prepared.data["token"])
        reply = await kit.tool("confirm", token=prepared.data["token"], ticket=ticket)
        if reply.status == "error" and "Timed out" in reply.message:
            # Unknown result: retry the SAME token, never prepare a new one.
            reply = await kit.tool("confirm", token=prepared.data["token"], ticket=ticket)
        return reply

    if task_id == "food_01_cancel_prepaid":
        await kit.tool("find_order", order_no=4521, customer_name="Rahul Verma")
        return await yes_then_confirm("prepare_cancel_order", order_no=4521, customer_name="Rahul Verma")
    if task_id == "food_02_cancel_raced_by_dispatch":
        return await yes_then_confirm("prepare_cancel_order", order_no=4530, customer_name="Aman Gupta")
    if task_id == "food_03_missing_item_reply_lost":
        return await yes_then_confirm(
            "prepare_refund_order", order_no=4512, customer_name="Priya", amount_rupees=40, reason="missing_item"
        )
    if task_id == "food_04_change_address":
        return await yes_then_confirm(
            "prepare_change_address", order_no=4561, customer_name="Priya Sharma",
            new_address="Flat 2C, Rose Apartments, Model Town, Ludhiana",
        )
    if task_id == "food_05_too_late_hand_over":
        refused = await kit.tool("prepare_cancel_order", order_no=4540, customer_name="Vikram")
        assert refused.status == "refused"
        return await kit.tool("hand_over_to_human", reason="order already left; caller wants a person", order_no=4540)
    raise AssertionError(f"no scripted agent for {task_id}")


async def checked(kit, task):
    db = await SqliteDatabase.open(kit.path)
    try:
        return await check_task(db, task)
    finally:
        await db.close()


def test_task_files_load_and_ids_match_file_names():
    files = sorted(TASKS_DIR.glob("*.yaml"))
    assert len(files) == 5
    for path in files:
        assert load_task(path).id == path.stem


@pytest.mark.parametrize("task_id", sorted(TASKS))
async def test_careful_agent_passes_every_task_in_safe_mode(open_call, task_id):
    task = TASKS[task_id]
    kit = await open_call(FoodPack(), mode="safe", task=task)
    await careful_safe_agent(kit, task_id)
    verdict = await checked(kit, task)
    assert verdict.passed, verdict.problems


async def test_careless_plain_agent_fails_the_lost_reply_task(open_call):
    task = TASKS["food_03_missing_item_reply_lost"]
    kit = await open_call(FoodPack(), mode="plain", task=task)
    refund = {"order_no": 4512, "customer_name": "Priya", "amount_rupees": 40, "reason": "missing_item"}
    if (await kit.tool("refund_order", **refund)).status == "error":
        await kit.tool("refund_order", **refund)  # retry after the timeout
    verdict = await checked(kit, task)
    assert not verdict.passed
    assert verdict.problems == ["refunds {'order_no': 4512}: expected 1 row, found 2"]


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
        await careful_safe_agent(kit, task.id)
        tables = [row["name"] for row in await kit.rows(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name != 'call_info' ORDER BY name"
        )]
        dumps.append({table: await kit.rows(f"SELECT * FROM {table}") for table in tables})
    assert dumps[0] == dumps[1]
