"""Storage: starting files, per-call copies, call settings and STRICT tables."""

import sqlite3

import pytest

from engine.storage.call_factory import CallFactory
from engine.storage.call_settings import CallSettings, read_call_settings
from engine.storage.database import sqlite_session
from packs.food.pack import FoodPack
from packs.toy.pack import ToyPack


async def test_new_call_has_engine_tables_pack_tables_seed_and_settings(tmp_path):
    factory = CallFactory(FoodPack(), tmp_path)
    path = await factory.new_call(CallSettings("c1", "food", "safe", True, task_id="t1"))
    async with sqlite_session(path) as db:
        settings = await read_call_settings(db)
        async with db.transaction(write=False) as tx:
            tables = {row["name"] for row in await tx.fetch_all("SELECT name FROM sqlite_master WHERE type = 'table'")}
            order = await tx.fetch_one("SELECT amount_paise, status FROM orders WHERE order_no = 4521")
    assert {"call_info", "prepared_actions", "tickets", "breakdown_switches", "timeline", "action_log"} <= tables
    assert {"customers", "orders", "order_items", "refunds", "handoffs"} <= tables
    assert order == {"amount_paise": 45000, "status": "preparing"}  # worked out from the items
    assert (settings.call_id, settings.mode, settings.ticket_check, settings.task_id) == ("c1", "safe", True, "t1")


async def test_each_call_gets_its_own_copy(tmp_path):
    factory = CallFactory(ToyPack(), tmp_path)
    first = await factory.new_call(CallSettings("c1", "toy"))
    second = await factory.new_call(CallSettings("c2", "toy"))
    async with sqlite_session(first) as db:
        async with db.transaction() as tx:
            await tx.execute("UPDATE boxes SET balance_paise = 1 WHERE name = 'a'")
    async with sqlite_session(second) as db:
        async with db.transaction(write=False) as tx:
            row = await tx.fetch_one("SELECT balance_paise FROM boxes WHERE name = 'a'")
    assert row["balance_paise"] == 10_000  # call 1's change never reaches call 2


async def test_a_call_copy_is_never_reused(tmp_path):
    factory = CallFactory(ToyPack(), tmp_path)
    await factory.new_call(CallSettings("same", "toy"))
    with pytest.raises(FileExistsError):
        await factory.new_call(CallSettings("same", "toy"))


def test_plain_mode_with_ticket_check_is_not_allowed():
    with pytest.raises(ValueError):
        CallSettings("c1", "toy", mode="plain", ticket_check=True)
    CallSettings("c1", "toy", mode="plain", ticket_check=False)  # fine


async def test_database_rejects_bad_values(tmp_path):
    path = await CallFactory(FoodPack(), tmp_path).new_call(CallSettings("c1", "food"))
    bad_rows = [
        "UPDATE orders SET status = 'shipped' WHERE order_no = 4521",  # not in the fixed list
        "UPDATE orders SET amount_paise = 450.5 WHERE order_no = 4521",  # money must be whole paise
        "INSERT INTO orders VALUES (45210, 1, 100, 'upi', 'x', 'placed')",  # 5-digit order number
        "INSERT INTO refunds (order_no, amount_paise, method, reason, at) VALUES (4521, 100, 'cash', 'x', 0)",
    ]
    async with sqlite_session(path) as db:
        for sql in bad_rows:
            with pytest.raises(sqlite3.IntegrityError):
                async with db.transaction() as tx:
                    await tx.execute(sql)


async def test_a_failed_transaction_leaves_nothing_behind(tmp_path):
    path = await CallFactory(ToyPack(), tmp_path).new_call(CallSettings("c1", "toy"))
    async with sqlite_session(path) as db:
        with pytest.raises(RuntimeError):
            async with db.transaction() as tx:
                await tx.execute("UPDATE boxes SET balance_paise = 0 WHERE name = 'a'")
                raise RuntimeError("crash in the middle")
        async with db.transaction(write=False) as tx:
            row = await tx.fetch_one("SELECT balance_paise FROM boxes WHERE name = 'a'")
    assert row["balance_paise"] == 10_000
