"""End to end over real MCP: start the tool server as its own process (as an
agent will), talk to it with the official MCP client, and check the database.

Ticket approval goes through the Guard's private door from this test process,
on its own connection, exactly as the Guard will do in Part 2: the door is
never offered as a tool.
"""

import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from cli.make_call import main as make_call
from engine.lab_controls.clock import WallClock
from engine.safety.guard_door import GuardDoor
from engine.storage.call_settings import read_call_settings
from engine.storage.database import sqlite_session

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK = PROJECT_ROOT / "packs" / "food" / "tasks" / "food_01_cancel_prepaid.yaml"
RAHUL = {"order_no": 4521, "customer_name": "Rahul Verma"}


def server_for(db_path):
    # One server process per call; the --db argument is the only way to pick the copy.
    return StdioServerParameters(
        command=sys.executable, args=["-m", "cli.serve_call", "--db", str(db_path)], cwd=str(PROJECT_ROOT)
    )


async def call(session, tool, **arguments):
    result = await session.call_tool(tool, arguments)
    return json.loads(result.content[0].text), result.is_error


async def test_safe_mode_over_mcp_cancels_once(tmp_path):
    db_path = await make_call(str(TASK), "safe", True, str(tmp_path))
    async with stdio_client(server_for(db_path)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = [tool.name for tool in (await session.list_tools()).tools]
            assert tools == [
                "find_order", "prepare_change_address", "prepare_cancel_order",
                "prepare_refund_order", "hand_over_to_human", "confirm",
            ]

            found, _ = await call(session, "find_order", **RAHUL)
            assert found["status"] == "ok"
            assert (found["data"]["status"], found["data"]["amount"]) == ("preparing", "₹450")

            prepared, _ = await call(session, "prepare_cancel_order", **RAHUL)
            token = prepared["data"]["token"]
            assert prepared["data"]["summary"] == "cancel order 4521 and refund ₹450 to card"

            # A made-up ticket is refused, not an error: the service worked and said no.
            fake, is_error = await call(session, "confirm", token=token, ticket="K-77")
            assert (fake["status"], is_error) == ("refused", False)

            # The Guard heard a clear "haan": it writes the ticket through its private door.
            async with sqlite_session(db_path) as guard_db:
                settings = await read_call_settings(guard_db)
                guard = GuardDoor(guard_db, WallClock(settings.started_at_wall))
                ticket = await guard.issue_ticket(token)

            done, is_error = await call(session, "confirm", token=token, ticket=ticket)
            assert (done["status"], is_error) == ("ok", False)
            again, _ = await call(session, "confirm", token=token, ticket=ticket)
            assert again["status"] == "already_done"

    async with sqlite_session(db_path) as db:
        async with db.transaction(write=False) as tx:
            order = await tx.fetch_one("SELECT status FROM orders WHERE order_no = 4521")
            refunds = await tx.fetch_all("SELECT amount_paise, method FROM refunds WHERE order_no = 4521")
            log = await tx.fetch_all("SELECT tool, outcome, acted FROM action_log ORDER BY id")
    assert order["status"] == "cancelled"
    assert refunds == [{"amount_paise": 45000, "method": "card"}]
    assert [(row["tool"], row["outcome"], row["acted"]) for row in log] == [
        ("find_order", "ok", 1),
        ("prepare_cancel_order", "ok", 0),  # ok, but nothing changed yet
        ("confirm", "refused", 0),  # the made-up ticket
        ("confirm", "ok", 1),  # the one real cancel
        ("confirm", "already_done", 0),
    ]


async def test_plain_mode_over_mcp_shows_plain_tools_and_reports_timeouts_as_errors(tmp_path):
    task = PROJECT_ROOT / "packs" / "food" / "tasks" / "food_03_missing_item_reply_lost.yaml"
    db_path = await make_call(str(task), "plain", False, str(tmp_path))
    async with stdio_client(server_for(db_path)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = [tool.name for tool in (await session.list_tools()).tools]
            assert "refund_order" in tools and "confirm" not in tools
            # The refund is saved but its reply is lost (task switch): the agent sees a tool error.
            refund = {"order_no": 4512, "customer_name": "Priya", "amount_rupees": 40, "reason": "missing_item"}
            lost, is_error = await call(session, "refund_order", **refund)  # waits 2 real seconds
            assert (lost["status"], is_error) == ("error", True)
            wrong, is_error = await call(session, "refund_order", **{**refund, "amount_rupees": 9999})
            assert (wrong["status"], is_error) == ("refused", False)
