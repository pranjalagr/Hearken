"""Checks the environment itself, so a broken setup fails here with a clear
message instead of as a confusing error deep inside a backend test."""

import sqlite3

import aiosqlite

# STRICT tables (decision 1) arrived in SQLite 3.37. Python ships its own
# SQLite, so an older Python can silently bring an older one.
MIN_SQLITE = (3, 37, 0)


def test_sqlite_supports_strict_tables():
    assert sqlite3.sqlite_version_info >= MIN_SQLITE, (
        f"SQLite {sqlite3.sqlite_version} is too old for STRICT tables; "
        f"need {'.'.join(map(str, MIN_SQLITE))}+"
    )


async def test_async_sqlite_rejects_wrong_type_in_strict_table():
    # Money is stored as whole paise. A STRICT table must refuse a decimal,
    # otherwise ₹450.50 could sneak in as 450.5 and round wrongly later.
    async with aiosqlite.connect(":memory:") as db:
        await db.execute("CREATE TABLE t (amount_paise INTEGER NOT NULL) STRICT")
        try:
            await db.execute("INSERT INTO t VALUES (450.5)")
        except sqlite3.IntegrityError:
            return
        raise AssertionError("STRICT table accepted a decimal amount")
