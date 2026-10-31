"""The storage layer. The engine talks to the two small interfaces here, never
to SQLite directly.

That is what lets Postgres be added later (decision 1) without touching the
engine. Differences between databases, such as how a row gets locked, belong
inside an implementation class, never in engine code.
"""

from __future__ import annotations

import asyncio
import sqlite3
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path
from typing import Any

import aiosqlite

Row = dict[str, Any]

# How long a writer waits for another writer to finish before giving up.
# Two confirms arriving at the same moment must queue up, not fail.
BUSY_WAIT_S = 5.0


class Tx(ABC):
    """One open transaction. Everything done through it is saved together, or not at all."""

    @abstractmethod
    async def fetch_one(self, sql: str, params: tuple = ()) -> Row | None: ...

    @abstractmethod
    async def fetch_all(self, sql: str, params: tuple = ()) -> list[Row]: ...

    @abstractmethod
    async def execute(self, sql: str, params: tuple = ()) -> int:
        """Run a change and return how many rows it touched."""


class Database(ABC):
    """One call's database."""

    @abstractmethod
    def transaction(self, write: bool = True) -> AbstractAsyncContextManager[Tx]:
        """Open a transaction. `write=False` is for short read-only peeks."""

    @abstractmethod
    async def close(self) -> None: ...


class SqliteTx(Tx):
    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    async def fetch_one(self, sql: str, params: tuple = ()) -> Row | None:
        async with self._conn.execute(sql, params) as cursor:
            row = await cursor.fetchone()
        return dict(row) if row is not None else None

    async def fetch_all(self, sql: str, params: tuple = ()) -> list[Row]:
        async with self._conn.execute(sql, params) as cursor:
            rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def execute(self, sql: str, params: tuple = ()) -> int:
        async with self._conn.execute(sql, params) as cursor:
            return cursor.rowcount


class SqliteDatabase(Database):
    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn
        # One connection can run only one transaction at a time. Tool calls in
        # the same process take turns here. Connections in other processes
        # (the Guard, a second backend) are kept apart by SQLite's write lock.
        self._turn = asyncio.Lock()

    @classmethod
    async def open(cls, path: str | Path) -> SqliteDatabase:
        # isolation_level=None stops Python from starting transactions on its
        # own, because we start them ourselves with BEGIN IMMEDIATE.
        conn = await aiosqlite.connect(str(path), isolation_level=None, timeout=BUSY_WAIT_S)
        conn.row_factory = sqlite3.Row
        # WAL lets readers work while one writer writes. synchronous=NORMAL is
        # the safe pairing with WAL and skips a slow disk flush on every commit.
        await conn.execute("PRAGMA journal_mode = WAL")
        await conn.execute("PRAGMA synchronous = NORMAL")
        # SQLite ignores REFERENCES unless this is switched on, per connection.
        await conn.execute("PRAGMA foreign_keys = ON")
        return cls(conn)

    @asynccontextmanager
    async def transaction(self, write: bool = True) -> AsyncIterator[Tx]:
        async with self._turn:
            # IMMEDIATE takes the write lock at the very start. With a plain
            # BEGIN, two confirms could both read "token waiting" before either
            # one writes, and both would act.
            await self._conn.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            try:
                yield SqliteTx(self._conn)
            except BaseException:
                await self._conn.execute("ROLLBACK")
                raise
            await self._conn.execute("COMMIT")

    async def run_script(self, sql: str) -> None:
        """Run many statements at once. Only used to build a pack's starting file."""
        await self._conn.executescript(sql)

    async def close(self) -> None:
        await self._conn.close()


@asynccontextmanager
async def sqlite_session(path: str | Path) -> AsyncIterator[SqliteDatabase]:
    """Open a database for a short job and always close it.

    Windows keeps an open database file locked, so a forgotten close would
    block copying or deleting it later.
    """
    db = await SqliteDatabase.open(path)
    try:
        yield db
    finally:
        await db.close()
