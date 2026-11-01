"""Makes one private database copy per call.

Many test calls use the same orders at the same time, so every call gets its
own copy of one starting file. A shared database would let one call's cancel
break another call's task. Copying a small file is fast, and the finished copy
doubles as the call's black-box snapshot.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

from engine.contract.pack import Pack
from engine.storage.call_settings import CallSettings, write_call_settings
from engine.storage.database import sqlite_session

ENGINE_TABLES = Path(__file__).with_name("engine_tables.sql")


class CallFactory:
    def __init__(self, pack: Pack, workdir: Path) -> None:
        self._pack = pack
        self._template = workdir / "templates" / f"{pack.name}.db"
        self._calls_dir = workdir / "calls"
        self._template_built = False
        # Many calls start at once in the lab (Part 5): without this, each would
        # rebuild the one template file while the others read it, and on
        # 1 Nov 2026 five such calls locked each other up for good.
        self._building = asyncio.Lock()

    async def build_template(self) -> Path:
        """Build the pack's starting file: engine tables + pack tables + seed data.

        Rebuilt once per factory, so a changed schema or seed is always picked up.
        """
        self._template.parent.mkdir(parents=True, exist_ok=True)
        self._template.unlink(missing_ok=True)
        async with sqlite_session(self._template) as db:
            await db.run_script(ENGINE_TABLES.read_text(encoding="utf-8"))
            await db.run_script(self._pack.schema_sql())
            async with db.transaction() as tx:
                await self._pack.load_seed(tx)
        self._template_built = True
        return self._template

    async def new_call(self, settings: CallSettings) -> Path:
        """Make a fresh copy for one call and write its settings into it."""
        if settings.pack != self._pack.name:
            raise ValueError(f"call is for pack {settings.pack!r}, factory makes {self._pack.name!r}")
        async with self._building:
            if not self._template_built:
                await self.build_template()
        path = self._calls_dir / f"{settings.call_id}.db"
        if path.exists():
            # Reusing a copy would mean the call does not start fresh.
            raise FileExistsError(f"call copy already exists: {path}")
        self._calls_dir.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(_copy_database, self._template, path)
        async with sqlite_session(path) as db:
            async with db.transaction() as tx:
                await write_call_settings(tx, settings)
        return path


def _copy_database(source: Path, target: Path) -> None:
    # SQLite's own backup copy, not a plain file copy: it stays correct even
    # if part of the data still sits in the WAL side file.
    src = sqlite3.connect(source)
    dst = sqlite3.connect(target)
    try:
        src.backup(dst)
    finally:
        # Close both, so Windows releases the files for the next copy.
        src.close()
        dst.close()
