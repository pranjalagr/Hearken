"""Run decision 2's Flash-Lite check: 20 Talker replies for a person to judge.

    python -m cli.flash_lite_check

Needs GEMINI_API_KEY in pakka/.env. Writes a report to runs/week1/ (not in
git) and prints where it is.
"""

from __future__ import annotations

import asyncio

from lab.week1.flash_lite_check import run

if __name__ == "__main__":
    path, replies = asyncio.run(run())
    errors = sum(1 for reply in replies if reply.error)
    print(f"{len(replies)} replies, {errors} errors")
    print(f"report: {path}")
