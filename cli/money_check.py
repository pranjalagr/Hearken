"""Check that the Talker says money with paise as "249 rupaye 50 paise"
(Part 2, decision 10). Judged by code, not by ear.

    python -m cli.money_check

Needs GEMINI_API_KEY in pakka/.env. Writes a report to runs/week1/ (not in
git) and prints where it is.
"""

from __future__ import annotations

import asyncio

from lab.week1.money_check import run

if __name__ == "__main__":
    path, passed, total = asyncio.run(run())
    print(f"passed {passed} of {total}")
    print(f"report: {path}")
