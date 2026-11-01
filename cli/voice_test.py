"""Run the week-1 voice test (Part 2, decisions 10 and 12).

    python -m cli.voice_test --round screen
        12 of the hardest lines through every candidate voice.
    python -m cli.voice_test --round full --voices sarvam-sunny-enhi,google-csagent-1
        all lines through the chosen finalists.
    python -m cli.voice_test --lines E3,E8,E9 --voices sarvam-sunny-enhi
        just these lines, e.g. to try a fix without a whole round.

Needs GEMINI_API_KEY and SARVAM_API_KEY in pakka/.env. Clips and the
listening pages go to runs/week1/voice_test/ (not in git); each kind of run
writes its own page (listen-screen.html, listen-full.html, listen-lines.html).
Clips already made are skipped, so a run stopped by a daily limit continues
where it left off.
"""

from __future__ import annotations

import argparse

from common.env import load_env, require
from common.google import google_client
from lab.week1.voice_test import (
    OUT_DIR,
    REQUEST_TIMEOUT_S,
    SCREEN_IDS,
    SETUPS,
    SETUPS_BY_NAME,
    google_speak,
    load_lines,
    run_round,
    sarvam_speak,
    timing_summary,
    write_page,
)


def main(round_name: str | None, line_ids: list[str] | None, voice_names: list[str] | None) -> None:
    load_env()
    lines = load_lines()
    if line_ids:
        # Kept in the file's order, so the page reads like the lines file.
        lines = [line for line in lines if line.id in line_ids]
    elif round_name == "screen":
        lines = [line for line in lines if line.id in SCREEN_IDS]
    label = "lines" if line_ids else round_name
    setups = [SETUPS_BY_NAME[name] for name in voice_names] if voice_names else list(SETUPS)
    sarvam_key = require("SARVAM_API_KEY")
    client = google_client(require("GEMINI_API_KEY"), REQUEST_TIMEOUT_S)

    def speak(setup, line):
        if setup.company == "sarvam":
            return sarvam_speak(setup, line, sarvam_key)
        return google_speak(setup, line, client)

    try:
        print(f"{label} run: {len(lines)} lines x {len(setups)} voices")
        timings = run_round(lines, setups, speak)
    finally:
        client.close()
    for setup in setups:
        summary = timing_summary(setup, timings, lines)
        print(f"{setup.name}: {summary['clips']} clips, {summary['errors']} failed, {summary['retries']} retries, "
              f"first sound median {summary['median_first_s']} s, slowest {summary['slowest_first_s']} s")
    print(f"listen: {write_page(lines, setups, timings, OUT_DIR, name=f'listen-{label}.html')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The week-1 voice test.")
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--round", choices=["screen", "full"])
    which.add_argument("--lines", help="comma-separated line ids, e.g. E3,E9")
    parser.add_argument("--voices", default=None, help="comma-separated voice names (default: all)")
    args = parser.parse_args()
    names = [name.strip() for name in args.voices.split(",")] if args.voices else None
    for name in names or []:
        if name not in SETUPS_BY_NAME:
            parser.error(f"unknown voice {name!r}; choose from {sorted(SETUPS_BY_NAME)}")
    ids = [line_id.strip() for line_id in args.lines.split(",")] if args.lines else None
    known = {line.id for line in load_lines()}
    for line_id in ids or []:
        if line_id not in known:
            # A typo must fail here, not quietly make an empty run.
            parser.error(f"unknown line {line_id!r}")
    main(args.round, ids, names)
