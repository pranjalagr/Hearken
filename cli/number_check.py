"""Can the Talker's AI point at the order number, with code reading it? (Part 2,
decision 27.) The AI picks the words, code reads them, on the transcript test set in
lab/week1/number_transcripts.py.

    python -m cli.number_check [--mode list|pick] [--tries 2] [--label run2] [--keys GROQ_API_KEY,...]
    python -m cli.number_check --provider sarvam --tries 1 --label sarvam1 [--kinds stammer,...]

list (the default, 7 Nov 2026): the AI lists and tags every number; code takes one order
number at once and sends two or more to the Thinker. pick: the AI picks one (run1).

Real Groq requests: one per case per try (about 830 a try), shared over the keys.
Every answer is saved as it comes (runs/week1/number_check_<label>.jsonl, not in git):
run the same command again to finish what's left without asking anything twice.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime

from common.env import load_env, require
from lab.week1.number_check import SARVAM_MODEL, live_run

KEYS = {"groq": "GROQ_API_KEY,GROQ_API_KEY_2,GROQ_API_KEY_3,GROQ_API_KEY_4", "sarvam": "SARVAM_API_KEY"}
MODELS = {"groq": "openai/gpt-oss-20b", "sarvam": SARVAM_MODEL}


def main() -> None:
    parser = argparse.ArgumentParser(description="The order-number check.")
    parser.add_argument("--tries", type=int, default=2, help="how many times each case is asked (the AI varies)")
    parser.add_argument("--label", default=datetime.now().strftime("%Y%m%d"), help="same label = resume")
    parser.add_argument("--provider", choices=["groq", "sarvam"], default="groq")
    parser.add_argument("--keys", default="", help="the .env names of the keys to share the work over; empty = the provider's")
    parser.add_argument("--mode", choices=["list", "pick"], default="list")
    # 7 Nov 2026: qwen/qwen3.8-27b with its thinking off ("none") made 3 wrong numbers on
    # the 205 Hindi-grammar cases, against 41 for gpt-oss-20b, at half the time.
    parser.add_argument("--model", default="", help="empty = the provider's")
    parser.add_argument("--effort", default="low", help="reasoning effort, Groq only; qwen takes none (Sarvam's thinking is always off)")
    # 8 Nov 2026: the Hindi-grammar kinds (corrections, taken back, two orders, "X ya Y")
    # are where wrong numbers come from, so a new prompt is tried on them first.
    parser.add_argument("--kinds", default="", help="comma-separated case kinds; empty = all")
    args = parser.parse_args()
    load_env()
    keys = {name: require(name) for name in (args.keys or KEYS[args.provider]).split(",")}  # names only: keys are never printed
    path, results = asyncio.run(live_run(keys, args.tries, args.label, lambda line: print(line, flush=True), args.mode,
                                         args.model or MODELS[args.provider], args.effort,
                                         set(args.kinds.split(",")) if args.kinds else None, args.provider))
    print(f"{len(results)} answers marked")
    print(f"report: {path}")


if __name__ == "__main__":
    main()
