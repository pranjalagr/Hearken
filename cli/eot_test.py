"""The end-of-turn test (Part 2, decision 25): which model tells best when a caller has finished?

    python -m cli.eot_test make [--limit 5]
        makes the clips from lab/eot_test/lines.yaml on Smallest.ai. Each recording is
        saved by its voice and text, so a rerun pays only for what's missing.
    python -m cli.eot_test review
        the page for approving the unfinished clips by ear: http://127.0.0.1:8091
    python -m cli.eot_test record
        the page for recording a real voice: http://127.0.0.1:8092 (Chrome or Edge).
    python -m cli.eot_test run [--set human]
        runs the models on the approved clips, on this computer's CPU (free), and writes
        the comparison to runs/week1/eot_test/.

Clips and verdicts stay under lab/sounds/eot/, results under runs/ (neither is in git).
"""

from __future__ import annotations

import argparse
import asyncio
import io
from datetime import datetime

import numpy as np
import soundfile
import yaml

from common.env import load_env, require
from lab.caller.mouth.voice import SavedClips, SmallestVoice
from lab.caller.settings import SETTINGS_FILE
from lab.eot_test.clips import ROOT, SOUNDS, ClipPlan, load_lines, load_manifest, make_clips, plan_clips


def voice_models() -> dict[str, str]:
    """Each caller voice's model, from the lab's own settings."""
    settings = yaml.safe_load(SETTINGS_FILE.read_text(encoding="utf-8"))
    return {entry["voice"]: entry["model"] for entry in [*settings["voices"].values(), *settings["angry_voices"].values()]}


def smallest_recordings():
    key = require("SMALLEST_API_KEY")  # read here, never printed
    models = voice_models()
    saved = SavedClips(SOUNDS / "raw")

    async def raw(plan: ClipPlan) -> tuple[np.ndarray, int]:
        await SmallestVoice(key, plan.voice, models[plan.voice], saved).pieces(plan.text)  # makes it only if missing
        samples, rate = soundfile.read(io.BytesIO(saved.path(plan.voice, plan.text).read_bytes()), dtype="float32", always_2d=True)
        return samples.mean(axis=1), rate

    return raw


def main() -> None:
    parser = argparse.ArgumentParser(description="The end-of-turn test.")
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("make", help="make the clips")
    make.add_argument("--limit", type=int, default=None, help="only this many clips (to try it out)")
    review = commands.add_parser("review", help="approve the unfinished clips by ear")
    review.add_argument("--port", type=int, default=8091)
    run = commands.add_parser("run", help="run the models on the approved clips and write the comparison")
    run.add_argument("--set", dest="clip_set", choices=["machine", "human"], default="machine")
    run.add_argument("--speaker", default="s1", help="the code used on the recording page")
    record = commands.add_parser("record", help="record a real voice")
    record.add_argument("--port", type=int, default=8092)
    args = parser.parse_args()

    if args.command == "make":
        load_env()
        plans = plan_clips(load_lines())[: args.limit]
        made = asyncio.run(make_clips(plans, smallest_recordings(), log=lambda line: print(line, flush=True)))
        print(f"{len(made)} of {len(plans)} clips made, each finished and unfinished, in {SOUNDS / 'clips'}")
    elif args.command == "run":
        # Imported here: these are heavy, and only this command needs them.
        from lab.eot_test.models import SPECS, load_model
        from lab.eot_test.review_server import load_verdicts
        from lab.eot_test.human import HUMAN_SOUNDS, prepare_human
        from lab.eot_test.score import HUMAN_NOTE, MACHINE_NOTE, approved, report, run_models, save_rows

        human = args.clip_set == "human"
        if human:
            print(f"{len(prepare_human(args.speaker))} recorded takes prepared", flush=True)
        sounds = HUMAN_SOUNDS if human else SOUNDS
        clips = approved(load_manifest(sounds), load_verdicts(sounds))
        print(f"{len(clips)} approved clips", flush=True)
        models = {spec.name: load_model(spec) for spec in SPECS}
        rows = run_models(clips, models, sounds, log=lambda line: print(line, flush=True))
        out = ROOT / "runs" / "week1" / "eot_test"
        prefix = "human_" if human else ""
        save_rows(rows, out / f"{prefix}scores.jsonl")
        (out / f"{prefix}report.md").write_text(report(rows, datetime.now().strftime("%d %b %Y %H:%M"), note=HUMAN_NOTE if human else MACHINE_NOTE), encoding="utf-8")
        print(f"report: {out / f'{prefix}report.md'}")
    elif args.command == "record":
        from pathlib import Path

        from lab.ears_test.record_server import make_server as make_record_server
        from lab.eot_test.human import HUMAN_SOUNDS, human_lines

        server = make_record_server(args.port, HUMAN_SOUNDS, human_lines(),
                                    Path(__file__).resolve().parents[1] / "lab" / "eot_test" / "record.html")
        print(f"recording page: http://127.0.0.1:{server.server_address[1]}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
    else:
        from lab.eot_test.review_server import make_server

        server = make_server(args.port)
        print(f"review page: http://127.0.0.1:{server.server_address[1]}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()


if __name__ == "__main__":
    main()
