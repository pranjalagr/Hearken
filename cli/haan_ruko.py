"""The "haan vs ruko" model (Part 4): what a caller says while the robot talks, keep or stop?

    python -m cli.haan_ruko make [--limit 20]
        makes the voice clips listed in lab/haan_ruko/phrases.yaml on Smallest.ai. Each
        clip is saved by its voice and words, so a rerun pays only for what's missing.
    python -m cli.haan_ruko time
        finds where the speech is in every clip, with the voice detector (free).
    python -m cli.haan_ruko import-real [--files 10]
        IndicVoices' first Hindi files, one wav per piece under lab/sounds/ (free).
    python -m cli.haan_ruko import-more [--first 10] [--last 82] [--ahead 1]
        the other Hindi files, one at a time: download, cut, delete the raw file. What they
        keep is one packed FLAC per file in ~/hearken_data/indicvoices, outside OneDrive. A
        stopped run carries on where it stopped (free).
    python -m cli.haan_ruko train [--rounds 15] [--real]
        trains the model on this computer (on the GPU if torch can use it), tests it, saves
        it as ONNX and writes the report to runs/week1/haan_ruko/ (free).
    python -m cli.haan_ruko record
        the page for recording a real voice for the test: http://127.0.0.1:8093 (Chrome or Edge).
    python -m cli.haan_ruko test-human [--speaker s1]
        tests the saved model on those recordings, never trained on (free).

Clips stay under lab/sounds/haan_ruko/, results under runs/ (neither is in git).
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime
from pathlib import Path

from common.env import load_env, require
from lab.caller.mouth.voice import SavedClips, SmallestVoice
from lab.caller.settings import load_caller_settings
from lab.haan_ruko.clips import ROOT, SOUNDS, load_manifest, make_clips
from lab.haan_ruko.phrases import Plan, load_phrases, plan_clips


def smallest_maker():
    key = require("SMALLEST_API_KEY")  # read here, never printed
    settings = load_caller_settings()
    models = {choice.voice: choice.model for choice in [*settings.voices.values(), *settings.angry_voices.values()]}
    saved = SavedClips(SOUNDS / "raw")

    async def make(plan: Plan) -> Path:
        await SmallestVoice(key, plan.voice, models[plan.voice], saved).pieces(plan.text)  # asks only if missing
        return saved.path(plan.voice, plan.text)

    return make


def main() -> None:
    parser = argparse.ArgumentParser(description="The haan vs ruko model.")
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("make", help="make the voice clips")
    make.add_argument("--limit", type=int, default=None, help="only this many clips (to try it out)")
    commands.add_parser("time", help="find where the speech is in every clip")
    train = commands.add_parser("train", help="train, test and save the model")
    train.add_argument("--rounds", type=int, default=15)
    train.add_argument("--real", action="store_true", help="also train and test on IndicVoices' real speakers")
    evaluate = commands.add_parser("evaluate", help="test the saved model again and rewrite its report, without training")
    evaluate.add_argument("--real", action="store_true", help="include IndicVoices' real speakers")
    real = commands.add_parser("import-real", help="download and pick IndicVoices' Hindi phone conversations")
    real.add_argument("--files", type=int, default=10, help="how many of the 82 Hindi files (about 0.5 GB each)")
    more = commands.add_parser("import-more", help="the other Hindi files, one at a time, each raw file deleted once cut")
    more.add_argument("--first", type=int, default=10)
    more.add_argument("--last", type=int, default=82, help="one past the last file")
    more.add_argument("--ahead", type=int, default=0, help="files downloading while one is cut (each one more raw file on disk)")
    record = commands.add_parser("record", help="record a real voice for the test")
    record.add_argument("--port", type=int, default=8093)
    human = commands.add_parser("test-human", help="test the saved model on the recordings")
    human.add_argument("--speaker", default="s1")
    args = parser.parse_args()
    out = ROOT / "runs" / "week1" / "haan_ruko"

    if args.command == "make":
        load_env()
        plans = plan_clips(load_phrases())
        chosen = plans[: args.limit] if args.limit else plans
        print(f"{len(chosen)} of {len(plans)} clips, {sum(len(p.text) for p in chosen)} characters", flush=True)
        made = asyncio.run(make_clips(chosen, smallest_maker(), log=lambda line: print(line, flush=True)))
        print(f"{len(made)} made: {dict(Counter(m['group'] for m in made))}; held back: {sum(m['held_back'] for m in made)}")
    elif args.command == "time":
        from lab.eot_test.human import load_detector
        from lab.haan_ruko.speech import time_all

        data = load_phrases()
        timings = time_all(load_manifest(), SOUNDS, load_detector(), data["keep"]["train"] + data["keep"]["held_back"],
                           log=lambda line: print(line, flush=True))
        left_out = Counter(t.why_left_out for t in timings if t.why_left_out)
        print(f"{len(timings) - sum(left_out.values())} usable, left out: {dict(left_out)}")
    elif args.command in ("train", "evaluate"):
        import json

        from lab.eot_test.clips import NOISE_FOLDER
        from lab.haan_ruko.model import OnnxModel, export_onnx
        from lab.haan_ruko.report import check_speed, report
        from lab.haan_ruko.score import play, zero_miss_line
        from lab.haan_ruko.sound import TEST_LINES, Noises
        from lab.haan_ruko.speech import load_timings
        from lab.haan_ruko.train import save, split, train

        sets = split(load_manifest(), load_timings(SOUNDS), SOUNDS)
        if args.real:
            from lab.haan_ruko.indicvoices import MORE, is_short_no, load_more, load_pieces
            from lab.haan_ruko.train import NO_PRACTICE, add_real, practise_more

            machine_train = len(sets.train)
            add_real(sets, load_pieces())
            newer = load_more() if (MORE / "packs").exists() else []
            add_real(sets, newer)
            print(f"real pieces from files 10 on: {len(newer)} ({dict(Counter(c['answer'] for c, _, _ in newer))})", flush=True)
            if args.command == "train":
                count = practise_more(sets.train, is_short_no, NO_PRACTICE)
                print(f"short real pieces with a no word, practised {NO_PRACTICE} times a round: {count}", flush=True)
            print(f"real people: {len(sets.train) - machine_train} pieces in training, {len(sets.real_check)} checking, "
                  f"{len(sets.real_test)} testing", flush=True)
        noises = Noises(NOISE_FOLDER)
        print(f"{len(sets.train)} training clips, {len(sets.check)} in the left-out voice, {len(sets.test)} held back", flush=True)
        if args.command == "train":
            from lab.haan_ruko.windows import Helpers

            # The CPU's other cores make each round's windows while this process feeds the GPU.
            with Helpers(NOISE_FOLDER) as helpers:
                model, history = train(sets, noises, args.rounds, log=lambda line: print(line, flush=True), helpers=helpers)
            save(model, history, out)
            export_onnx(model, out / "haan_ruko.onnx")
        else:
            # The saved model, tested again without training: the test can be rerun for free.
            history = json.loads((out / "history.json").read_text(encoding="utf-8"))
        # Marked through the saved ONNX file, as a live robot would run it, not the training copy.
        onnx = OnnxModel(out / "haan_ruko.onnx")
        check = play(sets.check + sets.real_check, TEST_LINES, noises, onnx.chances)
        test = play(sets.test, TEST_LINES, noises, onnx.chances)
        real_test = play(sets.real_test, TEST_LINES, noises, onnx.chances) if sets.real_test else None
        line = zero_miss_line(check)
        real_train = sum(1 for c, _, _ in sets.train if c.get("role") == "train")
        counts = {"train": len(sets.train), "check": len(sets.check) + len(sets.real_check), "test": len(sets.test),
                  "real_train": real_train, "real_test": len(sets.real_test)}
        text = report(check, test, line, history, check_speed(onnx.chances), counts, datetime.now().strftime("%d %b %Y %H:%M"), real_test)
        (out / "report.md").write_text(text, encoding="utf-8")
        (out / "line.txt").write_text(f"{line:.6f}\n", encoding="utf-8")
        print(f"report: {out / 'report.md'}")
    elif args.command == "import-real":
        from lab.haan_ruko.indicvoices import import_files

        load_env()
        pieces = import_files(args.files, require("HUGGING_FACE_API_KEY"), log=lambda line: print(line, flush=True))
        print(f"{len(pieces)} pieces: {dict(Counter(p.answer for p in pieces))}, roles {dict(Counter(p.role for p in pieces))}, "
              f"{len({p.speaker for p in pieces})} speakers")
    elif args.command == "import-more":
        from lab.haan_ruko.indicvoices import MORE, forget_download, hf_download, import_more

        load_env()
        downloads = MORE / "downloading"  # each raw file lives here only until it is cut
        done = import_more(args.first, args.last, hf_download(require("HUGGING_FACE_API_KEY"), downloads),
                           lambda name: forget_download(downloads, name), ahead=args.ahead, log=lambda line: print(line, flush=True))
        print(f"done: {len(done)} files, {sum(d['kept'] for d in done)} pieces kept "
              f"({sum(d['keep'] for d in done)} listening), {sum(d['pack_mb'] for d in done):.0f} MB of packs", flush=True)
    elif args.command == "record":
        from lab.ears_test.record_server import make_server
        from lab.haan_ruko.human import HUMAN_SOUNDS, check_take, human_takes

        server = make_server(args.port, HUMAN_SOUNDS, human_takes(), Path(__file__).resolve().parents[1] / "lab" / "haan_ruko" / "record.html", check_take)
        print(f"recording page: http://127.0.0.1:{server.server_address[1]}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
    elif args.command == "test-human":
        from lab.eot_test.clips import NOISE_FOLDER
        from lab.eot_test.human import load_detector
        from lab.haan_ruko.human import prepare_human
        from lab.haan_ruko.model import OnnxModel
        from lab.haan_ruko.report import by_group, mistakes, table
        from lab.haan_ruko.score import play, separation
        from lab.haan_ruko.sound import TEST_LINES, Noises

        takes = prepare_human(args.speaker, load_detector(), log=lambda line: print(line, flush=True))
        if not takes:
            raise SystemExit("no recordings yet: record them first with `python -m cli.haan_ruko record`")
        line = float((out / "line.txt").read_text())
        played = play(takes, TEST_LINES, Noises(NOISE_FOLDER), OnnxModel(out / "haan_ruko.onnx").chances)
        held = [p for p in played if p.clip["held_back"]]
        text = "\n".join([f"# Haan vs ruko on a real voice ({datetime.now():%d %b %Y %H:%M})", "",
                          f"{len(takes)} takes by speaker {args.speaker}, never trained on. The same line as the machine "
                          f"test ({line:.3f}), chosen on the machine voices, not on these takes.", "",
                          f"Tells real speech from listening sounds: {separation(played):.3f} (1 perfect, 0.5 guessing).", "",
                          *table(played, line, "All takes"), *by_group(played, line),
                          *table(held, line, "Only phrases the model never trained on"),
                          "## Mistakes", "", *mistakes(played, line, limit=80)])
        (out / "human_report.md").write_text(text + "\n", encoding="utf-8")
        print(f"report: {out / 'human_report.md'}")


if __name__ == "__main__":
    main()
