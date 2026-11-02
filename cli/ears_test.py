"""Run the week-1 ears test (Part 2, decision 24): which speech-to-text wins?

    python -m cli.ears_test record
        the recording page, at http://127.0.0.1:8090 (Chrome or Edge).
    python -m cli.ears_test prepare
        makes the phone and phone + noise versions of every recording.
    python -m cli.ears_test run --limit 1 --services sarvam
        streams clips to the services; start small: --limit is clips per service.
    python -m cli.ears_test report
        writes the comparison from the saved results (free: no new requests).

Needs SARVAM_API_KEY, SONIOX_API_KEY, DEEPGRAM_API_KEY and GEMINI_API_KEY in
pakka/.env; a service without its key is skipped, and says so. Recordings stay
under lab/sounds/ears/, results under runs/week1/ears_test/ (neither is in git).
Clips already heard are never sent again, so a run can stop and carry on.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from common.env import load_env
from lab.ears_test.clips import ROOT, SOUNDS, VERSIONS, load_lines, prepare
from lab.ears_test.runner import load_results, run_all
from lab.ears_test.score import report, score_service
from lab.ears_test.services import Deepgram, GoogleLive, Sarvam, Service, Soniox

OUT_DIR = ROOT / "runs" / "week1" / "ears_test"
RESULTS = OUT_DIR / "results.json"
# Soniox is left out: its code stays, but adding funds needs a card, which we
# don't have (2 Nov 2026), so it can't be run or compared.
SERVICE_NAMES = ("sarvam", "deepgram", "google")
# US dollars per audio minute, filled in only from each company's own pricing
# page once checked. Until then the report says so rather than show a guess.
PRICES: dict[str, float | None] = {
    "sarvam": None,
    "deepgram": 0.0092,  # deepgram.com/pricing, 2 Nov 2026: Nova-3 Multilingual streaming, pay as you go
    "google": None,
}
GOOGLE_TIMEOUT_S = 30.0


def make_services(names: list[str]) -> tuple[list[Service], list[str]]:
    """(services ready to use, names skipped for having no key). Keys are only
    read from the environment and never printed."""
    import os

    load_env()
    services: list[Service] = []
    skipped: list[str] = []
    for name in names:
        variable = {"sarvam": "SARVAM_API_KEY", "soniox": "SONIOX_API_KEY", "deepgram": "DEEPGRAM_API_KEY",
                    "google": "GEMINI_API_KEY"}[name]
        key = os.environ.get(variable, "").strip()
        if not key:
            skipped.append(f"{name} (no {variable} in .env)")
        elif name == "google":
            from common.google import google_client

            services.append(GoogleLive(google_client(key, GOOGLE_TIMEOUT_S)))
        else:
            services.append({"sarvam": Sarvam, "soniox": Soniox, "deepgram": Deepgram}[name](key))
    return services, skipped


def _manifest(sounds: Path = SOUNDS) -> dict:
    path = sounds / "clips.json"
    if not path.exists():
        raise SystemExit("no clips yet: run `python -m cli.ears_test prepare` first")
    return json.loads(path.read_text(encoding="utf-8"))


def write_report(names: list[str]) -> Path:
    manifest = _manifest()
    results = load_results(RESULTS)
    lines = load_lines()
    scores = [score_service(name, lines, results, manifest["clips"]) for name in names
              if any(key.startswith(f"{name}/") for key in results)]
    expected = len(manifest["clips"]) * len(VERSIONS)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    when = datetime.now()
    path = OUT_DIR / f"report_{when:%Y%m%d_%H%M}.md"
    path.write_text(report(scores, expected, PRICES, manifest["noise"], f"{when:%d %b %Y, %H:%M}"), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="The week-1 ears test.")
    commands = parser.add_subparsers(dest="command", required=True)
    record = commands.add_parser("record", help="the recording page")
    record.add_argument("--port", type=int, default=8090)
    commands.add_parser("prepare", help="make the phone versions of every recording")
    run = commands.add_parser("run", help="stream clips to the services")
    run.add_argument("--services", default=",".join(SERVICE_NAMES), help="comma-separated")
    run.add_argument("--limit", type=int, default=None, help="only this many clips per service (to try it out)")
    commands.add_parser("report", help="write the comparison from saved results")
    args = parser.parse_args()

    if args.command == "record":
        from lab.ears_test.record_server import make_server

        server = make_server(args.port)
        print(f"recording page: http://127.0.0.1:{server.server_address[1]}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
    elif args.command == "prepare":
        try:
            manifest = prepare()
        except ValueError as error:
            # No recordings yet, or too few for babble: the message says what to do.
            raise SystemExit(str(error)) from error
        print(f"{len(manifest['clips'])} recordings made into {len(VERSIONS)} versions each; noise: {manifest['noise']}")
    elif args.command == "run":
        names = [name.strip() for name in args.services.split(",")]
        for name in names:
            if name not in SERVICE_NAMES:
                parser.error(f"unknown service {name!r}; choose from {list(SERVICE_NAMES)}")
        clips = _manifest()["clips"]
        if args.limit is not None:
            clips = dict(sorted(clips.items())[:args.limit])
        services, skipped = make_services(names)
        for item in skipped:
            print(f"skipped: {item}")
        asyncio.run(run_all(services, clips, RESULTS, SOUNDS))
        print(f"report: {write_report(names)}")
    else:
        print(f"report: {write_report(list(SERVICE_NAMES))}")


if __name__ == "__main__":
    main()
