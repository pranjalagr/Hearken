"""The page where a person approves the unfinished clips by ear: this computer only.

Only a person can tell whether a cut clip really sounds like someone who paused, and a
clip that sounds finished under an "unfinished" label would mark good models wrong.
Every verdict is saved the moment it's given, so the listening can stop and carry on.
"""

from __future__ import annotations

import json
import os
import re
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from lab.eot_test.clips import SOUNDS, clip_path, load_manifest

PAGE_DIR = Path(__file__).with_name("review_page")
VERDICTS = ("ok", "reject")
_AUDIO = re.compile(r"/audio/(L\d+_[a-z]+)_(unfinished|finished)\.wav")


def verdicts_path(sounds: Path) -> Path:
    return sounds / "verdicts.json"


def load_verdicts(sounds: Path = SOUNDS) -> dict[str, str]:
    path = verdicts_path(sounds)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_verdict(clip_id: str, verdict: str, known: set[str], sounds: Path = SOUNDS) -> None:
    if clip_id not in known:
        raise ValueError(f"unknown clip {clip_id!r}")
    if verdict not in VERDICTS:
        raise ValueError(f"unknown verdict {verdict!r}")
    verdicts = load_verdicts(sounds)
    verdicts[clip_id] = verdict
    temp = verdicts_path(sounds).with_suffix(".tmp")
    temp.write_text(json.dumps(verdicts, indent=1), encoding="utf-8")
    os.replace(temp, verdicts_path(sounds))  # whole or not at all, so a crash can't lose the rest


class _ReviewHandler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".js": "text/javascript",
                      ".html": "text/html; charset=utf-8"}

    def __init__(self, *args, sounds: Path, **kwargs) -> None:
        self.sounds = sounds
        super().__init__(*args, directory=str(PAGE_DIR), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format: str, *args) -> None:
        pass

    def _send(self, status: int, body: bytes, kind: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, body: object) -> None:
        self._send(status, json.dumps(body, ensure_ascii=False).encode("utf-8"), "application/json")

    def do_GET(self) -> None:
        url = urlsplit(self.path)
        if url.path == "/clips":
            verdicts = load_verdicts(self.sounds)
            self._json(200, [{"id": clip["plan"]["id"], "text": clip["plan"]["text"], "voice": clip["plan"]["voice"],
                              "noise": clip["plan"]["noise"], "level": clip["plan"]["level"],
                              "verdict": verdicts.get(clip["plan"]["id"])} for clip in load_manifest(self.sounds)])
            return
        if url.path.startswith("/audio/"):
            # Only names of this exact shape: the path never reaches the disk as typed.
            match = _AUDIO.fullmatch(url.path)
            path = clip_path(self.sounds, match[1], match[2]) if match else None
            if path is None or not path.exists():
                self._json(404, {"error": "no such clip"})
                return
            self._send(200, path.read_bytes(), "audio/wav")
            return
        super().do_GET()

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        # Any website could otherwise post to this address from the person's own browser.
        origin = self.headers.get("Origin")
        if origin and urlsplit(origin).netloc != self.headers.get("Host"):
            self._json(403, {"error": "only this page may save verdicts"})
            return
        url = urlsplit(self.path)
        if url.path != "/verdict":
            self._json(404, {"error": "no such address"})
            return
        query = parse_qs(url.query)
        try:
            known = {clip["plan"]["id"] for clip in load_manifest(self.sounds)}
            save_verdict(query.get("clip", [""])[0], query.get("verdict", [""])[0], known, self.sounds)
            self._json(200, {"saved": True})
        except ValueError as error:
            self._json(400, {"error": str(error)})


def make_server(port: int = 8091, sounds: Path = SOUNDS) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("127.0.0.1", port), partial(_ReviewHandler, sounds=sounds))
