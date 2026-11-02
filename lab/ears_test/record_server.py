"""The recording page's small server: this computer only.

Browsers only allow the microphone on https or on this computer's own address,
so it listens on 127.0.0.1, as the web phone does. What it receives is checked,
because a broken or silent recording would quietly lower a service's score.
"""

from __future__ import annotations

import io
import json
import os
import re
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np
import soundfile

from lab.ears_test.clips import SOUNDS, EarsLine, load_lines, raw_path

PAGE_DIR = Path(__file__).with_name("record_page")
# A short name like "s1", never a person's real name: these end up in file paths.
_SPEAKER = re.compile(r"[a-z0-9]{1,12}")
MAX_BYTES = 8_000_000  # about 80 s of 48 kHz sound: far more than any line needs
MIN_S, MAX_S = 0.3, 30.0
# Quieter than this at its loudest is a muted or badly set microphone. Raised
# from 0.01 after a take peaking at 0.015 was let through (2 Nov 2026): raising
# the level later would raise the room's hiss with it, making the test unfair.
MIN_PEAK = 0.05
_BLOCK = 128  # the size of block the browser's audio system hands over
_LOUD_LEVEL = 3  # out of 32767: exact repeats in near-silence mean nothing
MAX_REPEATED = 0.05


class BadRecording(ValueError):
    """Why a recording was refused, in words the page shows to the person."""


def repeated_share(samples: np.ndarray) -> float:
    """How much of the sound repeats itself exactly, block after block. Real
    speech never does this, but a recorder running twice writes every block
    twice, or three times on the third take (2 Nov 2026)."""
    usable = len(samples) - len(samples) % _BLOCK
    blocks = np.rint(samples[:usable, 0] * 32767).astype(np.int32).reshape(-1, _BLOCK)
    loud = np.abs(blocks).max(axis=1) > _LOUD_LEVEL
    if not loud.any():
        return 0.0
    return float(np.sum(np.all(blocks[1:] == blocks[:-1], axis=1) & loud[1:]) / loud.sum())


def check_recording(data: bytes) -> None:
    try:
        samples, rate = soundfile.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception as error:
        raise BadRecording("that isn't a readable sound file") from error
    if repeated_share(samples) > MAX_REPEATED:
        raise BadRecording("the recorder doubled the sound: please refresh this page and record the line again")
    seconds = len(samples) / rate
    if not MIN_S <= seconds <= MAX_S:
        raise BadRecording(f"it lasted {seconds:.1f} s; a line must take {MIN_S} to {MAX_S:.0f} s")
    if float(np.abs(samples).max()) < MIN_PEAK:
        raise BadRecording("it was too quiet: speak closer, or raise the microphone level in Windows sound settings")


def save_recording(speaker: str, line_id: str, data: bytes, lines: list[EarsLine], sounds: Path = SOUNDS) -> Path:
    if not _SPEAKER.fullmatch(speaker):
        raise BadRecording("the speaker must be a short code like s1: small letters and digits only")
    if line_id not in {line.id for line in lines}:
        raise BadRecording(f"unknown line {line_id!r}")
    check_recording(data)
    target = raw_path(speaker, line_id, sounds)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(".tmp")
    temp.write_bytes(data)
    os.replace(temp, target)  # a redo replaces the old take whole, or not at all
    return target


class _RecordHandler(SimpleHTTPRequestHandler):
    # Pinned, as on the web phone: Windows can map .js to text/plain, and browsers refuse that for a module.
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".js": "text/javascript",
                      ".html": "text/html; charset=utf-8"}

    def __init__(self, *args, lines: list[EarsLine], sounds: Path, **kwargs) -> None:
        self.lines, self.sounds = lines, sounds
        super().__init__(*args, directory=str(PAGE_DIR), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format: str, *args) -> None:
        pass

    def _json(self, status: int, body: object) -> None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        url = urlsplit(self.path)
        if url.path == "/lines":
            speaker = parse_qs(url.query).get("speaker", [""])[0]
            known = _SPEAKER.fullmatch(speaker) is not None
            self._json(200, [{"id": line.id, "say": line.say,
                              "done": known and raw_path(speaker, line.id, self.sounds).exists()} for line in self.lines])
            return
        super().do_GET()

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        if not 0 < length <= MAX_BYTES:
            self._json(400, {"error": "the recording is empty or too long"})
            return
        # Read before any answer, even a refusal: answering while the upload is
        # still coming makes Windows abort the connection, so the sender sees a
        # broken line instead of the reason (a test caught this 1 run in 5).
        body = self.rfile.read(length)
        # Any website could otherwise post to this address from the person's own browser.
        origin = self.headers.get("Origin")
        if origin and urlsplit(origin).netloc != self.headers.get("Host"):
            self._json(403, {"error": "only this page may save recordings"})
            return
        url = urlsplit(self.path)
        if url.path != "/save":
            self._json(404, {"error": "no such address"})
        else:
            query = parse_qs(url.query)
            try:
                save_recording(query.get("speaker", [""])[0], query.get("line", [""])[0], body, self.lines, self.sounds)
                self._json(200, {"saved": True})
            except BadRecording as error:
                self._json(400, {"error": str(error)})


def make_server(port: int = 8090, sounds: Path = SOUNDS, lines: list[EarsLine] | None = None) -> ThreadingHTTPServer:
    handler = partial(_RecordHandler, lines=lines if lines is not None else load_lines(), sounds=sounds)
    return ThreadingHTTPServer(("127.0.0.1", port), handler)
