"""Fetch the background noise recordings for the pretend caller (Part 3, C5).

    python -m cli.caller_sounds

Sound files stay out of git (C18), so this re-makes them. All are CC0 on BigSoundBank
(checked 4 Nov 2026): free for any use, no credit needed. Saved under lab/sounds/noise/.
"""

from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import soundfile

from common.audio.resample import resample
from common.audio.sound_files import read_sound, write_sound

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "lab" / "sounds" / "noise"
BASE = "https://bigsoundbank.com/UPLOAD/mp3/"
# C5 names a TV too, but no CC0 TV recording was found, so that kind waits.
SOURCES = {"traffic": "0122.mp3", "kitchen": "0624.mp3", "crowd": "0481.mp3"}
# Files are kept at the phone's working rate: smaller, and all the same.
KEEP_RATE = 16000


def main() -> None:
    FOLDER.mkdir(parents=True, exist_ok=True)
    for kind, name in SOURCES.items():
        target = FOLDER / f"{kind}.flac"
        if target.exists():
            print(f"{kind}: already here")
            continue
        with urllib.request.urlopen(BASE + name, timeout=60) as response:
            data = response.read()
        temp = FOLDER / f"{kind}.tmp.mp3"
        temp.write_bytes(data)
        samples, rate = read_sound(temp)
        temp.unlink()
        write_sound(target, resample(samples, rate, KEEP_RATE), KEEP_RATE)
        info = soundfile.info(io.BytesIO(target.read_bytes()))
        print(f"{kind}: {info.duration:.0f} s from {BASE}{name}")


if __name__ == "__main__":
    main()
