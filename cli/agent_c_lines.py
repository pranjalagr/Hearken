"""Make Agent C's pre-made lines once, as 8 kHz clips in the robot's own voice
(decision 11, build step 8).

    python -m cli.agent_c_lines list   # which clips are there and which are missing (no network)
    python -m cli.agent_c_lines make   # make the missing ones on Sarvam (needs the Sarvam key)

The clips go to the folder in agents/agent_c/settings.yaml (waiting_lines.clips_folder,
under runs/, outside git: sound files never go in git), one folder per voice. A clip
already there is never made again, so a rerun costs only what is missing. The voice
is the live voice's own model and speaker, so a clip and a live line sound the same.
Sarvam bills by the character: the whole set is a few hundred characters.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agents.agent_c.robot import PROJECT_ROOT, WORDS
from agents.agent_c.settings import AgentCSettings, load_settings
from agents.agent_c.talker.lines import clip_lines, clip_name
from common.audio.pcm import pcm16_to_float
from common.audio.sound_files import write_sound
from common.retry import worth_retrying
from common.wire.protocol import SAMPLE_RATE
from engine.contract.spoken import SpokenWords

SARVAM_STREAM_URL = "https://api.sarvam.ai/text-to-speech/stream"  # the one lab/week1/voice_test.py measured
REQUEST_TIMEOUT_S = 30.0  # one short line; past this it is stuck, not slow
TRIES = 3  # a 429, a 5xx or a stuck request is tried again (common/retry.py); anything else fails at once
RETRY_WAIT_S = 2.0


def clip_folder(settings: AgentCSettings) -> Path:
    return PROJECT_ROOT / settings.section("waiting_lines")["clips_folder"] / settings.voice["speaker"]


def wanted(settings: AgentCSettings) -> list[tuple[Path, str]]:
    """(file, words) for every line that should have a clip."""
    folder = clip_folder(settings)
    return [(folder / clip_name(*line), line[3]) for line in clip_lines(SpokenWords.from_yaml(WORDS), settings.greeting)]


def sarvam_line(text: str, settings: AgentCSettings, key: str,
                opener: Callable[..., Any] = urllib.request.urlopen) -> bytes:
    """One line as raw 16-bit 8 kHz sound, from the live voice's model and speaker."""
    body = {"text": text, "language_code": settings.voice["language"], "model": settings.voice["model"],
            "speaker": settings.voice["speaker"], "speech_sample_rate": SAMPLE_RATE,
            "output_audio_codec": "linear16", "enable_preprocessing": True}
    request = urllib.request.Request(SARVAM_STREAM_URL, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                     headers={"api-subscription-key": key, "Content-Type": "application/json"},
                                     method="POST")
    with opener(request, timeout=REQUEST_TIMEOUT_S) as response:
        kind = response.headers.get("Content-Type", "")
        if not kind.startswith("audio/pcm"):
            raise RuntimeError(f"expected raw PCM audio, got {kind!r}")
        pcm = response.read()
    if not pcm or len(pcm) % 2:
        raise RuntimeError(f"broken audio: {len(pcm)} bytes")
    return pcm


def make(settings: AgentCSettings, key: str, opener: Callable[..., Any] = urllib.request.urlopen,
         sleep: Callable[[float], None] = time.sleep, log: Callable[[str], None] = print) -> int:
    """Makes every missing clip; returns how many were made."""
    made = 0
    for path, text in wanted(settings):
        if path.exists():
            continue
        for attempt in range(TRIES):
            try:
                pcm = sarvam_line(text, settings, key, opener)
                break
            except Exception as error:
                if attempt == TRIES - 1 or not worth_retrying(error):
                    raise
                log(f"retrying {path.name}: {error}")
                sleep(RETRY_WAIT_S)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp.wav")
        write_sound(temp, pcm16_to_float(pcm), SAMPLE_RATE)
        temp.replace(path)  # whole or not at all: a half-written clip must never play
        made += 1
        log(f"made {path.name}: {text}")
    return made


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Agent C's pre-made lines.")
    parser.add_argument("action", choices=["list", "make"])
    args = parser.parse_args(argv)
    settings = load_settings()
    if args.action == "list":
        for path, text in wanted(settings):
            print(f"{'have   ' if path.exists() else 'MISSING'} {path.name}  {text}")
        return
    from common.env import load_env, sarvam_key, sarvam_key_name

    load_env()
    print(f"using the Sarvam key in {sarvam_key_name()}")  # the name only, never the key
    made = make(settings, sarvam_key())
    print(f"made {made} clips in {clip_folder(settings)}")


if __name__ == "__main__":
    main(sys.argv[1:])
