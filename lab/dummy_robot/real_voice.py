"""A puppet that really talks (Part 3, C14), in the voice the robots will use.

The puppet's beeps were enough for a scripted caller. A caller with real ears hears
nothing in a beep, so its checks need real speech. Every line is known before the
call, so all of them are made first and saved: a line is paid for once, ever.
"""

from __future__ import annotations

import re
import urllib.request
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from agents.agent_c.talker.voice_spelling import spell_for_voice
from common.audio.pcm import pcm16_to_float, split_pieces, float_to_pcm16
from common.audio.resample import PHONE_RATE
from common.audio.sound_files import read_sound, write_sound
from common.wire.protocol import PIECE_SAMPLES
from lab.caller.mouth.voice import SavedClips
from lab.dummy_robot.robot import AskKey, Say, SayWithPause, Step
from lab.week1.voice_test import SETUPS_BY_NAME, VoiceLine, sarvam_speak, with_retries

ROOT = Path(__file__).resolve().parents[2]
CLIPS = ROOT / "lab" / "sounds" / "robot"
SETUP = SETUPS_BY_NAME["sarvam-sunny-enhi"]  # decision 12's voice
# Enough to tell which language the voice should read a line in: lines that
# contain none of these are English. Every puppet line is checked by a test.
_HINDI_WORDS = re.compile(
    r"\b(hai|hain|hoon|kar|karke|doon|nahi|aap|aapke|aapko|aapka|bataiye|ho|gaya|gaye|ji|ke|ka|ki|ko|mein|main|par|yeh|"
    r"namaste|maaf|achha|zaroor|theek|haan|kaunsa|dhanyavaad|kuch|bol|raha|dono|naya|abhi|sakta|saka)\b",
    re.IGNORECASE,
)


def lines_of(script: Iterable[Step]) -> list[str]:
    found: list[str] = []
    for step in script:
        if isinstance(step, (Say, AskKey)):
            found.append(step.text)
        elif isinstance(step, SayWithPause):
            found += [step.first, step.second]
    return found


def looks_hindi(line: str) -> bool:
    return _HINDI_WORDS.search(line) is not None


class SavedRobotVoice:
    """Pieces for known lines. `pieces` is called during the call, so it only reads
    what `make` prepared; a line it doesn't know is a bug in the script."""

    def __init__(self, key: str, clips: SavedClips | None = None, opener: Callable[..., Any] = urllib.request.urlopen) -> None:
        self._key = key
        self._clips = clips or SavedClips(CLIPS)
        self._opener = opener
        self._made: dict[str, list[bytes]] = {}

    def make(self, lines: Iterable[str], sleep: Callable[[float], None] | None = None) -> None:
        for line in lines:
            if line in self._made:
                continue
            path = self._clips.path(SETUP.name, line)
            if not path.exists():
                # The chosen spelling for this voice (4 Nov 2026), like the real Talker's.
                language = "hinglish" if looks_hindi(line) else "english"
                request = VoiceLine("puppet", "everyday", "hi" if looks_hindi(line) else "en", spell_for_voice(line, language), "")
                kwargs = {"sleep": sleep} if sleep is not None else {}
                spoken = with_retries(lambda request=request: sarvam_speak(SETUP, request, self._key, self._opener), **kwargs)
                path.parent.mkdir(parents=True, exist_ok=True)
                write_sound(path, pcm16_to_float(spoken.pcm), PHONE_RATE)
            samples, _rate = read_sound(path)
            self._made[line] = [float_to_pcm16(piece) for piece in split_pieces(samples, PIECE_SAMPLES)]

    def __call__(self, text: str) -> list[bytes]:
        return self._made[text]
