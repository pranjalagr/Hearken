"""Lines code plays itself: from a clip made once in the robot's own voice, or, without
one, through the live voice (decision 11).

A clip plays at once: no voice request, no ~1 s wait for its first sound. The clips
live outside git (sound files never go in git), in one folder per voice, and are made by
`python -m cli.agent_c_lines make`. A fresh copy of the project has none, so every line
still works through the live voice, and the diary says so once per call.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path

import numpy as np
from pipecat.frames.frames import Frame, TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame, TTSTextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from agents.agent_c.talker.voice_spelling import spell_for_voice
from agents.common.diary import Diary
from common.audio.pcm import float_to_pcm16
from common.audio.sound_files import read_sound
from common.wire.protocol import SAMPLE_RATE
from engine.contract.spoken import LANGUAGES, SpokenWords

# The lines made as clips: every line with no {slot} in these groups, the fixed lines
# decided on 17 Nov 2026 (fixed where the words must be exact or instant). A line with a
# number ("4521, dekh raha hoon") can't be made ahead, so it always goes live, as do the
# Guard's yes-question and every AI line.
CLIP_GROUPS = ("waiting", "silent", "talk", "ladder", "result", "line")
GREETING = ("greeting", "greeting")  # the settings' first line (decision 13): a clip too


def clip_name(group: str, value: str, language: str, text: str) -> str:
    """The file a line's clip is kept in. A short hash of the words is part of it, so a
    line reworded in words.yaml finds no clip (and goes live) instead of playing the old words."""
    return f"{group}.{value}.{language}.{hashlib.sha256(text.encode('utf-8')).hexdigest()[:8]}.wav"


def clip_lines(words: SpokenWords, greeting: str = "") -> list[tuple[str, str, str, str]]:
    """(group, value, language, text) for every line that is made as a clip. The greeting
    is said in Hinglish, as every robot in the lab opens."""
    lines = [(*GREETING, "hinglish", spell_for_voice(greeting, "hinglish"))] if greeting else []
    for group in CLIP_GROUPS:
        for value, phrases in words.groups.get(group, {}).items():
            for language in LANGUAGES:
                text = phrases[language]
                if "{" not in text:
                    # The clip says what the live voice would: the same spelling fixes.
                    lines.append((group, value, language, spell_for_voice(text, language)))
    return lines


class Clips:
    """The clips found on disk, as 16-bit 8 kHz sound ready for the wire."""

    def __init__(self, sounds: dict[str, bytes]) -> None:
        self._sounds = sounds

    @classmethod
    def load(cls, folder: Path, names: Iterable[str], diary: Diary) -> Clips:
        """Read the clips once, when the robot starts, never during a call. A clip at the
        wrong rate is skipped: played as 8 kHz it would sound sped up or slowed down."""
        sounds: dict[str, bytes] = {}
        missing = []
        for name in names:
            path = folder / name
            if not path.exists():
                missing.append(name)
                continue
            samples, rate = read_sound(path)
            if rate != SAMPLE_RATE or not len(samples):
                diary.error(f"clip {path} is {rate} Hz with {len(samples)} samples, not {SAMPLE_RATE} Hz: using the live voice")
                continue
            sounds[name] = float_to_pcm16(np.asarray(samples, dtype=np.float32))
        if missing:
            diary.error(f"no clips for {len(missing)} lines in {folder}: those go to the live voice "
                        "(make them with python -m cli.agent_c_lines make)")
        return cls(sounds)

    def get(self, name: str) -> bytes | None:
        return self._sounds.get(name)

    def __len__(self) -> int:
        return len(self._sounds)


class ClipOut(FrameProcessor):
    """Sits right after the voice and puts a clip on the way to the line. After the
    voice, not at the pipeline's start: the ears read every sound frame that passes
    them, and must never hear the robot's own clip as the caller."""

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        # Pipecat's base process_frame forwards nothing: a processor that doesn't push
        # every frame swallows the StartFrame and the call never starts (the StopSwitch
        # did exactly that on the first real call, 16 Nov 2026).
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

    async def play(self, sound: bytes, text: str) -> None:
        """The clip, sent exactly as a voice service sends a line: started, its sound as
        voice audio, its words, stopped. The output reports robot speech only for voice
        audio, so plain audio frames never counted as the robot speaking at all: the Guard's
        mark, the goodbye's end and "say the result again if cut" never saw a clip line
        (17 Nov 2026, food_08: the clip "Haan ya nahi?" sent no mark, and every later
        "haan" counted as early)."""
        context = f"clip-{uuid.uuid4().hex[:8]}"
        await self.push_frame(TTSStartedFrame(context_id=context))
        await self.push_frame(TTSAudioRawFrame(sound, SAMPLE_RATE, 1, context_id=context))
        await self.push_frame(TTSTextFrame(text, "sentence", context_id=context))
        await self.push_frame(TTSStoppedFrame(context_id=context))


def clips_by_words(clips: Clips, words: SpokenWords, greeting: str = "") -> dict[str, bytes]:
    """Each fixed line's clip, found by the words the voice would say (after the spelling
    fixes), so the robot's say() can play a clip for exactly that line."""
    found = {}
    for group, value, language, text in clip_lines(words, greeting):
        sound = clips.get(clip_name(group, value, language, text))
        if sound is not None:
            found[text] = sound
    return found
