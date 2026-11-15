"""The end-of-turn test on a real human voice (Part 2, decision 25).

The person says each sentence once, the way they really would, pausing where the page
shows. That one take gives both clips: cut at the pause it's unfinished, and whole it's
finished. Stopping for good would be wrong: someone who knows they are done lets their
voice fall, and it's the voice of a person with more to say that the models must recognise.

Takes are never refused for how the person paused (the user, 9 Nov 2026: real speech has
short pauses, and telling someone to pause longer makes it less real). A take with no
pause that can be cut is simply left out of the test.

Pauses are found by a real voice detector (Silero), the kind a live call uses. Two ways of
judging by loudness failed on a real room (9 Nov 2026): a fixed level couldn't see pauses
in 15 of 40 takes, because the room was above it, and a level worked out per take called
soft speech quiet and found 2.5 s "pauses" that weren't there. Silero found all 40.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile

from common.audio.noise import set_speech_level
from common.audio.phone_line import band_limit
from common.audio.resample import PHONE_RATE, resample
from lab.eot_test.clips import GUARD_S, MIN_PAUSE_S, MIN_SIDE_S, NOISE_FOLDER, SOUNDS, TRAIL_S, Clip, ClipPlan, clip_path, load_lines, on_the_phone

HUMAN_SOUNDS = SOUNDS.parent / "eot_human"
PAUSE_MARK = "   ⏸   "
DETECTOR_RATE = 16_000
DETECTOR_FRAME = 512  # samples, Silero's size at 16 kHz
DETECTOR_CONTEXT = 64  # samples of the last frame it needs in front of each new one
MIN_SPEECH_S = 0.08  # shorter is a click or a breath: left in, it would split a pause
Runs = list[tuple[float, float]]  # stretches of speech, (start, end) in seconds


@dataclass(frozen=True)
class HumanLine:
    id: str
    number: int
    text: str
    say: str  # what the recording page shows


def human_lines(lines: list[str] | None = None) -> list[HumanLine]:
    """Two of every three test sentences: 40 takes, about 5 minutes, with every kind of
    sentence still in it, because lines.yaml is in groups by kind."""
    all_lines = load_lines() if lines is None else lines
    return [HumanLine(f"L{n:02d}", n, text, text.replace("...", PAUSE_MARK))
            for n, text in enumerate(all_lines, 1) if n % 3 != 0]


def phone_voice(samples: np.ndarray, rate: int) -> np.ndarray:
    """A recording at the phone's rate, band and speech level, before the codec."""
    return np.asarray(band_limit(set_speech_level(resample(samples, rate, PHONE_RATE), PHONE_RATE)), dtype=np.float32)


def load_detector() -> Callable[[np.ndarray], Runs]:
    """Silero's voice detector: 16 kHz audio to its stretches of speech."""
    from huggingface_hub import hf_hub_download
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(hf_hub_download("onnx-community/silero-vad", "onnx/model.onnx"), options, providers=["CPUExecutionProvider"])
    frame_s = DETECTOR_FRAME / DETECTOR_RATE

    def detect(audio: np.ndarray) -> Runs:
        state = np.zeros((2, 1, 128), dtype=np.float32)
        context = np.zeros(DETECTOR_CONTEXT, dtype=np.float32)
        chances = []
        for at in range(0, len(audio) - DETECTOR_FRAME, DETECTOR_FRAME):
            piece = audio[at:at + DETECTOR_FRAME].astype(np.float32)
            chance, state = session.run(None, {"input": np.concatenate([context, piece])[None], "state": state,
                                               "sr": np.array(DETECTOR_RATE, dtype=np.int64)})
            chances.append(float(np.ravel(chance)[0]))
            context = piece[-DETECTOR_CONTEXT:]
        speaking = np.array(chances) > 0.5
        edges = np.flatnonzero(np.diff(np.concatenate([[0], speaking.astype(np.int8), [0]])))
        return [(float(a * frame_s), float(b * frame_s)) for a, b in zip(edges[::2], edges[1::2])]

    return detect


def find_human_pause(runs: Runs) -> tuple[float, float, float, float] | None:
    """(first sound, pause start, pause end, last sound) in seconds, or None. The longest
    pause: in 36 of 40 real takes it was at the page's mark, and any place a person
    stopped and went on is truly unfinished."""
    kept = [run for run in runs if run[1] - run[0] >= MIN_SPEECH_S]
    if len(kept) < 2:
        return None
    first, last = kept[0][0], kept[-1][1]
    pauses = [(left[1], right[0]) for left, right in zip(kept, kept[1:])
              if right[0] - left[1] >= MIN_PAUSE_S and min(left[1] - first, last - right[0]) >= MIN_SIDE_S]
    if not pauses:
        return None
    start, end = max(pauses, key=lambda pause: pause[1] - pause[0])
    return first, start, end, last


def followed_by_quiet(speech: np.ndarray, real_quiet: np.ndarray, room: np.ndarray, length: int) -> np.ndarray:
    """Speech, then exactly `length` samples of quiet: what really followed it, then more
    of the same room. A natural pause can be shorter than the moments the test asks at,
    and digital silence there would tell a model which clips were cut."""
    quiet = real_quiet[:length]
    missing = length - len(quiet)
    if missing > 0:
        quiet = np.concatenate([quiet, np.resize(room, missing)])
    return np.concatenate([speech, quiet])


def build_human_pair(voice: np.ndarray, pause: tuple[float, float, float, float], rate: int = PHONE_RATE) -> tuple[np.ndarray, np.ndarray, float, float, float] | None:
    """(unfinished, finished, where speech ends in each, the real pause in seconds)."""
    first, start, end, last = (int(point * rate) for point in pause)
    guard = int(rate * GUARD_S)
    trail = int(rate * TRAIL_S)
    cut, stop = min(start + guard, len(voice)), min(last + guard, len(voice))
    pause_quiet = voice[cut:max(cut, end - guard)]
    end_quiet = voice[stop:]
    room = np.concatenate([voice[:max(0, first - guard)], end_quiet, pause_quiet])
    if len(room) == 0:
        return None
    return (followed_by_quiet(voice[:cut], pause_quiet, room, trail), followed_by_quiet(voice[:stop], end_quiet, room, trail),
            cut / rate, stop / rate, (end - start) / rate)


def prepare_human(speaker: str = "s1", sounds: Path = HUMAN_SOUNDS, noise_folder: Path = NOISE_FOLDER,
                  log: Callable[[str], None] = print, detect: Callable[[np.ndarray], Runs] | None = None) -> list[Clip]:
    """Each saved take into the same two files and manifest the machine clips have, so
    the same test reads both. Every clip counts as approved: the label is what the person
    really did, so there's nothing left to judge by ear."""
    (sounds / "clips").mkdir(parents=True, exist_ok=True)
    detector = detect or load_detector()
    made: list[Clip] = []
    for line in human_lines():
        path = sounds / "raw" / speaker / f"{line.id}.wav"
        if not path.exists():
            continue
        samples, rate = soundfile.read(path, dtype="float32", always_2d=True)
        mono = samples.mean(axis=1)
        pause = find_human_pause(detector(resample(mono, rate, DETECTOR_RATE)))
        pair = None if pause is None else build_human_pair(phone_voice(mono, rate), pause)
        if pair is None:
            log(f"{line.id}: no pause to cut at: left out")
            continue
        unfinished, finished, unfinished_end, finished_end, pause_s = pair
        plan = ClipPlan(f"{line.id}_{speaker}", line.number, speaker, line.text, None, None, round(pause_s, 2), 0)
        for kind, audio in (("unfinished", unfinished), ("finished", finished)):
            soundfile.write(clip_path(sounds, plan.id, kind), on_the_phone(audio, plan, noise_folder), PHONE_RATE, subtype="PCM_16")
        made.append(Clip(plan, PHONE_RATE, round(unfinished_end, 3), round(finished_end, 3)))
    (sounds / "clips.json").write_text(json.dumps([asdict(clip) for clip in made], ensure_ascii=False, indent=1), encoding="utf-8")
    (sounds / "verdicts.json").write_text(json.dumps({clip.plan.id: "ok" for clip in made}, indent=1), encoding="utf-8")
    return made
