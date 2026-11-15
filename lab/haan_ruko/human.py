"""The haan vs ruko test on a real voice (Part 4, step 4).

The person records what they'd say while the robot talks. These takes are never trained on:
they only test, so the score says how the model does on a real person, not on the machine
voices it learned from (machine voices flattered the end-of-turn models: 97% against 60%).
"""

from __future__ import annotations

import io
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile

from common.audio.resample import PHONE_RATE, resample
from lab.haan_ruko.clips import SOUNDS
from lab.haan_ruko.phrases import KEEP, MIXED, STOP, _short_id, load_phrases
from lab.haan_ruko.speech import Timing, time_clip

HUMAN_SOUNDS = SOUNDS.parent / "haan_ruko_human"
# How many of each to record: every held-back phrase, plus some the model trained on (in
# machine voices), so the test shows both a new voice and new phrases. About 100 takes.
FROM_TRAINING = {KEEP: 22, STOP: 17, MIXED: 10}
EVERYDAY_SENTENCES = 20  # plain caller sentences, from the end-of-turn lines


@dataclass(frozen=True)
class Take:
    id: str
    group: str
    answer: str
    text: str
    say: str  # what the recording page shows
    held_back: bool


def human_takes(data: dict | None = None, sentences: list[str] | None = None) -> list[Take]:
    data = data or load_phrases()
    takes: list[Take] = []
    for group in (KEEP, STOP, MIXED):
        chosen = [(True, t) for t in data[group]["held_back"]] + [(False, t) for t in data[group]["train"][: FROM_TRAINING[group]]]
        takes += [Take(f"{group}-{_short_id(text)}", group, group, text, text, held) for held, text in chosen]
    if sentences is None:
        from lab.eot_test.clips import load_lines

        sentences = [line.replace("...", "") for line in load_lines()]
    for text in sentences[::max(1, len(sentences) // EVERYDAY_SENTENCES)][:EVERYDAY_SENTENCES]:
        takes.append(Take(f"sentence-{_short_id(text)}", "sentence", STOP, text, text, False))
    return takes


QUIETEST_PEAK = 0.01  # a muted microphone; anything above it is a real, if soft, take


def check_take(data: bytes) -> None:
    """The recording page's usual checks, but a soft take is fine: a quiet "hmm" is exactly
    what a listening sound is, so the ears test's 0.05 loudness rule would refuse real ones."""
    from lab.ears_test.record_server import MAX_REPEATED, MAX_S, MIN_S, BadRecording, repeated_share

    try:
        samples, rate = soundfile.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception as error:
        raise BadRecording("that isn't a readable sound file") from error
    if repeated_share(samples) > MAX_REPEATED:
        raise BadRecording("the recorder doubled the sound: please refresh this page and record it again")
    seconds = len(samples) / rate
    if not MIN_S <= seconds <= MAX_S:
        raise BadRecording(f"it lasted {seconds:.1f} s; a take must last {MIN_S} to {MAX_S:.0f} s")
    if float(np.abs(samples).max()) < QUIETEST_PEAK:
        raise BadRecording("nothing was heard: check the microphone is on and not muted")


def prepare_human(speaker: str, detect: Callable[[np.ndarray], list[tuple[float, float]]], sounds: Path = HUMAN_SOUNDS,
                  log: Callable[[str], None] = print) -> list[tuple[dict, Timing, np.ndarray]]:
    """Each saved take at phone rate, with its speech timed the same way as the machine clips."""
    from lab.eot_test.human import DETECTOR_RATE

    data = load_phrases()
    listening = data[KEEP]["train"] + data[KEEP]["held_back"]
    ready, timings = [], []
    for take in human_takes(data):
        path = sounds / "raw" / speaker / f"{take.id}.wav"
        if not path.exists():
            continue
        samples, rate = soundfile.read(io.BytesIO(path.read_bytes()), dtype="float32", always_2d=True)
        mono = samples.mean(axis=1)
        timing = time_clip(f"{take.id}_{speaker}", take.answer, take.text, detect(resample(mono, rate, DETECTOR_RATE)), listening)
        timings.append(asdict(timing))
        if timing.why_left_out:
            log(f"{take.id}: {timing.why_left_out}: left out")
            continue
        clip = {**asdict(take), "id": timing.id, "voice": speaker, "file": str(path)}
        ready.append((clip, timing, resample(mono, rate, PHONE_RATE)))
    sounds.mkdir(parents=True, exist_ok=True)
    (sounds / "timings.json").write_text(json.dumps(timings, indent=1), encoding="utf-8")
    return ready
