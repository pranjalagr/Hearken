"""The haan vs ruko model's training windows, made on every core of the CPU (Part 4, step 6).

Each round every training clip goes on a new random rough line and becomes a few windows of
the sound picture. Once training moved to the GPU (11 Nov 2026), this was most of a round:
about 95 s for 11,800 clips, against 12 s of training. Python threads barely helped (2.8 s
to 2.4 s for 600 clips): the work is many small numpy steps, and threads take turns at them.
So helper processes do it, about twice as fast (2,000 clips: 15.7 s alone, 8.7 s with 4
helpers; each core runs slower when all four work, 0.95 s a chunk against 0.65 s). Clips go
to them a hundred at a time, so no process keeps a second copy of all the sound: this laptop
has 7.8 GB, and under 2 GB was free.

Kept apart from train.py, and free of torch, so a helper starts light.
"""

from __future__ import annotations

import multiprocessing
import os
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import numpy as np

from lab.haan_ruko.score import answer_at, check_times
from lab.haan_ruko.sound import FRAMES, LEAD_S, N_MELS, RATE, Noises, log_mel, on_a_rough_line, random_roughness, window
from lab.haan_ruko.speech import Timing

JITTER_S = 0.04  # training checks move a little, so the model doesn't learn the exact 0.1 s grid
NOISE_ONLY_SHARE = 0.1  # windows of background with nobody speaking, answer keep
NOISE_ONLY_S = 1.2
# Real pieces are many and each has up to 18 checks: a few random ones per clip each round
# keep a round to minutes, and a new round picks different moments.
CHECKS_PER_CLIP = 4
HELPERS = 4  # this laptop's 4 cores (i5-11300H): 6 or 8 helpers were no faster (9.5 s, 8.6 s against 8.9 s)
CHUNK = 100  # clips sent to a helper at once: a few MB each way, and enough work to be worth sending

Item = tuple[dict, Timing, np.ndarray]


def clip_windows(item: Item, seed: int, noises: Noises, checks_per_clip: int | None) -> tuple[list[np.ndarray], list[int]]:
    """One clip on its own random rough line, as windows and their right answers. Each clip has
    its own seed, so the result is the same whichever process makes it."""
    clip, timing, audio = item
    rng = np.random.default_rng(seed)
    line, speed = on_a_rough_line(audio, random_roughness(rng), noises, dither_seed=int(rng.integers(0, 10_000)))
    features = log_mel(line)
    times = check_times(timing)
    if checks_per_clip is not None and len(times) > checks_per_clip:
        times = [times[i] for i in sorted(rng.choice(len(times), checks_per_clip, replace=False))]
    xs, ys = [], []
    for t in times:
        t = t + float(rng.uniform(-JITTER_S, JITTER_S))
        answer = answer_at(clip["answer"], timing, t)
        if answer is None:
            continue
        try:
            xs.append(window(features, LEAD_S + t * speed))
        except ValueError:
            continue
        ys.append(answer)
    return xs, ys


def noise_window(seed: int, noises: Noises) -> np.ndarray:
    """Background alone, nobody speaking: the right answer is keep."""
    rng = np.random.default_rng(seed)
    line, _ = on_a_rough_line(np.zeros(int(NOISE_ONLY_S * RATE), np.float32), random_roughness(rng), noises,
                              dither_seed=int(rng.integers(0, 10_000)))
    return window(log_mel(line), LEAD_S + float(rng.uniform(0.3, NOISE_ONLY_S)))


def _clip_chunk(items: list[Item], seeds: np.ndarray, noises: Noises, checks_per_clip: int | None) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for item, seed in zip(items, seeds):
        x, y = clip_windows(item, int(seed), noises, checks_per_clip)
        xs += x
        ys += y
    return (np.stack(xs) if xs else np.zeros((0, N_MELS, FRAMES), np.float32)), np.array(ys, dtype=np.float32)


def _noise_chunk(seeds: np.ndarray, noises: Noises) -> np.ndarray:
    return np.stack([noise_window(int(seed), noises) for seed in seeds])


# --- the helper processes ---------------------------------------------------------------

_HELPER_NOISES: Noises | None = None  # each helper's own copy, loaded once when it starts
# The thread counts numpy's maths libraries read when they start (OpenBLAS, MKL, OpenMP).
_ONE_THREAD = ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")


def _start_helper(noise_folder: Path) -> None:
    global _HELPER_NOISES
    _HELPER_NOISES = Noises(noise_folder)


def _helper_clip_chunk(task: tuple[list[Item], np.ndarray, int | None]) -> tuple[np.ndarray, np.ndarray]:
    items, seeds, checks_per_clip = task
    return _clip_chunk(items, seeds, _HELPER_NOISES, checks_per_clip)


def _helper_noise_chunk(seeds: np.ndarray) -> np.ndarray:
    return _noise_chunk(seeds, _HELPER_NOISES)


class Helpers:
    """Processes that make windows, started once for a whole training run (starting one takes
    about 2 s). Each loads its own copy of the noise recordings from `noise_folder`, which must
    be the folder the in-process `Noises` came from."""

    def __init__(self, noise_folder: Path, count: int = HELPERS) -> None:
        # One maths thread per helper: with numpy's default each helper starts 8, so 4 helpers
        # would run 32 on this CPU's 8. One was even a little faster alone (1,000 clips: 6.5 s
        # against 7.2 s, 11 Nov 2026). Read when a helper imports numpy, so it is set before
        # they start, and put back after: this process's numpy is already running.
        before = {name: os.environ.get(name) for name in _ONE_THREAD}
        os.environ.update({name: "1" for name in _ONE_THREAD})
        try:
            # "spawn" is the only way Windows starts processes; named here so Linux behaves the same.
            self._pool = multiprocessing.get_context("spawn").Pool(count, initializer=_start_helper, initargs=(noise_folder,))
        finally:
            for name, value in before.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def map(self, work: Callable, tasks: Iterable) -> Iterator:
        # In order, and fed as the helpers ask for more, so only a few chunks wait in memory.
        return self._pool.imap(work, tasks)

    def close(self) -> None:
        self._pool.close()
        self._pool.join()

    def __enter__(self) -> Helpers:
        return self

    def __exit__(self, *_) -> None:
        self.close()


# --- one round's windows ----------------------------------------------------------------


def _most_windows(train: list[Item], checks_per_clip: int | None) -> int:
    """At least as many windows as a round can make, so they are written straight into one
    array, not gathered in a list and then copied: a round's windows are about 1.4 GB."""
    clips = sum(len(check_times(timing)) if checks_per_clip is None else min(checks_per_clip, len(check_times(timing)))
                for _, timing, _ in train)
    return clips + int(clips * NOISE_ONLY_SHARE)


def training_windows(train: list[Item], noises: Noises, rng: np.random.Generator,
                     checks_per_clip: int | None = CHECKS_PER_CLIP, helpers: Helpers | None = None) -> tuple[np.ndarray, np.ndarray]:
    """One round's windows and right answers. With `helpers`, they make them (using their own
    copy of the same noises); without, this process does, as the tests do."""
    seeds = rng.integers(0, np.iinfo(np.int64).max, len(train))
    tasks = [(train[at:at + CHUNK], seeds[at:at + CHUNK], checks_per_clip) for at in range(0, len(train), CHUNK)]
    chunks = helpers.map(_helper_clip_chunk, tasks) if helpers else (_clip_chunk(*task[:2], noises, task[2]) for task in tasks)
    x = np.empty((_most_windows(train, checks_per_clip), N_MELS, FRAMES), np.float32)
    y = np.empty(len(x), np.float32)
    count = 0
    for xs, ys in chunks:
        x[count:count + len(xs)], y[count:count + len(ys)] = xs, ys
        count += len(xs)
    noise_seeds = rng.integers(0, np.iinfo(np.int64).max, int(count * NOISE_ONLY_SHARE))
    noise_tasks = [noise_seeds[at:at + CHUNK] for at in range(0, len(noise_seeds), CHUNK)]
    for xs in (helpers.map(_helper_noise_chunk, noise_tasks) if helpers else (_noise_chunk(task, noises) for task in noise_tasks)):
        x[count:count + len(xs)] = xs
        y[count:count + len(xs)] = 0
        count += len(xs)
    return x[:count], y[:count]
