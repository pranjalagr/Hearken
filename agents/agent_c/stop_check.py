"""Stop or keep talking (decision 26): the haan-ruko model on the caller's sound while
the robot talks. Agent C's only stop-or-keep check: no word list, no word count.

Agent A stopped on any caller sound, so a "haan haan" while it spoke cut its own
yes-question off (practice run 7, 15 Nov 2026). Here the voice detector never
interrupts by itself: the model looks at the last 2 s of caller sound every 0.1 s
while the robot talks, and only its "stop" (above the line) stops the voice, bumps the
job version and pauses any ticket being confirmed. After a stop it waits for the
caller's turn to finish before it can stop again.

One check is well under 20 ms on a CPU (Part 4), so it runs on the event loop itself,
between two 20 ms pieces; a thread would cost more than the check.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Protocol

import numpy as np

from common.wire.protocol import PIECE_MS


class StopModel(Protocol):
    def chance(self, audio: np.ndarray) -> float:
        """The chance this is real speech, so stop: audio is float32 in [-1, 1], 8 kHz."""
        ...


class OnnxStopModel:
    """The saved Part 4 model (lab/haan_ruko/), run as the lab ran it: one thread,
    log-mel features from lab/haan_ruko/sound.py, so live and test hear the same."""

    def __init__(self, path: Path) -> None:
        import onnxruntime as ort

        from lab.haan_ruko.sound import FLOOR, FRAMES, log_mel

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        self._log_mel = log_mel
        self._frames = FRAMES
        self._quiet = float(np.log(FLOOR))  # what silence looks like to the model

    def chance(self, audio: np.ndarray) -> float:
        features = self._log_mel(audio)[:, -self._frames:]
        if features.shape[1] < self._frames:  # the call's first 2 s: pad the start with quiet
            quiet = np.full((features.shape[0], self._frames - features.shape[1]), self._quiet, dtype=np.float32)
            features = np.concatenate([quiet, features], axis=1)
        [logit] = self._session.run(None, {"log_mel": features[None, None].astype(np.float32)})[0]
        return float(1.0 / (1.0 + np.exp(-logit)))


class StopListener:
    def __init__(self, model: StopModel, line: float, window_s: float, every_s: float,
                 on_stop: Callable[[float], Awaitable[None]], on_slow: Callable[[float], None] = lambda ms: None) -> None:
        self._model = model
        self._line = line
        self._pieces: deque[bytes] = deque(maxlen=round(window_s * 1000 / PIECE_MS))
        self._every = max(1, round(every_s * 1000 / PIECE_MS))
        self._on_stop = on_stop
        self._on_slow = on_slow
        self._count = 0
        self.stopped = False  # said stop; quiet until the caller's turn finishes

    async def heard(self, piece: bytes, robot_talking: bool) -> None:
        """Every caller piece, in order."""
        self._pieces.append(piece)
        self._count += 1
        if not robot_talking or self.stopped or self._count % self._every:
            return
        started = time.perf_counter()
        chance = self._model.chance(self._audio())
        took_ms = (time.perf_counter() - started) * 1000
        if took_ms > PIECE_MS:
            self._on_slow(took_ms)  # one check must fit between two pieces (Part 4)
        if chance >= self._line:
            self.stopped = True
            await self._on_stop(chance)

    def turn_finished(self) -> None:
        self.stopped = False

    def _audio(self) -> np.ndarray:
        samples = np.frombuffer(b"".join(self._pieces), dtype="<i2")
        return samples.astype(np.float32) / np.float32(32768)
