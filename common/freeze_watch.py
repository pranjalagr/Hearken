"""Finds the moments the event loop stalls for over 100 ms, and says why (decision 14).

Every wait must be async: one blocking call would freeze every call sharing the
program. Used by the tests (a freeze of ours fails the test) and by the lab
runner (every stall is written into the run's folder, so "the lab fell behind"
comes with its cause instead of a guess; moved here from tests/conftest.py on
15 Nov 2026, when long Agent A calls kept stalling by 150-250 ms).

How it is measured:
- a heartbeat task ticks every 10 ms on the event loop;
- a watcher thread checks it every 5 ms; when the heartbeat is clearly late, it
  looks at what the main thread is doing. Waiting for events: the OS woke the
  process late (a busy or paging laptop), not a freeze of ours. Running code: a
  freeze, recorded with exactly which code was running.

Not asyncio's debug mode: it records a stack trace for every small step, and
under bursts of audio messages that bookkeeping alone took over 100 ms, so it
raised false alarms and slowed the code it was measuring.

Garbage collection: a process holding ~165,000 objects took 50 ms for one full
collection, enough to push a heartbeat past 100 ms. So, as a low-delay service
does after start-up, the watch parks every existing object first (gc.freeze):
collections afterwards only scan what was made since.
"""

from __future__ import annotations

import asyncio
import gc
import sys
import threading
import time
from dataclasses import dataclass, field

FREEZE_LIMIT_S = 0.1  # decision 14
HEARTBEAT_S = 0.01
# Sampling starts at 50 ms late: on Windows a 10 ms sleep often wakes after 16 or
# 31 ms (the 15.6 ms system timer), which is normal.
SAMPLE_FROM_S = 0.05


@dataclass
class Stall:
    at_s: float  # seconds after the watch started
    wall: str  # clock time, to match against call folders
    late_ms: float  # how late the heartbeat got, at its worst
    ours: bool  # our code (or a library it called) was running: a freeze we caused
    where: list[str] = field(default_factory=list)  # the first sample of what ran, when ours


def in_event_loop_plumbing(frame) -> bool:
    """True if the main thread is inside the event loop's own machinery: waiting for
    events, or in the OS socket calls asyncio makes for itself (once a single WSARecv
    on asyncio's wake-up socket took over 100 ms, in a test that opened no sockets).

    A late heartbeat while there means the operating system was slow, not our code.
    Socket and selector frames are looked through to see who called them, so a
    blocking socket call in our own code still counts."""
    while frame is not None:
        path = frame.f_code.co_filename.replace("\\", "/")
        if "/asyncio/" in path:
            return True
        if not path.endswith(("/socket.py", "/selectors.py")):
            return False
        frame = frame.f_back
    return False


def where(frame, depth: int = 6) -> list[str]:
    """The innermost frames as "file:line in function", without touching the disk (no
    source lines): formatting a stack reads source files while holding Python's lock,
    which would make the main thread wait for the watcher and add the very stall it
    measures."""
    lines = []
    while frame is not None and len(lines) < depth:
        lines.append(f"  {frame.f_code.co_filename}:{frame.f_lineno} in {frame.f_code.co_name}")
        frame = frame.f_back
    return lines


class FreezeWatch:
    def __init__(self, limit_s: float = FREEZE_LIMIT_S) -> None:
        self.stalls: list[Stall] = []
        self._limit_s = limit_s
        self._main_thread = threading.get_ident()
        self._started = time.perf_counter()
        self._last_beat = self._started
        self._stop = threading.Event()
        self._beat_task: asyncio.Task | None = None
        self._watcher = threading.Thread(target=self._watch, name="freeze-watcher", daemon=True)

    @property
    def freezes(self) -> list[str]:
        """The stalls our own code caused, each with the code that was running."""
        return [f"blocked over {self._limit_s * 1000:.0f} ms in:\n" + "\n".join(stall.where) for stall in self.stalls if stall.ours]

    def start(self) -> None:
        gc.collect()  # before the heartbeat starts, so its own time is never counted
        gc.freeze()
        self._started = time.perf_counter()
        self._last_beat = self._started
        self._beat_task = asyncio.create_task(self._heartbeat())
        self._watcher.start()

    async def stop(self) -> list[str]:
        self._stop.set()
        self._watcher.join()
        self._beat_task.cancel()
        await asyncio.gather(self._beat_task, return_exceptions=True)
        gc.unfreeze()
        return self.freezes

    async def _heartbeat(self) -> None:
        while True:
            self._last_beat = time.perf_counter()
            await asyncio.sleep(HEARTBEAT_S)

    def _watch(self) -> None:
        # Every 5 ms while the heartbeat is clearly late, note what the main thread
        # runs. Once it is over the limit, one stall is recorded for that late stretch;
        # its lateness is updated until the next beat, so it shows the worst.
        current_beat, samples, stall = None, [], None
        while not self._stop.wait(0.005):
            beat = self._last_beat
            if beat != current_beat:
                current_beat, samples, stall = beat, [], None
            late = time.perf_counter() - beat - HEARTBEAT_S
            if late <= SAMPLE_FROM_S:
                continue
            frame = sys._current_frames().get(self._main_thread)
            if frame is not None and not in_event_loop_plumbing(frame):
                samples.append(where(frame))
            if late <= self._limit_s:
                continue
            if stall is None:
                stall = Stall(round(beat - self._started, 2), time.strftime("%H:%M:%S"), round(late * 1000, 1), bool(samples),
                              samples[0] if samples else [])
                self.stalls.append(stall)
            else:
                stall.late_ms = round(late * 1000, 1)
                if samples and not stall.ours:
                    stall.ours, stall.where = True, samples[0]
