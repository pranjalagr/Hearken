"""The call box: everything one call runs lives here, and dies with the call
(Part 2 decision 14).

The worry it answers: a phone call drops, but a background worker (the
Thinker) keeps running forever, spending money and memory, or even acts for a
caller who is gone. So:

- **One box per call.** Every task of the call is started through the box.
  When the call ends, for any reason, the box cancels all of them.
- **Ending.** The robot or the line ends it (hang-up), or the box itself does:
  no caller audio for 30 s (a connection can die without saying so), or the
  15-minute cap.
- **A change already sent is shielded.** Cancelling would only stop *our*
  waiting; the backend may still save it. So the box waits for its answer, up
  to 10 s, to record the truth: finished, or "result unknown".
- **Nothing new starts once the box is closing.**
- **Crash-safe tasks** (the Thinker) report a crash to `on_crash` instead of
  killing the call, so the Talker can apologise and hand over. A crash in any
  other task ends the call: a broken ear or wire can't be talked around.

It works like an asyncio.TaskGroup, written out by hand because a TaskGroup
cancels every child on the first error, and it has no way to keep a shielded
change running or to let a crash-safe task fail alone.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

NO_AUDIO = "no_audio"
TIME_CAP = "time_cap"
CRASH = "crash"
DONE = "done"  # the main task returned without naming a reason


class BoxClosing(RuntimeError):
    """The call is ending: no new work may start."""


class _Unknown:
    def __repr__(self) -> str:
        return "UNKNOWN"


# What a shielded change returns when its answer never came in time.
UNKNOWN = _Unknown()


@dataclass(frozen=True)
class CallBoxSettings:
    # Starting guesses from decisions 14 and 30; tuned from real calls.
    no_audio_s: float = 30.0
    max_call_s: float = 15 * 60.0
    change_wait_s: float = 10.0


@dataclass
class CallResult:
    reason: str
    finished_changes: list[str] = field(default_factory=list)
    unknown_changes: list[str] = field(default_factory=list)  # "result unknown" in the black box
    crashes: list[tuple[str, str]] = field(default_factory=list)  # crash-safe tasks that crashed
    error: BaseException | None = None  # the crash that ended the call, if one did


class CallBox:
    def __init__(
        self,
        settings: CallBoxSettings = CallBoxSettings(),
        on_crash: Callable[[str, BaseException], None] | None = None,
    ) -> None:
        self._settings = settings
        self._on_crash = on_crash
        self._tasks: set[asyncio.Task[Any]] = set()
        self._changes: set[asyncio.Task[Any]] = set()
        self._ended = asyncio.Event()
        self._closing = False
        self._result = CallResult(reason="")
        self._last_audio = 0.0

    @property
    def closing(self) -> bool:
        return self._closing or self._ended.is_set()

    async def run(self, main: Callable[[CallBox], Awaitable[None]]) -> CallResult:
        """Run one call. Returns once every task of it is finished or cancelled."""
        loop = asyncio.get_running_loop()
        self._last_audio = loop.time()
        self._spawn("watchdog", self._watch(loop.time()), crash_safe=False)
        self._spawn("main", self._main(main), crash_safe=False)
        await self._ended.wait()
        await self._close()
        return self._result

    def start(self, name: str, work: Coroutine[Any, Any, Any], *, crash_safe: bool = False) -> asyncio.Task[Any] | None:
        """Start a task of this call. Returns None, without running it, if the
        call is already ending."""
        if self.closing:
            work.close()  # never started: close it so Python doesn't warn it was never awaited
            return None
        return self._spawn(name, work, crash_safe)

    async def change(self, name: str, work: Coroutine[Any, Any, Any]) -> Any:
        """Run a change already sent to the backend (a confirm), shielded.

        Returns its answer, or UNKNOWN if none came within change_wait_s. Even
        if the task awaiting it is cancelled because the call ended, the change
        itself keeps going, and the box waits for it before closing.
        """
        if self.closing:
            work.close()
            raise BoxClosing(f"the call is ending; {name} was not sent")
        task = asyncio.create_task(self._bounded_change(name, work), name=f"change:{name}")
        self._changes.add(task)
        return await asyncio.shield(task)

    def heard_audio(self) -> None:
        """The caller's audio arrived; called for every piece."""
        self._last_audio = asyncio.get_running_loop().time()

    def end(self, reason: str) -> None:
        """End the call. The first reason given is the one kept."""
        if not self._ended.is_set():
            self._result.reason = reason
            self._ended.set()

    # ---- inside the box -------------------------------------------------------

    def _spawn(self, name: str, work: Coroutine[Any, Any, Any], crash_safe: bool) -> asyncio.Task[Any]:
        task = asyncio.create_task(self._guarded(name, work, crash_safe), name=name)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def _guarded(self, name: str, work: Coroutine[Any, Any, Any], crash_safe: bool) -> Any:
        try:
            return await work
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not crash_safe:
                if self._result.error is None:
                    self._result.error = exc
                self.end(CRASH)
                return None
            self._result.crashes.append((name, repr(exc)))
            if self._on_crash is not None:
                try:
                    self._on_crash(name, exc)
                except Exception as handler_exc:
                    # The crash handler is how the caller hears an apology; if
                    # even that breaks, the call can't go on safely.
                    self._result.error = handler_exc
                    self.end(CRASH)
            return None

    async def _main(self, main: Callable[[CallBox], Awaitable[None]]) -> None:
        await main(self)
        self.end(DONE)

    async def _watch(self, started: float) -> None:
        """Ends the call on the time cap, or when the caller's audio stopped."""
        loop = asyncio.get_running_loop()
        cap = started + self._settings.max_call_s
        while True:
            now = loop.time()
            if now >= cap:
                self.end(TIME_CAP)
                return
            quiet_until = self._last_audio + self._settings.no_audio_s
            if now >= quiet_until:
                self.end(NO_AUDIO)
                return
            await asyncio.sleep(min(cap, quiet_until) - now)

    async def _bounded_change(self, name: str, work: Coroutine[Any, Any, Any]) -> Any:
        try:
            answer = await asyncio.wait_for(work, self._settings.change_wait_s)
        except TimeoutError:
            # We stopped waiting; the backend's all-or-nothing save went on
            # without us. The black box records that we don't know.
            self._result.unknown_changes.append(name)
            return UNKNOWN
        except Exception:
            # The line to the backend broke mid-change: the save may or may
            # not have happened, so this is just as unknown. The error still
            # goes to whoever is waiting, so it can retry (decision 19).
            self._result.unknown_changes.append(name)
            raise
        self._result.finished_changes.append(name)
        return answer

    async def _close(self) -> None:
        self._closing = True
        current = asyncio.current_task()
        tasks = [task for task in self._tasks if task is not current]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        # Shielded changes were not cancelled; each is bounded by its own limit.
        await asyncio.gather(*self._changes, return_exceptions=True)
