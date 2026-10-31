"""The stopwatch: a time stamp on every step of a call (Part 2 decision 31).

Each record says what happened and when, on two clocks:
- `at_ns`: time.perf_counter_ns, for exact durations. On Windows it is the
  system-wide performance counter, so the robot's and the caller's processes
  on one machine read the same clock and their records line up.
- `wall_ns`: time.time_ns, for reading a record as a date.

Records go to a sink: in memory for tests, or one JSON line each in a file
(the call's events.jsonl, Part 3 C16). The stopwatch never decides anything
and never slows a call down: writing one record is a dict and a list append,
or one line to a file.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol


class Sink(Protocol):
    def write(self, record: dict[str, Any]) -> None: ...


class MemorySink:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def write(self, record: dict[str, Any]) -> None:
        self.records.append(record)


class JsonlSink:
    """One JSON object per line, flushed at once, so a crashed call still
    leaves every step it reached on disk."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("a", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        self._file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class Stopwatch:
    def __init__(
        self,
        call_id: str,
        sink: Sink,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
        wall_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        self._call_id = call_id
        self._sink = sink
        self._clock_ns = clock_ns
        self._wall_ns = wall_ns

    def mark(self, event: str, **fields: Any) -> int:
        """Record one moment. Returns its time, so a caller can measure from it."""
        at = self._clock_ns()
        self._write(event, at, fields)
        return at

    @contextmanager
    def span(self, event: str, **fields: Any) -> Iterator[None]:
        """Time a step: writes "<event>.start", then "<event>.end" with `ms`.

        The end record is written even if the step fails, with the error, so a
        crashed step still shows how long it ran before it broke.
        """
        start = self.mark(f"{event}.start", **fields)
        error: str | None = None
        try:
            yield
        except BaseException as exc:
            error = repr(exc)
            raise
        finally:
            end = self._clock_ns()
            extra = {"ms": (end - start) / 1_000_000, **fields}
            if error is not None:
                extra["error"] = error
            self._write(f"{event}.end", end, extra)

    def _write(self, event: str, at: int, fields: dict[str, Any]) -> None:
        self._sink.write({"call": self._call_id, "event": event, "at_ns": at, "wall_ns": self._wall_ns(), **fields})
