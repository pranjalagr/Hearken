"""The robot's diary (Part 5, L5): what went on inside the robot, one timed
line per event, in the call's folder (robot.jsonl).

The caller's side records what the customer experienced (Part 3, C16); this
records why the robot did what it did: what it heard, what it said, which
tools it used, what it decided, how long each step took and what it cost.

Every robot writes the same kinds of lines, so A, B and C can be compared.
A robot without a part simply has no lines of that kind (B has no Guard, so
no "decision" lines). Times use the one shared clock (time.perf_counter_ns,
L6), so the lab can put these lines on the same timeline as the caller's.

The diary explains a call; it never scores one (L9): the robot can't grade itself.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from common.stopwatch import Sink

# The core kinds, in the order L5 lists them.
KINDS = ("heard", "said", "tool", "decision", "step", "cost", "error")


class Diary:
    def __init__(self, call_id: str, sink: Sink, clock_ns: Callable[[], int] = time.perf_counter_ns) -> None:
        self._call_id = call_id
        self._sink = sink
        self._clock_ns = clock_ns

    def write(self, kind: str, **fields: Any) -> None:
        if kind not in KINDS:
            # A made-up kind would be a line no scorer and no other robot knows.
            raise ValueError(f"unknown diary kind {kind!r}; use one of {KINDS}")
        self._sink.write({"call": self._call_id, "kind": kind, "at_ns": self._clock_ns(), **fields})

    def heard(self, text: str | None, **fields: Any) -> None:
        """What the robot heard (None: it heard speech but has no ears to read it)."""
        self.write("heard", text=text, **fields)

    def said(self, text: str, filled: str | None = None, **fields: Any) -> None:
        """What the robot meant to say; `filled` is the text after its slots were filled."""
        self.write("said", text=text, filled=filled if filled is not None else text, **fields)

    def tool(self, name: str, inputs: dict[str, Any], status: str, ms: float, **fields: Any) -> None:
        self.write("tool", name=name, inputs=inputs, status=status, ms=round(ms, 1), **fields)

    def decision(self, what: str, **fields: Any) -> None:
        self.write("decision", what=what, **fields)

    def step(self, name: str, ms: float, **fields: Any) -> None:
        self.write("step", name=name, ms=round(ms, 1), **fields)

    def cost(self, service: str, amount: float, unit: str, **fields: Any) -> None:
        self.write("cost", service=service, amount=amount, unit=unit, **fields)

    def error(self, text: str, **fields: Any) -> None:
        self.write("error", text=text, **fields)


class _NoSink:
    def write(self, record: dict[str, Any]) -> None:
        pass


# For robots run without a lab (the web phone, most tests): writes nothing.
NO_DIARY = Diary("none", _NoSink())
