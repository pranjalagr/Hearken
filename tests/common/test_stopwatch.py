"""The stopwatch: marks, spans, and the JSON-lines file."""

import json
import time

import pytest

from common.stopwatch import JsonlSink, MemorySink, Stopwatch


class FakeClock:
    def __init__(self) -> None:
        self.ns = 1_000

    def __call__(self) -> int:
        return self.ns


def test_a_mark_records_the_event_its_fields_and_the_call():
    sink, clock = MemorySink(), FakeClock()
    watch = Stopwatch("call-1", sink, clock_ns=clock, wall_ns=lambda: 42)
    assert watch.mark("ears.final", words=3) == 1_000
    assert sink.records == [{"call": "call-1", "event": "ears.final", "at_ns": 1_000, "wall_ns": 42, "words": 3}]


def test_a_span_records_start_end_and_duration():
    sink, clock = MemorySink(), FakeClock()
    watch = Stopwatch("c", sink, clock_ns=clock)
    with watch.span("tool", name="find_order"):
        clock.ns += 2_500_000  # 2.5 ms pass inside the step
    start, end = sink.records
    assert (start["event"], end["event"]) == ("tool.start", "tool.end")
    assert end["ms"] == 2.5 and end["name"] == "find_order" and "error" not in end


def test_a_failing_step_still_records_its_end_and_the_error():
    sink, clock = MemorySink(), FakeClock()
    watch = Stopwatch("c", sink, clock_ns=clock)
    with pytest.raises(ValueError):
        with watch.span("thinker.step"):
            clock.ns += 1_000_000
            raise ValueError("boom")
    end = sink.records[-1]
    assert end["event"] == "thinker.step.end" and end["ms"] == 1.0 and "boom" in end["error"]


def test_the_real_clock_measures_real_time():
    sink = MemorySink()
    with Stopwatch("c", sink).span("wait"):
        time.sleep(0.02)
    assert 15 <= sink.records[-1]["ms"] < 500


def test_jsonl_sink_writes_one_line_per_record_and_keeps_appending(tmp_path):
    path = tmp_path / "calls" / "events.jsonl"
    sink = JsonlSink(path)
    watch = Stopwatch("c", sink)
    watch.mark("call.started")
    watch.mark("call.ended", reason="hang_up")
    sink.close()
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [line["event"] for line in lines] == ["call.started", "call.ended"]
    assert lines[1]["reason"] == "hang_up"
    assert lines[0]["at_ns"] <= lines[1]["at_ns"]
