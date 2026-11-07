"""The speed and turn-taking numbers of one call (Part 5, L9).

Every number is measured at the caller's ear, after the call, from the timed
notes in the call's folder: the caller's own speech, the robot sound it
heard, the backend's tool times. The robot's diary never sets a score.

    1. reply delay       caller's sentence ends -> caller hears robot sound
    2. tool silence      longest stretch with no robot sound while a tool runs
    3. stops on "ruko"   robot sound ends within stop_within_s of the "ruko"
    4. wrong stops       a "haan haan" made the robot stop mid-sentence
    5. robot cut-offs    caller lines the robot started talking over
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from common.wire.protocol import PIECE_MS
from lab.scorer.call_record import CallRecord, Span

# A robot that starts within this much of the caller's line start is
# answering a turn that ended, not cutting a sentence in half.
CUT_OFF_GRACE_MS = 100


@dataclass
class CallTiming:
    reply_delays_ms: list[float] = field(default_factory=list)
    tool_silences_ms: list[float] = field(default_factory=list)
    stops: list[dict[str, Any]] = field(default_factory=list)  # one per "ruko" said while the robot talked
    wrong_stops: list[dict[str, Any]] = field(default_factory=list)  # one per "haan haan" said while it talked
    cut_offs: int = 0
    turns: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def measure(record: CallRecord, stop_within_s: float, wrong_stop_quiet_s: float) -> CallTiming:
    timing = CallTiming()
    sound = record.robot_sound
    lines = record.caller_lines()
    timing.turns = len(lines)
    for line in lines:
        delay = _reply_delay(line, sound)
        if delay is not None:
            timing.reply_delays_ms.append(delay)
    caller_speech = [Span(line["start_ms"], line["end_ms"]) for line in lines]
    caller_speech += [Span(d["caller_ms"], d["caller_ms"] + PIECE_MS) for d in record.director_lines()]
    for _tool, span in record.tool_spans():
        timing.tool_silences_ms.append(_longest_silence(span, sound, caller_speech))
    for at in record.stops():
        if _playing(sound, at):
            ended = _sound_ends_after(sound, at)
            stopped = ended is not None and ended - at <= stop_within_s * 1000
            timing.stops.append({"at_ms": at, "stopped": stopped, "after_ms": None if ended is None else ended - at})
    clears = record.clears()
    for at in record.backchannels():
        if _playing(sound, at):
            # A wrong stop is the robot cutting its own sentence (it cleared
            # the caller's speaker) soon after the "haan haan".
            cleared = [c for c in clears if 0 <= c - at <= wrong_stop_quiet_s * 1000]
            timing.wrong_stops.append({"at_ms": at, "stopped": bool(cleared)})
    # Per caller line: one robot sentence is many sound spans, one per breath, and counted
    # by span a single overlap looked like four (4 Nov 2026).
    starts = [span.start_ms for span in sound]
    timing.cut_offs = sum(any(line["start_ms"] + CUT_OFF_GRACE_MS < start < line["end_ms"] for start in starts) for line in lines)
    return timing


def _reply_delay(line: dict[str, Any], sound: list[Span]) -> float | None:
    """None when there's no clean answer to time: the robot was already
    talking when the caller finished (an overlap), or never spoke again."""
    end = line["end_ms"]
    if _playing(sound, end - PIECE_MS):
        return None
    after = [span.start_ms for span in sound if span.start_ms >= end]
    return min(after) - end if after else None


def _longest_silence(tool: Span, sound: list[Span], caller: list[Span]) -> float:
    """The longest stretch inside the tool's run with neither robot sound nor
    the caller talking (when the caller talks, the robot's silence is right)."""
    busy = sorted([*sound, *caller], key=lambda s: s.start_ms)
    longest, cursor = 0.0, tool.start_ms
    for span in busy:
        if span.end_ms <= cursor or span.start_ms >= tool.end_ms:
            continue
        longest = max(longest, span.start_ms - cursor)
        cursor = max(cursor, span.end_ms)
    return max(longest, tool.end_ms - cursor) if cursor < tool.end_ms else longest


def _playing(sound: list[Span], at: float) -> bool:
    return any(span.start_ms <= at < span.end_ms for span in sound)


def _sound_ends_after(sound: list[Span], at: float) -> float | None:
    for span in sound:
        if span.start_ms <= at < span.end_ms:
            return span.end_ms
    return None
