"""The wire's parts without sockets: messages, the caller's speaker, the beat."""

import pytest

from common.wire.pacer import Pacer
from common.wire.playback import Playback
from common.wire.protocol import (
    PIECE_BYTES,
    SILENT_PIECE,
    Clear,
    Cleared,
    End,
    Key,
    Mark,
    Played,
    ProtocolError,
    Start,
    check_piece,
    decode,
    encode,
)

EVERY_MESSAGE = [
    Start("call-1"),
    Mark("q1"),
    Played("q1", 1240),
    Clear(),
    Cleared(37, 980, ("q1", "q2")),
    Cleared(0, 0, ()),
    Key("#", 20),
    End("bye"),
]


@pytest.mark.parametrize("message", EVERY_MESSAGE)
def test_every_message_survives_a_round_trip(message):
    assert decode(encode(message)) == message


def test_the_exact_text_on_the_wire():
    """Pinned, because the web phone's JavaScript writes and reads these by hand."""
    assert encode(Start("c1")) == '{"type":"start","call_id":"c1","sample_rate":8000,"piece_ms":20}'
    assert encode(Played("q1", 40)) == '{"type":"played","name":"q1","caller_ms":40}'
    assert encode(Cleared(3, 60, ("q1",))) == '{"type":"cleared","played_pieces":3,"caller_ms":60,"dropped_marks":["q1"]}'
    assert encode(Clear()) == '{"type":"clear"}'


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        "[1, 2]",
        '{"name": "q1"}',  # no type
        '{"type": "shout"}',  # unknown type
        '{"type": "mark"}',  # missing field
        '{"type": "mark", "name": "q1", "extra": 1}',  # extra field
        '{"type": "mark", "name": ""}',  # empty name
        '{"type": "played", "name": "q1", "caller_ms": "40"}',  # wrong type
        '{"type": "played", "name": "q1", "caller_ms": true}',  # bool is not a time
        '{"type": "played", "name": "q1", "caller_ms": 30}',  # not a piece boundary
        '{"type": "played", "name": "q1", "caller_ms": -20}',
        '{"type": "key", "digit": "A", "caller_ms": 0}',
        '{"type": "key", "digit": "12", "caller_ms": 0}',
        '{"type": "start", "call_id": "c", "sample_rate": 16000, "piece_ms": 20}',
        '{"type": "start", "call_id": "c", "sample_rate": 8000, "piece_ms": 10}',
        '{"type": "start", "call_id": "", "sample_rate": 8000, "piece_ms": 20}',
        '{"type": "cleared", "played_pieces": -1, "caller_ms": 0, "dropped_marks": []}',
        '{"type": "cleared", "played_pieces": 1, "caller_ms": 0, "dropped_marks": [3]}',
        '{"type": "end", "reason": ""}',
    ],
)
def test_anything_off_spec_is_refused(text):
    with pytest.raises(ProtocolError):
        decode(text)


def test_an_audio_piece_must_be_exactly_320_bytes():
    check_piece(SILENT_PIECE)
    assert PIECE_BYTES == 320
    for size in (0, 319, 321, 640):
        with pytest.raises(ProtocolError):
            check_piece(bytes(size))


# ---- the caller's speaker ------------------------------------------------------


def piece(n: int) -> bytes:
    return bytes([n % 256]) + bytes(PIECE_BYTES - 1)


def test_an_empty_speaker_plays_silence():
    assert Playback().tick(0) == (None, [])


def test_pieces_play_one_per_tick_and_a_mark_is_reached_when_its_audio_ends():
    speaker = Playback()
    for n in range(3):
        speaker.add_audio(piece(n))
    speaker.add_mark("q1")
    assert speaker.tick(100) == (piece(0), [])
    assert speaker.tick(120) == (piece(1), [])
    assert speaker.tick(140) == (piece(2), [("q1", 160)])  # the end of the last piece
    assert speaker.tick(160) == (None, [])
    assert speaker.played_pieces == 3


def test_a_mark_with_nothing_before_it_is_reached_at_the_start_of_the_tick():
    speaker = Playback()
    speaker.add_mark("now")
    speaker.add_audio(piece(0))
    speaker.add_mark("after")
    assert speaker.tick(40) == (piece(0), [("now", 40), ("after", 60)])


def test_marks_between_pieces_each_come_at_their_own_time():
    speaker = Playback()
    speaker.add_audio(piece(0))
    speaker.add_mark("a")
    speaker.add_mark("b")
    speaker.add_audio(piece(1))
    speaker.add_mark("c")
    assert speaker.tick(0) == (piece(0), [("a", 20), ("b", 20)])
    assert speaker.tick(20) == (piece(1), [("c", 40)])


def test_clear_drops_unplayed_audio_and_its_marks():
    speaker = Playback()
    for n in range(5):
        speaker.add_audio(piece(n))
    speaker.add_mark("q1")
    speaker.tick(0)
    speaker.tick(20)
    assert speaker.queued_pieces == 3 and speaker.is_playing
    assert speaker.clear() == ["q1"]
    assert speaker.played_pieces == 2 and not speaker.is_playing
    assert speaker.tick(40) == (None, [])


def test_the_speaker_refuses_a_wrong_size_piece():
    with pytest.raises(ProtocolError):
        Playback().add_audio(b"\x00" * 10)


# ---- the beat ------------------------------------------------------------------


class FakeTime:
    def __init__(self) -> None:
        self.now = 10.0
        self.slept: list[float] = []

    def clock(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.slept.append(round(seconds, 6))
        self.now += seconds


async def test_ticks_come_at_fixed_times_from_one_start():
    fake = FakeTime()
    ticks = Pacer(0.02, clock=fake.clock, sleep=fake.sleep).ticks()
    seen = []
    async for n in ticks:
        seen.append(n)
        fake.now += 0.005  # each tick's own work takes 5 ms
        if n == 3:
            break
    assert seen == [0, 1, 2, 3]
    assert fake.slept == [0.015, 0.015, 0.015]  # 20 ms beat minus the 5 ms of work


async def test_late_ticks_are_caught_up_at_once_and_none_is_skipped():
    fake = FakeTime()
    ticks = Pacer(0.02, clock=fake.clock, sleep=fake.sleep).ticks()
    seen = []
    async for n in ticks:
        seen.append(n)
        if n == 0:
            fake.now += 0.065  # the computer was busy for 65 ms
        if n == 4:
            break
    assert seen == [0, 1, 2, 3, 4]
    assert fake.slept == [0.015]  # ticks 1-3 were overdue: no sleep; tick 4 waited


def test_a_beat_needs_a_positive_period():
    with pytest.raises(ValueError):
        Pacer(0)
