"""The wire's messages, and the one place that checks them.

Audio travels as binary frames: exactly one 20 ms piece each, 160 samples of
16-bit little-endian PCM at 8 kHz (320 bytes). Everything else is a text frame
holding one JSON object with a "type".

The caller's clock (`caller_ms`): the caller's side sends one audio piece every
20 ms for the whole call, so "pieces sent so far x 20" is a clock both sides
can read exactly. The robot knows that the k-th caller piece it receives
started at caller_ms = 20 * k, and every message from the caller is stamped on
the same clock. Comparing "the question finished playing at caller_ms X" with
"the haan started at caller_ms Y" therefore needs no guess about network delay
(Part 2 decision 22).

Messages:
  caller -> robot   start, key, played, cleared, end
  robot -> caller   mark, clear, end

Anything that doesn't match exactly is refused with ProtocolError: a wrong
guess about audio format would be heard as noise, so it is better to stop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from common.audio.resample import PHONE_RATE

SAMPLE_RATE = PHONE_RATE
PIECE_MS = 20
PIECE_SAMPLES = SAMPLE_RATE * PIECE_MS // 1000  # 160
PIECE_BYTES = PIECE_SAMPLES * 2  # 320: 16-bit samples
SILENT_PIECE = bytes(PIECE_BYTES)

KEYPAD = frozenset("0123456789*#")

# Our biggest message is a 320-byte piece or a short JSON line; anything far
# bigger is a bug or an attack, so it is refused instead of buffered.
MAX_MESSAGE_BYTES = 64 * 1024

# WebSocket close codes used on the wire.
CLOSE_NORMAL = 1000
CLOSE_PROTOCOL_ERROR = 1002


def close_reason(text: str) -> str:
    """WebSocket close reasons must fit in 123 bytes of UTF-8."""
    return text.encode()[:120].decode(errors="ignore")


class ProtocolError(ValueError):
    """The other side broke the wire's rules."""


@dataclass(frozen=True)
class Start:
    """Caller -> robot, first message: the call begins."""

    call_id: str
    sample_rate: int = SAMPLE_RATE
    piece_ms: int = PIECE_MS


@dataclass(frozen=True)
class Mark:
    """Robot -> caller: "tell me when the audio before this has been played"."""

    name: str


@dataclass(frozen=True)
class Played:
    """Caller -> robot: everything before mark `name` finished playing at caller_ms."""

    name: str
    caller_ms: int


@dataclass(frozen=True)
class Clear:
    """Robot -> caller: stop playing; throw away the audio not yet played."""


@dataclass(frozen=True)
class Cleared:
    """Caller -> robot, the answer to Clear: how much was really heard.

    `played_pieces` counts every robot piece played since the call began, so
    the robot can tell exactly where in its sentence it was cut off.
    `dropped_marks` were thrown away unplayed: for a yes-question, that means
    the caller never heard its end.
    """

    played_pieces: int
    caller_ms: int
    dropped_marks: tuple[str, ...] = ()


@dataclass(frozen=True)
class Key:
    """Caller -> robot: a keypad key was pressed."""

    digit: str
    caller_ms: int


@dataclass(frozen=True)
class End:
    """Either way: the call is over (a clean hang-up)."""

    reason: str


Message = Start | Mark | Played | Clear | Cleared | Key | End

_TYPES: dict[str, type] = {
    "start": Start,
    "mark": Mark,
    "played": Played,
    "clear": Clear,
    "cleared": Cleared,
    "key": Key,
    "end": End,
}
_NAMES = {cls: name for name, cls in _TYPES.items()}

# Field name -> expected JSON type, per message type.
_FIELDS: dict[type, dict[str, type]] = {
    Start: {"call_id": str, "sample_rate": int, "piece_ms": int},
    Mark: {"name": str},
    Played: {"name": str, "caller_ms": int},
    Clear: {},
    Cleared: {"played_pieces": int, "caller_ms": int, "dropped_marks": list},
    Key: {"digit": str, "caller_ms": int},
    End: {"reason": str},
}


def encode(message: Message) -> str:
    _check(message)
    body: dict[str, Any] = {"type": _NAMES[type(message)]}
    for field in _FIELDS[type(message)]:
        value = getattr(message, field)
        body[field] = list(value) if isinstance(value, tuple) else value
    return json.dumps(body, ensure_ascii=False, separators=(",", ":"))


def decode(text: str) -> Message:
    try:
        body = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"not JSON: {exc}") from None
    if not isinstance(body, dict):
        raise ProtocolError("a message must be a JSON object")
    kind = body.pop("type", None)
    cls = _TYPES.get(kind) if isinstance(kind, str) else None
    if cls is None:
        raise ProtocolError(f"unknown message type {kind!r}")
    fields = _FIELDS[cls]
    if set(body) != set(fields):
        raise ProtocolError(f"{kind} needs exactly the fields {sorted(fields)}, got {sorted(body)}")
    for name, expected in fields.items():
        value = body[name]
        # bool is a kind of int in Python, but true is never a time or a count.
        if not isinstance(value, expected) or (expected is int and isinstance(value, bool)):
            raise ProtocolError(f"{kind}.{name} must be {expected.__name__}")
    if cls is Cleared:
        body["dropped_marks"] = tuple(body["dropped_marks"])
    message = cls(**body)
    _check(message)
    return message


def check_piece(data: bytes) -> None:
    if len(data) != PIECE_BYTES:
        raise ProtocolError(f"an audio piece must be {PIECE_BYTES} bytes, got {len(data)}")


def _check(message: Message) -> None:
    """The rules a well-typed message must still follow."""
    if isinstance(message, Start):
        if not message.call_id:
            raise ProtocolError("start.call_id is empty")
        if (message.sample_rate, message.piece_ms) != (SAMPLE_RATE, PIECE_MS):
            raise ProtocolError(f"the wire is {SAMPLE_RATE} Hz in {PIECE_MS} ms pieces")
    elif isinstance(message, (Mark, Played)):
        if not message.name:
            raise ProtocolError("a mark needs a name")
    elif isinstance(message, Key):
        if message.digit not in KEYPAD:
            raise ProtocolError(f"not a keypad key: {message.digit!r}")
    elif isinstance(message, Cleared):
        if message.played_pieces < 0:
            raise ProtocolError("played_pieces must be >= 0")
        if not all(isinstance(name, str) and name for name in message.dropped_marks):
            raise ProtocolError("dropped_marks must be mark names")
    elif isinstance(message, End) and not message.reason:
        raise ProtocolError("end needs a reason")
    caller_ms = getattr(message, "caller_ms", 0)
    if caller_ms < 0 or caller_ms % PIECE_MS:
        # Every caller time is a piece boundary: 20 x pieces sent.
        raise ProtocolError(f"caller_ms must be a non-negative multiple of {PIECE_MS}")
