"""The four speech-to-text candidates (Part 2, decision 24), behind one shape.

Each service takes audio pieces and reports what it hears to `Heard`; nothing
outside this file knows how a service talks. Facts from each service's own docs,
2 Nov 2026, to be proved by the first real clip:

- Sarvam: the realtime endpoint, because the older /speech-to-text/ws documents
  no partial words, and decision 24 needs them. Its `mode` applies only to final
  text, so partial words may not come in English letters; the score shows this.
- Soniox: stt-rt-v5, with Hindi and English hints. The docs don't list 8 kHz or
  say which script Hindi comes in.
- Deepgram: nova-3 with language=multi, whose list includes Hindi. The docs don't
  say which script Hindi comes in.
- Google: needs 16 kHz, so the phone audio is raised to that. Nothing is gained:
  the sound above 4 kHz is already gone, exactly as on a real call.

No service gets a word list, so the comparison starts from each one as it comes.
"""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import urlencode

import websockets

SARVAM_URL = "wss://api.sarvam.ai/speech-to-text-realtime/ws"
SONIOX_URL = "wss://stt-rt.soniox.com/transcribe-websocket"
DEEPGRAM_URL = "wss://api.deepgram.com/v1/listen"
GOOGLE_MODEL = "gemini-3.5-transcribe-live"
SONIOX_MODEL = "stt-rt-v5"
DEEPGRAM_MODEL = "nova-3"
SARVAM_MODEL = "saaras:v3-realtime"
# All lines use one setup: a robot can't know the language before the caller
# speaks. Hindi is the base because most of our callers mix Hindi into English.
SARVAM_LANGUAGE = "hi-IN"
SONIOX_HINTS = ["hi", "en"]


class EarsError(Exception):
    """A service's refusal, with an HTTP-style status so the shared retry rules
    (common/retry.py) can judge it the same way as any other service's."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class Heard:
    """What came back and when. Partial text replaces itself until it's final."""

    clock: Callable[[], float]
    started: float = 0.0
    partials: int = 0
    first_text_at: float | None = None
    last_final_at: float | None = None
    last_activity: float = 0.0
    last_partial: str = ""
    finals: list[str] = field(default_factory=list)

    def _seen(self) -> float:
        now = self.clock() - self.started
        self.last_activity = now
        if self.first_text_at is None:
            self.first_text_at = now
        return now

    def partial(self, text: str) -> None:
        if text.strip():
            self._seen()
            self.partials += 1
            self.last_partial = text.strip()

    def final(self, text: str) -> None:
        if text.strip():
            self.last_final_at = self._seen()
            self.finals.append(text.strip())

    def final_piece(self, piece: str) -> None:
        """For a service that sends final text in bits that carry their own
        spaces: joining those with spaces would split words apart."""
        if piece:
            self.last_final_at = self._seen()
            self.finals[-1:] = [(self.finals[-1] if self.finals else "") + piece]

    @property
    def text(self) -> str:
        return " ".join(part.strip() for part in self.finals).strip()


class Service(Protocol):
    name: str
    rate: int  # the audio rate this service is sent

    async def stream(self, pieces: AsyncIterator[bytes], heard: Heard) -> None:
        """Send every piece and report what comes back, until the stream ends."""


async def _together(sending: Awaitable[None], reading: Awaitable[None]) -> None:
    """Run both. When the reply side ends, sending has no purpose. An error on
    either side comes out as itself, not wrapped in an exception group, because
    the retry rules look at its type and status."""
    send, read = asyncio.ensure_future(sending), asyncio.ensure_future(reading)
    try:
        while not read.done():
            await asyncio.wait({send, read}, return_when=asyncio.FIRST_COMPLETED)
            if send.done() and not send.cancelled() and send.exception():
                raise send.exception()
            if send.done() and not read.done():
                await read
        read.result()
    finally:
        for task in (send, read):
            task.cancel()
        await asyncio.gather(send, read, return_exceptions=True)


@dataclass
class Sarvam:
    key: str
    url: str = SARVAM_URL
    name: str = "sarvam"
    rate: int = 8000
    stream_type: str = "balanced"
    # How long a silence ends an utterance. None keeps Sarvam's own default
    # (500 ms in its docs); shorter gives final text sooner, if accuracy holds.
    silence_ms: int | None = None

    async def stream(self, pieces: AsyncIterator[bytes], heard: Heard) -> None:
        settings = {"model": SARVAM_MODEL, "language_code": SARVAM_LANGUAGE, "mode": "translit",
                    "stream_type": self.stream_type, "encoding": "linear16", "sample_rate": self.rate}
        if self.silence_ms is not None:
            settings["silence_duration_ms"] = self.silence_ms
        query = urlencode(settings)
        try:
            async with websockets.connect(f"{self.url}?{query}", additional_headers={"API-SUBSCRIPTION-KEY": self.key}) as ws:

                async def send() -> None:
                    async for piece in pieces:
                        await ws.send(json.dumps({"event": "audio_input", "audio": base64.b64encode(piece).decode()}))
                    await ws.send(json.dumps({"event": "end"}))

                async def read() -> None:
                    async for raw in ws:
                        message = json.loads(raw)
                        event = message.get("event")
                        if event == "transcript.partial":
                            heard.partial(message.get("text", ""))
                        elif event == "transcript.final":
                            heard.final(message.get("text", ""))
                        elif event == "session.end":
                            return
                        elif event == "error" and message.get("is_fatal", True):
                            raise EarsError(f"Sarvam: {message.get('message')}", message.get("status_code"))

                await _together(send(), read())
        except websockets.ConnectionClosedError as error:
            # Around everything: the close can reach the sending side first.
            raise _sarvam_closed(error) from error


def _sarvam_closed(error: websockets.ConnectionClosedError) -> EarsError:
    """Its close codes, from its docs: 1011 is its own fault, so worth another
    try. 1003 can also mean a bad key, and retrying can't fix that."""
    code = error.rcvd.code if error.rcvd else None
    return EarsError(f"Sarvam closed the line with {code}", 500 if code == 1011 else None)


@dataclass
class Soniox:
    key: str
    url: str = SONIOX_URL
    name: str = "soniox"
    rate: int = 8000

    async def stream(self, pieces: AsyncIterator[bytes], heard: Heard) -> None:
        async with websockets.connect(self.url, additional_headers={"Authorization": f"Bearer {self.key}"}) as ws:
            await ws.send(json.dumps({"model": SONIOX_MODEL, "audio_format": "s16le", "sample_rate": self.rate,
                                      "num_channels": 1, "language_hints": SONIOX_HINTS}))

            async def send() -> None:
                async for piece in pieces:
                    await ws.send(piece)
                await ws.send("")  # its docs: an empty text frame ends the stream; an empty binary one doesn't

            async def read() -> None:
                async for raw in ws:
                    message = json.loads(raw)
                    if message.get("error_code"):
                        raise EarsError(f"Soniox: {message.get('error_type')}: {message.get('error_message')}",
                                        message["error_code"])
                    tokens = message.get("tokens", [])
                    # Final tokens arrive once each, in order, and carry their own spaces.
                    heard.final_piece("".join(t["text"] for t in tokens if t.get("is_final")))
                    open_part = "".join(t["text"] for t in tokens if not t.get("is_final"))
                    if open_part:
                        heard.partial(open_part)
                    if message.get("finished"):
                        return

            await _together(send(), read())


@dataclass
class Deepgram:
    key: str
    url: str = DEEPGRAM_URL
    name: str = "deepgram"
    rate: int = 8000

    async def stream(self, pieces: AsyncIterator[bytes], heard: Heard) -> None:
        query = urlencode({"model": DEEPGRAM_MODEL, "language": "multi", "encoding": "linear16", "sample_rate": self.rate,
                           "channels": 1, "interim_results": "true"})
        async with websockets.connect(f"{self.url}?{query}", additional_headers={"Authorization": f"Token {self.key}"}) as ws:

            async def send() -> None:
                async for piece in pieces:
                    await ws.send(piece)
                await ws.send(json.dumps({"type": "CloseStream"}))

            async def read() -> None:
                async for raw in ws:
                    message = json.loads(raw)
                    if message.get("type") != "Results":
                        continue
                    text = message["channel"]["alternatives"][0].get("transcript", "")
                    if message.get("is_final"):
                        heard.final(text)
                    else:
                        heard.partial(text)

            await _together(send(), read())


@dataclass
class GoogleLive:
    client: Any
    name: str = "google"
    rate: int = 16000

    async def stream(self, pieces: AsyncIterator[bytes], heard: Heard) -> None:
        from google.genai import types

        config = types.LiveConnectConfig(
            response_modalities=["TEXT"],
            # Empty language list: its docs say this detects the language,
            # including speech that switches between languages.
            input_audio_transcription=types.AudioTranscriptionConfig(language_codes=[], mode="VERBATIM"),
        )
        async with self.client.aio.live.connect(model=GOOGLE_MODEL, config=config) as session:

            async def send() -> None:
                async for piece in pieces:
                    await session.send_realtime_input(audio=types.Blob(data=piece, mime_type=f"audio/pcm;rate={self.rate}"))
                await session.send_realtime_input(audio_stream_end=True)

            async def read() -> None:
                # One receive() ends with each turn, and a clip can have several;
                # the runner ends the listening once the line has gone quiet.
                while True:
                    got_any = False
                    async for message in session.receive():
                        got_any = True
                        content = message.server_content
                        if content is None:
                            continue
                        if content.interim_input_transcription and content.interim_input_transcription.text:
                            heard.partial(content.interim_input_transcription.text)
                        if content.input_transcription and content.input_transcription.text:
                            heard.final(content.input_transcription.text)
                    if not got_any:
                        return

            await _together(send(), read())
