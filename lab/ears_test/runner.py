"""Stream every clip to every service at the speed a real caller talks.

The same care as the voice test, so limited free requests aren't wasted: results
are saved after every clip, finished clips are never sent again, a daily limit
stops that service at once, and every retry is logged.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from pathlib import Path
from typing import Any

from common.audio.pcm import float_to_pcm16
from common.audio.resample import resample
from common.audio.sound_files import read_sound
from common.retry import asked_wait_s, asks_too_long
from lab.ears_test.clips import VERSIONS, clip_path
from lab.ears_test.services import Heard, Service
from lab.week1.flash_lite_check import with_retries

PIECE_S = 0.1  # 100 ms pieces: the size Google's docs ask for, and fine for the others
# Once all the audio is sent and nothing has come back for this long, the service
# has said all it will. Not every service closes the line by itself.
QUIET_S = 4.0
# On top of the clip's length: past this the request is stuck, not slow.
STUCK_AFTER_S = 20.0
# Lasting failures in a row (a bad key, an unsupported setting): stop that service
# for this run rather than spend its limit failing the same way on every clip.
STOP_AFTER_FAILURES = 3
RETRY_WAITS_S = (15.0, 30.0, 60.0)


def audio_bytes(path: Path, rate: int) -> bytes:
    samples, file_rate = read_sound(path)
    return float_to_pcm16(resample(samples, file_rate, rate))


async def _paced(
    audio: bytes, rate: int, on_start: Callable[[float], None], on_done: Callable[[], None],
    clock: Callable[[], float], sleep: Callable[[float], Awaitable[None]],
) -> AsyncIterator[bytes]:
    """Pieces at real speed, timed from the start, so small delays don't add up."""
    size = int(rate * PIECE_S) * 2
    start = clock()
    on_start(start)
    for index in range(0, len(audio), size):
        wait = start + (index / size) * PIECE_S - clock()
        if wait > 0:
            await sleep(wait)
        yield audio[index:index + size]
    on_done()


async def listen_once(
    service: Service, audio: bytes, clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep, quiet_s: float = QUIET_S,
) -> Heard:
    heard = Heard(clock)
    heard.started = clock()
    sent: list[float] = []

    def audio_starts(at: float) -> None:
        # The clip's clock starts with its first sound, not before the line opens:
        # counting the connection made every service look about 2 s slow on the
        # first real clips (2 Nov 2026). On a call the line is already open.
        heard.started = at

    pieces = _paced(audio, service.rate, audio_starts, lambda: sent.append(clock() - heard.started), clock, sleep)
    limit = len(audio) / (2 * service.rate) + STUCK_AFTER_S
    task = asyncio.ensure_future(service.stream(pieces, heard))
    try:
        while not task.done():
            await asyncio.wait({task}, timeout=0.05)
            now = clock() - heard.started
            if sent and now - max(sent[0], heard.last_activity) > quiet_s:
                return heard
            if now > limit:
                raise TimeoutError(f"{service.name} was stuck for {now:.0f} s")
        task.result()
        return heard
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def _row(heard: Heard, clip: dict[str, Any], retries: int) -> dict[str, Any]:
    start, end = clip["speech_start_s"], clip["speech_end_s"]
    return {
        "final": heard.text,
        "last_partial": heard.last_partial,
        "partials": heard.partials,
        # More than one means the service split the caller's speech: a short
        # silence setting can do this to "chaar... paanch..." said with pauses.
        "final_parts": len(heard.finals),
        # From the caller starting to speak until the first words came back.
        "first_text_s": None if heard.first_text_at is None else round(heard.first_text_at - start, 3),
        # From the caller finishing until the final words came back.
        "final_after_speech_s": None if heard.last_final_at is None else round(heard.last_final_at - end, 3),
        "retries": retries,
    }


def load_results(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_results(path: Path, results: dict[str, Any]) -> None:
    # Whole or not at all: a crash mid-write must not lose what's already paid for.
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(results, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, path)


async def run_service(
    service: Service, clips: dict[str, dict[str, Any]], results: dict[str, Any], results_path: Path, sounds: Path,
    versions: Iterable[str] = VERSIONS, log: Callable[[str], None] = print,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    listen: Callable[[Service, bytes], Awaitable[Heard]] = listen_once,
) -> None:
    failures_in_a_row = 0
    for clip_key, clip in sorted(clips.items()):
        speaker, line_id = clip_key.split("/")
        for version in versions:
            key = f"{service.name}/{clip_key}/{version}"
            if key in results and "error" not in results[key]:
                continue
            audio = audio_bytes(clip_path(version, speaker, line_id, sounds), service.rate)
            retries: list[float] = []

            def note_retry(error: BaseException, wait: float, key: str = key) -> None:
                retries.append(wait)
                log(f"  {key}: retrying in {wait:.0f} s after {type(error).__name__}: {str(error)[:200]}")

            try:
                heard = await with_retries(lambda: listen(service, audio), RETRY_WAITS_S, sleep, note_retry)
            except Exception as error:
                results[key] = {"error": f"{type(error).__name__}: {str(error)[:300]}", "retries": len(retries)}
                save_results(results_path, results)
                log(f"  {key}: FAILED {results[key]['error']}")
                failures_in_a_row += 1
                if asks_too_long(error):
                    log(f"  {service.name} asks to wait {asked_wait_s(error) / 3600:.1f} h: stopped; run again later")
                    return
                if failures_in_a_row >= STOP_AFTER_FAILURES:
                    log(f"  {service.name}: {failures_in_a_row} failures in a row: stopped, to save its limit")
                    return
                continue
            failures_in_a_row = 0
            results[key] = _row(heard, clip, len(retries))
            save_results(results_path, results)
            log(f"  {key}: {heard.text[:70]!r}")


async def run_all(services: list[Service], clips: dict[str, dict[str, Any]], results_path: Path, sounds: Path,
                  versions: Iterable[str] = VERSIONS, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Services run side by side, each sending one clip at a time: a clip takes
    its real length to stream, and they don't share any limit."""
    results = load_results(results_path)
    versions = tuple(versions)
    await asyncio.gather(*(run_service(s, clips, results, results_path, sounds, versions, log) for s in services))
    return results
