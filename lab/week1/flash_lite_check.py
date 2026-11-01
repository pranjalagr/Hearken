"""Decision 2's free check: does gemini-3.5-flash-lite sound natural as the Talker?

20 moments of a call (talker_situations.yaml) go to the model with the
Talker's real style prompt, set up as decision 8 says: temperature at the
default 1.0 (Google's advice for Gemini 3) and thinking_level minimal. The
replies land in a report for a person to judge: 18 or more natural, and
Flash-Lite stays the Talker; fewer, and the Talker moves to gemini-3.8-flash.

The check never judges naturalness itself: that is exactly what a person must
do. It does record what code can measure: reply time, and thinking tokens
(Google says minimal thinking "does not guarantee that thinking is off", so
we look).
"""

from __future__ import annotations

import asyncio
import statistics
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from common.retry import asked_wait_s, asks_too_long, worth_retrying

ROOT = Path(__file__).resolve().parents[2]
SITUATIONS = Path(__file__).with_name("talker_situations.yaml")
STYLE_PROMPT = ROOT / "agents" / "agent_c" / "talker" / "prompts" / "style.md"

MODEL = "gemini-3.5-flash-lite"
FALLBACK_MODEL = "gemini-3.8-flash"
PASS_MARK = 18
# Decision 8: replies are kept short by the prompt (and later by code); this
# cap is only a backstop against a runaway answer.
MAX_OUTPUT_TOKENS = 200
# The free tier limits requests per minute. Measured on 1 Nov 2026: 20 calls
# 1 s apart hit "too many requests" and one reply needed 70 s of retries. So
# calls are spaced to stay under about 13 a minute; retries remain a backstop.
PAUSE_S = 4.5
RETRY_WAITS_S = (10.0, 20.0, 40.0)
# A short reply with minimal thinking comes in seconds; without a limit the
# library would wait on a stuck request forever (see common/google.py).
REQUEST_TIMEOUT_S = 30.0

SPEAKERS = {"caller": "Caller", "robot": "You"}


@dataclass(frozen=True)
class Situation:
    id: int
    good: str  # what a good reply does
    turns: tuple[tuple[str, str], ...]  # (speaker, text), oldest first
    notes: tuple[str, ...]

    @property
    def caller_line(self) -> str:
        return self.turns[-1][1]


@dataclass(frozen=True)
class Answer:
    text: str
    thought_tokens: int | None


@dataclass(frozen=True)
class Reply:
    situation: Situation
    text: str
    seconds: float
    thought_tokens: int | None = None
    error: str | None = None


Ask = Callable[[str], Awaitable[Answer]]


def load_situations(path: Path = SITUATIONS) -> list[Situation]:
    """Strict: a typo in the file must fail here, not as an odd reply."""
    situations = [situation_from(item) for item in yaml.safe_load(path.read_text(encoding="utf-8"))]
    if [s.id for s in situations] != list(range(1, len(situations) + 1)):
        raise ValueError("situation ids must be 1, 2, 3... in order")
    return situations


def situation_from(item: dict[str, Any], extra_keys: frozenset[str] = frozenset()) -> Situation:
    """One situation from the file. extra_keys: fields another check adds
    (the money check's expected wording), read by that check itself."""
    if set(item) - extra_keys != {"id", "good", "turns", "notes"}:
        raise ValueError(f"situation {item.get('id')}: needs exactly id, good, turns, notes")
    turns = []
    for turn in item["turns"]:
        if len(turn) != 1 or next(iter(turn)) not in SPEAKERS:
            raise ValueError(f"situation {item['id']}: a turn is caller: or robot:")
        speaker, text = next(iter(turn.items()))
        turns.append((speaker, str(text)))
    if not turns or turns[-1][0] != "caller":
        raise ValueError(f"situation {item['id']}: the last turn must be the caller's")
    return Situation(int(item["id"]), str(item["good"]), tuple(turns), tuple(item["notes"] or ()))


def packet(situation: Situation) -> str:
    """What the Talker reads for one reply (decision 15): the call so far,
    then the notes. The fixed rules go in the system prompt, first."""
    lines = ["The call so far:"]
    lines += [f"{SPEAKERS[speaker]}: {text}" for speaker, text in situation.turns]
    lines.append("")
    if situation.notes:
        lines.append("NOTES:")
        lines += [f"- {note}" for note in situation.notes]
    else:
        lines.append("NOTES: none yet")
    lines += ["", "Your next reply:"]
    return "\n".join(lines)


async def ask_all(
    situations: list[Situation],
    ask: Ask,
    pause_s: float = PAUSE_S,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> list[Reply]:
    """One after another, never in parallel: the free tier counts requests
    per minute, and a failed reply is recorded, not fatal."""
    replies = []
    for n, situation in enumerate(situations):
        if n:
            await sleep(pause_s)
        start = clock()
        try:
            answer = await ask(packet(situation))
            replies.append(Reply(situation, answer.text.strip(), clock() - start, answer.thought_tokens))
        except Exception as exc:
            replies.append(Reply(situation, "", clock() - start, error=f"{type(exc).__name__}: {exc}"))
    return replies


async def with_retries(
    call: Callable[[], Awaitable[Any]],
    waits: tuple[float, ...] = RETRY_WAITS_S,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    on_retry: Callable[[BaseException, float], None] | None = None,
) -> Any:
    """The shared rules (common/retry.py): too many requests, a server error
    or a stuck request is tried again, after the wait the service asks for if
    that is longer; a daily limit or a bad key fails at once.
    on_retry(error, wait) is told of each retry: silent retries hid where a
    judge check's 8 minutes went (1 Nov 2026)."""
    for wait in (*waits, None):
        try:
            return await call()
        except Exception as error:
            if wait is None or not worth_retrying(error) or asks_too_long(error):
                raise
            wait = max(wait, asked_wait_s(error) or 0.0)
            if on_retry:
                on_retry(error, wait)
            await sleep(wait)
    raise AssertionError("unreachable")


def gemini_asker(client: Any, model: str, style: str) -> Ask:
    from google.genai import types

    config = types.GenerateContentConfig(
        system_instruction=style,
        temperature=1.0,
        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL),
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )

    async def ask(text: str) -> Answer:
        response = await with_retries(
            lambda: client.aio.models.generate_content(model=model, contents=text, config=config)
        )
        usage = response.usage_metadata
        return Answer(response.text or "", usage.thoughts_token_count if usage else None)

    return ask


def report(replies: list[Reply], model: str, when: datetime) -> str:
    """A table to judge in VS Code's Markdown preview, with the model name and
    date next to the numbers (decision 2)."""
    seconds = [r.seconds for r in replies if r.error is None]
    thinking = [r for r in replies if (r.thought_tokens or 0) > 0]
    lines = [
        "# Flash-Lite check (Part 2, decision 2)",
        "",
        f"- **Model:** `{model}`, run on {when:%d %b %Y at %H:%M}",
        f"- **Settings:** temperature 1.0, thinking_level minimal, max {MAX_OUTPUT_TOKENS} output tokens",
        f"- **Style prompt:** `{STYLE_PROMPT.relative_to(ROOT).as_posix()}`",
        f"- **Rule:** {PASS_MARK} or more of {len(replies)} natural: `{model}` stays the Talker. "
        f"Fewer: the Talker moves to `{FALLBACK_MODEL}`.",
        "",
        "**How to judge:** put ✅ or ❌ in the last column. Natural means a real, polite "
        "support person could say it, in the caller's language, and it does what the "
        "'good reply' column says.",
        "",
        "| # | A good reply... | Caller said | Flash-Lite replied | Natural? |",
        "|---|---|---|---|---|",
    ]
    for reply in replies:
        answer = f"**ERROR:** {reply.error}" if reply.error else reply.text
        lines.append(
            f"| {reply.situation.id} | {_cell(reply.situation.good)} | {_cell(reply.situation.caller_line)} "
            f"| {_cell(answer)} |  |"
        )
    lines += ["", "## Measured"]
    if seconds:
        lines.append(
            f"- **Reply time:** median {statistics.median(seconds):.2f} s, slowest {max(seconds):.2f} s "
            "(whole reply, not first word; includes network)"
        )
    lines.append(f"- **Replies that used thinking tokens:** {len(thinking)} of {len(replies)}")
    errors = [r for r in replies if r.error]
    lines.append(f"- **Errors:** {len(errors)}")
    return "\n".join(lines) + "\n"


def _cell(text: str) -> str:
    """Keep a table cell on one line, and a | inside the text from splitting it."""
    return " ".join(text.split()).replace("|", "\\|")


async def run(out_dir: Path = ROOT / "runs" / "week1") -> tuple[Path, list[Reply]]:
    from common.env import load_env, require
    from common.google import google_client

    load_env()
    client = google_client(require("GEMINI_API_KEY"), REQUEST_TIMEOUT_S)
    try:
        replies = await ask_all(load_situations(), gemini_asker(client, MODEL, STYLE_PROMPT.read_text(encoding="utf-8")))
    finally:
        await client.aio.aclose()
    when = datetime.now()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"flash_lite_check_{when:%Y%m%d_%H%M}.md"
    path.write_text(report(replies, MODEL, when), encoding="utf-8")
    return path, replies
