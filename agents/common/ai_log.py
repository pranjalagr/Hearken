"""One diary row per AI request: which service, model and key, what it was for, how long it
took, how it ended, and the service's own reason when it gives one (18 Nov 2026).

Why: in run agent-c-changed the Talker's AI timed out 17 times and the yes-judge twice, and
the diary could not say which key was slow, whether Google was busy (503, "overloaded") or
just slow, or whether the time went in a queue or in the model. Each row now says:
- the key by its .env name ("GEMINI_API_KEY_4"), never the key itself: every message is
  scrubbed of every key before it is written;
- ms, and for a streamed reply first_ms (time to its first words): a long wait for the first
  words is the service's queue, a long tail is the model writing;
- outcome: "ok"; "error" with the service's status and message; or "cut off": our own time
  limit ran out, or another key answered first (the yes-judge asks several keys at once);
- what the service says about itself: tokens in, out and spent thinking, the finish reason,
  Groq's own queue and compute times, and headers that tell time or limits (server-timing,
  retry-after, the rate-limit counts).
Writing the row is a list append in the diary's sink: nothing here waits or blocks.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable, Iterable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

AI_REQUEST = "ai request"

# Headers that tell why a request was slow or refused, when a service sends them.
REASON_HEADERS = ("server-timing", "retry-after", "x-groq-region", "x-ratelimit-remaining-requests",
                  "x-ratelimit-remaining-tokens", "x-ratelimit-reset-requests", "x-ratelimit-reset-tokens")
MESSAGE_CHARS = 240  # enough for Google's "The model is overloaded. Please try again later."


@dataclass
class Request:
    """One request being timed. The client adds what the service said about it."""

    started: float = field(default_factory=time.perf_counter)
    first_at: float | None = None
    details: dict[str, Any] = field(default_factory=dict)
    failed: dict[str, Any] | None = None  # an error answer that came back without raising
    # Time spent on our side in the middle of a streamed reply (the Talker handing a sentence
    # to the voice, which waits while the caller talks): taken out of ms, logged on its own.
    ours_s: float = 0.0

    @asynccontextmanager
    async def our_time(self) -> AsyncIterator[None]:
        """Wrap our own work done in the middle of the request, so it isn't counted as the AI's."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.ours_s += time.perf_counter() - start

    def first_piece(self) -> None:
        """A streamed reply's first words arrived."""
        if self.first_at is None:
            self.first_at = time.perf_counter()

    def add(self, details: Mapping[str, Any]) -> None:
        self.details.update({key: value for key, value in details.items() if value not in (None, "", {})})

    def refused(self, status: int, message: str, headers: Mapping[str, str] | None = None) -> None:
        """The service answered with an error (a 429, a 5xx) that the client handles itself."""
        self.failed = {"status": status, "message": message, **_reason_headers(headers)}


class AiLog:
    """Writes the rows. `decision` is the diary's decision(what, **fields); `secrets` are
    every key the robot holds, scrubbed out of anything written."""

    def __init__(self, decision: Callable[..., None], secrets: Iterable[str] = ()) -> None:
        self._decision = decision
        self._secrets = [secret for secret in secrets if secret]

    @asynccontextmanager
    async def request(self, service: str, model: str, key: str, purpose: str) -> AsyncIterator[Request]:
        """Time one request: `async with log.request(...) as req:` around it."""
        req = Request()
        outcome, error_fields = "ok", {}
        try:
            yield req
        except asyncio.CancelledError:
            # Our own limit (asyncio.wait_for) ran out, or another racer won: the service
            # never answered in time. Not an error of the service's own.
            outcome = "cut off"
            raise
        except Exception as error:
            outcome, error_fields = "error", error_details(error)
            raise
        finally:
            if outcome == "ok" and req.failed is not None:
                outcome, error_fields = "error", req.failed
            self._write(service, model, key, purpose, req, outcome, error_fields)

    def _write(self, service: str, model: str, key: str, purpose: str, req: Request, outcome: str,
               error_fields: Mapping[str, Any]) -> None:
        now = time.perf_counter()
        row: dict[str, Any] = {"service": service, "model": model, "key": key or "unnamed", "purpose": purpose,
                               "ms": round((now - req.started - req.ours_s) * 1000), "outcome": outcome}
        if req.ours_s:
            row["our_wait_ms"] = round(req.ours_s * 1000)
        if req.first_at is not None:
            row["first_ms"] = round((req.first_at - req.started) * 1000)
        row.update(req.details)
        row.update(error_fields)
        if "message" in row:
            row["message"] = self._scrub(str(row["message"]))[:MESSAGE_CHARS]
        try:
            self._decision(AI_REQUEST, **row)
        except Exception:
            pass  # the log must never break a call

    def _scrub(self, text: str) -> str:
        for secret in self._secrets:
            text = text.replace(secret, "***")
        return text


NO_LOG: AiLog | None = None  # the clients' default: tests and tools that log nothing


@asynccontextmanager
async def timed(log: AiLog | None, service: str, model: str, key: str, purpose: str) -> AsyncIterator[Request]:
    """log.request(...), or a request nobody writes down when there is no log."""
    if log is None:
        yield Request()
        return
    async with log.request(service, model, key, purpose) as req:
        yield req


# ---- what each service says about a request ------------------------------------------------

def _reason_headers(headers: Mapping[str, str] | None) -> dict[str, str]:
    if not headers:
        return {}
    lower = {str(name).lower(): str(value) for name, value in dict(headers).items()}
    return {name.replace("-", "_"): lower[name] for name in REASON_HEADERS if name in lower}


def gemini_details(response: Any) -> dict[str, Any]:
    """Tokens, finish reason, model version and timing headers from a Gemini answer (or the
    last piece of a streamed one, which carries the usage)."""
    out: dict[str, Any] = {}
    usage = getattr(response, "usage_metadata", None)
    if usage is not None:
        out.update(tokens_in=getattr(usage, "prompt_token_count", None),
                   tokens_out=getattr(usage, "candidates_token_count", None),
                   tokens_thinking=getattr(usage, "thoughts_token_count", None),
                   traffic=_name(getattr(usage, "traffic_type", None)))
    candidates = getattr(response, "candidates", None) or []
    if candidates:
        out["finish"] = _name(getattr(candidates[0], "finish_reason", None))
    out["model_version"] = getattr(response, "model_version", None)
    http = getattr(response, "sdk_http_response", None)
    out.update(_reason_headers(getattr(http, "headers", None)))
    return out


def openai_style_details(response: Any) -> dict[str, Any]:
    """Tokens and timing from an OpenAI-style chat answer (Groq, Sarvam). Groq also says how
    long the request queued and how long the model ran (usage.queue_time etc., in seconds)."""
    out: dict[str, Any] = dict(_reason_headers(getattr(response, "headers", None)))
    try:
        usage = response.json().get("usage") or {}
    except Exception:
        return out
    out.update(tokens_in=usage.get("prompt_tokens"), tokens_out=usage.get("completion_tokens"))
    for name in ("queue_time", "prompt_time", "completion_time", "total_time"):
        if isinstance(usage.get(name), (int, float)):
            out[f"service_{name.removesuffix('_time')}_ms"] = round(usage[name] * 1000)
    return out


def error_details(error: BaseException) -> dict[str, Any]:
    """The status and message an error carries: Google's APIError has .code and .message;
    an httpx error has its response."""
    out: dict[str, Any] = {"error": type(error).__name__, "message": str(error)}
    code = getattr(error, "code", None)
    if isinstance(code, int):
        out["status"] = code
    if getattr(error, "status", None):
        out["status_name"] = str(error.status)
    response = getattr(error, "response", None)
    if response is not None and hasattr(response, "status_code"):
        out["status"] = response.status_code
        out.update(_reason_headers(getattr(response, "headers", None)))
    return out


def _name(value: Any) -> Any:
    """An enum's name ("STOP", "ON_DEMAND"), or the value as it is."""
    return getattr(value, "name", value)
