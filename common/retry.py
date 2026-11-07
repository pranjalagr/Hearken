"""When to try a call to an outside service (Google, Sarvam) again, shared by
every check and the lab's judge, so they all follow the same rules:

- retry too many requests (429), a server error (5xx) or a stuck request;
  anything else (a bad key, a wrong voice name) fails at once;
- when the service says how long to wait, honour it, but a wait too long to
  sit through (a daily limit) fails at once: on 1 Nov 2026 a judge check
  spent 60 s retrying a daily cap that had 14 hours left.

Moved here from the voice test (week 1), where these rules were first worked
out against the real libraries.
"""

from __future__ import annotations

import re
import urllib.error

import httpx

# The longest wait a service may ask for before a retry. A per-minute limit
# asks under a minute ("retry in 54s"); a daily one asks hours ("retry in
# 24h23m46s"), and waiting through 3 retries of every clip for that only
# wasted 105 s each on 1 Nov 2026.
MAX_ASKED_WAIT_S = 120.0


def timed_out(error: BaseException) -> bool:
    """A stuck request, from either service. Each one hides the timeout
    inside another error, so the chain of causes is followed:
    - urllib (Sarvam) raises TimeoutError, or URLError around it when connecting;
    - the Google library raises its own private APITimeoutError around
      httpx's TimeoutException, both before the reply and mid-stream."""
    seen: set[int] = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, (TimeoutError, httpx.TimeoutException)):
            return True
        if isinstance(error, urllib.error.URLError) and isinstance(error.reason, TimeoutError):
            return True
        error = error.__cause__ or error.__context__
    return False


def http_status(error: BaseException) -> int | None:
    """urllib (Sarvam) keeps the HTTP status in .code; the Google library in
    .status_code (its errors have no .code: a hand-made stand-in with .code
    hid this until the real library was tested); plain httpx (Groq) only on
    the error's .response."""
    for source in (error, getattr(error, "response", None)):
        for name in ("status_code", "code"):
            value = getattr(source, name, None)
            if isinstance(value, int):
                return value
    return None


def worth_retrying(error: BaseException) -> bool:
    """Too many requests (429), a server error (5xx) or a stuck request: wait
    and try again. Anything else (a wrong voice name, a bad key) fails at once."""
    if timed_out(error):
        return True
    status = http_status(error)
    return status is not None and (status == 429 or status >= 500)


# Google's free tier says how long to wait in words, inside the error text:
# "Please retry in 24h23m46s" or "Please retry in 54s" (seen 1 Nov 2026).
_RETRY_IN = re.compile(r"retry in (?=\d)(?:(\d+)h)?(?:(\d+)m)?(?:(\d+(?:\.\d+)?)s)?", re.IGNORECASE)


def asked_wait_s(error: BaseException) -> float | None:
    """How long the service itself asks us to wait, if it says: the standard
    Retry-After header (in seconds), else Google's "retry in 24h23m46s"."""
    # urllib's HTTPError keeps headers on itself; the Google library's and httpx's errors on .response.
    for source in (error, getattr(error, "response", None)):
        headers = getattr(source, "headers", None)
        value = headers.get("Retry-After") if hasattr(headers, "get") else None
        # Seconds, whole or not ("7.5"); the header's other form, a date, is skipped.
        if value is not None and re.fullmatch(r"\d+(?:\.\d+)?", str(value).strip()):
            return float(value)
    match = _RETRY_IN.search(str(error))
    if match is None or not any(match.groups()):
        return None
    hours, minutes, seconds = (float(part or 0) for part in match.groups())
    return hours * 3600 + minutes * 60 + seconds


def asks_too_long(error: BaseException, longest_s: float = MAX_ASKED_WAIT_S) -> bool:
    """A daily limit, not a per-minute one: no retry here can outlast it. Work someone
    is waiting on live passes its own, shorter limit."""
    asked = asked_wait_s(error)
    return asked is not None and asked > longest_s
