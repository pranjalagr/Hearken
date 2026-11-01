"""The Google AI client, made safe for anything that must finish: a time
limit on every request, and no retries of its own.

Checked against the library (2.34) with a fake network, 1 Nov 2026:
- by default it never times out, and quietly retries a 503 three times
  (4 requests, 6 s of hidden waiting) that would count as a slow first
  sound;
- its retry setting can't say "never" for interactions: attempts=0 is
  turned into 1, which the interactions client reads as one retry.
So retries are switched off on the interactions client itself, and left to
the caller's own retry helper, which restarts the clock and logs each one.

The same day, a lab judge check on the library's default client sat on one
open request for 11 minutes with nothing to show, so every Google client in
this project comes from here, each with the time limit its job needs.
"""

from __future__ import annotations

from typing import Any

import httpx


def google_client(key: str, timeout_s: float, http_client: httpx.Client | None = None) -> Any:
    """timeout_s is the caller's policy (a voice clip needs far less than a
    judge that thinks). http_client lets tests put a fake network under the
    real library."""
    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key=key,
        http_options=types.HttpOptions(timeout=int(timeout_s * 1000), httpx_client=http_client),
    )
    # The interactions client reads this on every call; None means never retry.
    client.interactions.sdk_configuration.retry_config = None
    return client
