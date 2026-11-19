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

# Gemini refuses a request whose deadline is under 10 s ("Manually set deadline 8s is too
# short. Minimum allowed deadline is 10s", 400). A caller that needs a shorter limit (Agent
# C's 8 s Thinker step, 1 s Guard judge; 16 Nov 2026) gives the client this and holds its
# own limit with asyncio.wait_for.
GOOGLE_MIN_TIMEOUT_S = 10.0


class ClientRing:
    """Several Gemini clients, each on its own key, taking turns: every `.aio` use hands
    out the next one. The owner's choice (16 Nov 2026): Agent C's Talker, Thinker and
    judge sent 19 requests in 75 s on one key, over the free tier's 15 a minute, so the
    Thinker's and judge's requests go round the owner's keys. Only the client changes per
    request; the Thinker keeps its own history, so nothing is lost between keys."""

    def __init__(self, clients: list[Any], names: list[str] | None = None) -> None:
        if not clients:
            raise ValueError("a client ring needs at least one client")
        self._clients = clients
        # Each client's key by its .env name, for the AI request log (18 Nov 2026): which key
        # was slow. Never the key itself.
        self._names = list(names or [])
        self._next = 0

    @property
    def aio(self) -> Any:
        return self.take()[1].aio

    def take(self) -> tuple[str, Any]:
        """The next client, with its key's .env name ("" when not named)."""
        at = self._next % len(self._clients)
        self._next += 1
        return (self._names[at] if at < len(self._names) else ""), self._clients[at]


def take_client(client: Any) -> tuple[str, Any]:
    """(key name, client) from a ClientRing, or ("", client) from a plain client."""
    return client.take() if isinstance(client, ClientRing) else ("", client)


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
