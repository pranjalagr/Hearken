"""The Sarvam chat client, made safe the same way as common/groq.py: a time limit on
every request, and no retries of its own (the caller's retry helper does those, by
common/retry.py). Plain httpx, for the same reason: one POST is all the lab needs.
"""

from __future__ import annotations

from typing import Any

import httpx

SARVAM_API = "https://api.sarvam.ai/v1"


def sarvam_client(key: str, timeout_s: float, transport: httpx.AsyncBaseTransport | None = None,
                  verify: Any = True) -> httpx.AsyncClient:
    """The key lives only in the client's headers, so no request code ever handles it.
    transport lets tests put a fake network underneath."""
    return httpx.AsyncClient(base_url=SARVAM_API, headers={"api-subscription-key": key},
                             timeout=timeout_s, transport=transport, verify=verify)
