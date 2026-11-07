"""The Groq client (OpenAI-style chat API), made safe the same way as
common/google.py: a time limit on every request, and no retries of its own
(httpx never retries; the caller's retry helper does, by common/retry.py).

Plain httpx, not Groq's SDK: one POST is all the lab needs, and the SDK
would bring its own hidden retries back.
"""

from __future__ import annotations

from typing import Any

import httpx

GROQ_API = "https://api.groq.com/openai/v1"


def groq_client(key: str, timeout_s: float, transport: httpx.AsyncBaseTransport | None = None,
                verify: Any = True) -> httpx.AsyncClient:
    """The key lives only in the client's headers, so no request code ever
    handles it. transport lets tests put a fake network underneath."""
    return httpx.AsyncClient(base_url=GROQ_API, headers={"Authorization": f"Bearer {key}"},
                             timeout=timeout_s, transport=transport, verify=verify)
