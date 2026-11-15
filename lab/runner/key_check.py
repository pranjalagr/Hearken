"""Before a batch with AI robots: does every service accept the chosen keys?

On 15 Nov 2026 a key whose project had lost Gemini's free tier ran 9 calls in
which the robot never had a brain, while each call still spent Sarvam credit
and 5 minutes. One tiny request per key before the batch costs almost nothing
and stops that at once. Keys are read here and never printed: a refusal names
the .env line, not the key.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from collections.abc import Callable

from common.env import gemini_key, gemini_key_name, groq_key, groq_key_name, load_env, sarvam_key, sarvam_key_name

TIMEOUT_S = 20.0


def gemini_accepts(model: str, timeout_s: float = TIMEOUT_S) -> str | None:
    """None if the robots' Gemini key answers; otherwise why not."""
    from google.genai import types

    from common.google import google_client

    # Kept in a name, not chained: a client used only inside one expression was closed
    # before its request went out ("client has been closed", 15 Nov 2026).
    client = google_client(gemini_key(), timeout_s)
    try:
        client.models.generate_content(model=model, contents="ok", config=types.GenerateContentConfig(max_output_tokens=1))
    except Exception as exc:  # any refusal: the message says which (403, 429, bad key)
        return f"{gemini_key_name()}: {str(exc)[:200]}"
    return None


def sarvam_accepts(timeout_s: float = TIMEOUT_S) -> str | None:
    """None if the Sarvam key answers; otherwise why not. Speaks one word, the voice
    test's own proven request: about 2 characters of credit."""
    from lab.week1.voice_test import SETUPS_BY_NAME, VoiceLine, sarvam_request

    request = sarvam_request(SETUPS_BY_NAME["sarvam-sunny-enhi"], VoiceLine("key-check", "check", "hi", "ok", ""), sarvam_key())
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as reply:
            reply.read(64)
    except urllib.error.HTTPError as exc:
        return f"{sarvam_key_name()}: HTTP {exc.code} {exc.read(300).decode(errors='replace')}"
    except OSError as exc:
        return f"{sarvam_key_name()}: {exc!r}"
    return None


def groq_accepts(model: str, timeout_s: float = TIMEOUT_S) -> str | None:
    """None if the pretend caller's Groq key answers; otherwise why not. Asks for one
    token, so a key that is full for the day (tokens per day) is caught too."""
    import httpx

    try:
        reply = httpx.post("https://api.groq.com/openai/v1/chat/completions", timeout=timeout_s,
                           headers={"Authorization": f"Bearer {groq_key()}"},
                           json={"model": model, "messages": [{"role": "user", "content": "ok"}], "max_tokens": 1})
    except httpx.HTTPError as exc:
        return f"{groq_key_name()}: {exc!r}"
    if reply.status_code != 200:
        return f"{groq_key_name()}: HTTP {reply.status_code} {reply.text[:200]}"
    return None


def refused_keys(brain_model: str, checks: tuple[Callable[[], str | None], ...] | None = None,
                 caller_model: str | None = None) -> list[str]:
    """Every key the services refuse, by its .env name. Empty: the batch may start."""
    load_env()
    if checks is None:
        checks = (lambda: gemini_accepts(brain_model), sarvam_accepts)
        if caller_model:
            checks += (lambda: groq_accepts(caller_model),)
    return [problem for check in checks if (problem := check()) is not None]
