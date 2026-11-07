"""The real services behind the chain caller, chosen by lab/caller/settings.yaml.

The one place the caller's parts are put together: nothing else knows which company
hears, thinks or speaks, so a part can be swapped by a setting.
"""

from __future__ import annotations

import os
import zlib
from functools import partial
from pathlib import Path
from typing import Any

from common.env import load_env, require
from common.groq import groq_client
from lab.caller.brain.brain import Brain, gemini_ask, groq_ask
from lab.caller.chain import ChainCaller, ChainParts
from lab.caller.mouth.phone_mic import make_noise_bed
from lab.caller.mouth.voice import SavedClips, SmallestVoice
from lab.caller.settings import CallerSettings, load_caller_settings
from lab.ears_test.services import Deepgram, Sarvam, shared_tls

ROOT = Path(__file__).resolve().parents[2]
CLIPS = ROOT / "lab" / "sounds" / "caller"
NOISE = ROOT / "lab" / "sounds" / "noise"
# The chosen setup for Sarvam's ears (decision 24).
SARVAM_FAST = {"silence_ms": 250, "stream_type": "fast"}
GOOGLE_MIN_TIMEOUT_S = 10.0


def _brain_client(settings: CallerSettings) -> Any:
    """One client for the whole batch. Making one takes up to 0.4 s of blocked event
    loop (measured 4 Nov 2026), which every live call would feel."""
    if settings.brain_company == "groq":
        return groq_client(require("GROQ_API_KEY"), settings.brain_timeout_s, verify=shared_tls())
    if settings.brain_company == "gemini":
        from common.google import google_client

        # Google refuses a limit under 10 s (its 400 error, 4 Nov 2026).
        return google_client(require("GEMINI_API_KEY"), max(settings.brain_timeout_s, GOOGLE_MIN_TIMEOUT_S))
    raise ValueError(f"unknown brain company {settings.brain_company!r}")


def _ears(settings: CallerSettings) -> Any:
    if settings.ears_company == "deepgram":
        return Deepgram(require("DEEPGRAM_API_KEY"))
    if settings.ears_company == "sarvam":
        return Sarvam(require("SARVAM_API_KEY"), **SARVAM_FAST)
    raise ValueError(f"unknown ears company {settings.ears_company!r}")


class RealCaller:
    """Makes a chain caller for each call, from the real services (run_call's CallerMaker).
    `brain_company` and `brain_model` swap the brain for the comparison of C3."""

    def __init__(self, settings: CallerSettings | None = None, use_noise: bool = True) -> None:
        load_env()
        # Off for puppet runs: a puppet has no real voice detector, so noise looks like
        # the caller talking and it acts on a yes nobody said (4 Nov 2026). Real robots
        # have ears, and they get the noise the task asks for.
        self._use_noise = use_noise
        self._settings = settings or load_caller_settings()
        # Before any call starts, so no live call waits for them.
        shared_tls()
        self._client = _brain_client(self._settings)

    async def warm_up(self) -> None:
        """The first request in a program loads more code and opens the first secure
        connection: on 4 Nov 2026 that froze the first call's 20 ms beat by 130 ms, and
        the lab rightly reran it. Done here, before any call is live. Asking for the
        model list uses no tokens."""
        if self._settings.brain_company == "groq":
            await self._client.get("/models")

    async def close(self) -> None:
        close = getattr(self._client, "aclose", None) or self._client.aio.aclose
        await close()

    def __call__(self, plan: Any, on_event: Any) -> ChainCaller:
        settings = self._settings
        spec = plan.task.caller
        choice = settings.voice_for(spec.voice, spec.personality)

        def used(usage: dict[str, Any]) -> None:
            on_event("caller.cost", {"service": "brain", "tokens": usage.get("total_tokens", 0)})

        def retried(error: BaseException, wait: float) -> None:
            # In the call's folder: a stuck call must show what it was waiting for.
            on_event("caller.brain_retry", {"why": str(error)[:200], "wait_s": round(wait, 1)})

        make_ask = groq_ask if settings.brain_company == "groq" else gemini_ask
        ask = make_ask(self._client, settings.brain_model, used, retried)

        def spoke(characters: int) -> None:
            on_event("caller.cost", {"service": "voice", "model": choice.model, "characters": characters})

        voice = SmallestVoice(require("SMALLEST_API_KEY"), choice.voice, choice.model, SavedClips(CLIPS), on_cost=spoke)
        make_noise = None
        noisy = self._use_noise and spec.noise is not None
        if noisy:
            # From the task's name, not the call's, so A, B and C hear the same stretch (C13).
            start = zlib.crc32(plan.task.id.encode())
            make_noise = partial(make_noise_bed, NOISE, spec.noise.kind, spec.noise.level, start)
        parts = ChainParts(
            Brain(spec, ask), voice, _ears(settings), make_noise,
            describe={"brain": f"{settings.brain_company} {settings.brain_model}", "ears": settings.ears_company,
                      "voice": f"smallest {choice.model} {choice.voice}",
                      "noise": f"{spec.noise.kind} {spec.noise.level}" if noisy else None},
        )
        return ChainCaller(parts, plan.task, on_event, settings)


def missing_keys(settings: CallerSettings) -> list[str]:
    """Names of the keys a real run needs but .env doesn't have. Keys are never printed."""
    load_env()
    needed = {"SMALLEST_API_KEY", "SARVAM_API_KEY",
              {"groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY"}[settings.brain_company],
              {"deepgram": "DEEPGRAM_API_KEY", "sarvam": "SARVAM_API_KEY"}[settings.ears_company]}
    return sorted(name for name in needed if not os.environ.get(name, "").strip())
