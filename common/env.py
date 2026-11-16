"""Secrets such as API keys, read from the project's .env file.

The .env file is git-ignored, so keys never reach GitHub. This reads it into
the environment without ever printing a value: error messages name the
missing variable, never its content.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ENV = Path(__file__).resolve().parents[1] / ".env"


def load_env(path: Path = PROJECT_ENV) -> None:
    """Put NAME=value lines into os.environ. A variable already set outside
    the file wins, so a shell can override the file for one run."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(name.strip(), value)


def require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is not set: put a line {name}=... in pakka/.env")
    return value


# Which .env line holds the Sarvam key for this run. The user keeps several and
# picks one per run (15 Nov 2026); robots and the pretend caller all read the
# same one, and nothing ever switches keys on its own.
SARVAM_KEY_CHOICE = "HEARKEN_SARVAM_KEY"


def sarvam_key_name() -> str:
    return os.environ.get(SARVAM_KEY_CHOICE, "").strip() or "SARVAM_API_KEY"


def sarvam_key() -> str:
    return require(sarvam_key_name())


# The same for the robots' Gemini key: the user picks which .env line a run uses
# (15 Nov 2026), and nothing switches keys on its own.
GEMINI_KEY_CHOICE = "HEARKEN_GEMINI_KEY"


def gemini_key_name() -> str:
    return os.environ.get(GEMINI_KEY_CHOICE, "").strip() or "GEMINI_API_KEY"


def gemini_key() -> str:
    return require(gemini_key_name())


# Every working Gemini key, by .env name, comma-separated: set by the lab when the key
# order rotates Gemini (lab/key_order.yaml, the owner's choice of 16 Nov 2026).
GEMINI_POOL_CHOICE = "HEARKEN_GEMINI_KEYS"


def gemini_pool() -> list[str]:
    """The keys to take turns over; just the chosen key when no pool is set."""
    names = [name.strip() for name in os.environ.get(GEMINI_POOL_CHOICE, "").split(",") if name.strip()]
    return [require(name) for name in names] or [gemini_key()]


# The same for the Groq key (the pretend caller's brain and the judge): picked per run
# by the user (15 Nov 2026), never switched by code.
GROQ_KEY_CHOICE = "HEARKEN_GROQ_KEY"


def groq_key_name() -> str:
    return os.environ.get(GROQ_KEY_CHOICE, "").strip() or "GROQ_API_KEY"


def groq_key() -> str:
    return require(groq_key_name())
