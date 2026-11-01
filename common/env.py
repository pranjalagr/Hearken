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
