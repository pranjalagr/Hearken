"""Makes the haan vs ruko voice clips on Smallest.ai and lists them (Part 4, step 2).

Clips are saved clean, as the company sent them. The phone sound, noise, echo and speed
changes are added later, many times over and at random, while training: that costs
nothing, and a fixed rough copy would teach the model one kind of bad line only.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from pathlib import Path

from lab.haan_ruko.phrases import Plan

ROOT = Path(__file__).resolve().parents[2]
SOUNDS = ROOT / "lab" / "sounds" / "haan_ruko"


async def make_clips(plans: list[Plan], make: Callable[[Plan], Awaitable[Path]], sounds: Path = SOUNDS,
                     log: Callable[[str], None] = print, give_up_after: int = 5) -> list[dict]:
    """Every plan into a saved clip, and one list of them all. `make` returns a plan's file,
    asking the voice company only if it isn't saved yet. A failure is reported and left out,
    so a rerun fills the gaps; a run of failures (a daily limit, credit run out) stops."""
    made: list[dict] = []
    in_a_row = 0
    for plan in plans:
        try:
            path = await make(plan)
            in_a_row = 0
        except Exception as error:  # noqa: BLE001 - reported per clip; a rerun fills the gaps
            in_a_row += 1
            log(f"{plan.id}: no clip: {type(error).__name__}: {str(error)[:120]}")
            if in_a_row >= give_up_after:
                log(f"{in_a_row} failures in a row: stopping")
                break
            continue
        made.append({**asdict(plan), "file": path.relative_to(sounds).as_posix()})
        if len(made) % 100 == 0:
            log(f"{len(made)} of {len(plans)} made")
    sounds.mkdir(parents=True, exist_ok=True)
    (sounds / "clips.json").write_text(json.dumps(made, ensure_ascii=False, indent=1), encoding="utf-8")
    return made


def load_manifest(sounds: Path = SOUNDS) -> list[dict]:
    return json.loads((sounds / "clips.json").read_text(encoding="utf-8"))
