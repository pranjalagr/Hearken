"""Every pack the lab can run, by name.

The only place that knows all packs, so the engine never imports a pack.
Adding a pack means adding one line here, and nothing in engine/.
"""

from __future__ import annotations

from engine.contract.pack import Pack
from packs.food.pack import FoodPack
from packs.toy.pack import ToyPack

PACKS: dict[str, type[Pack]] = {
    FoodPack.name: FoodPack,
    ToyPack.name: ToyPack,
}


def get_pack(name: str) -> Pack:
    try:
        return PACKS[name]()
    except KeyError:
        raise ValueError(f"unknown pack {name!r}; known: {', '.join(PACKS)}") from None
