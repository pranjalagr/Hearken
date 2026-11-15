"""Training the haan vs ruko model (Part 4, step 6).

Splits, so no score comes from something the model learned on:
- train: every voice but one, phrases not held back
- check: the left-out voice, same phrases: picks the best round and the line
- test: the held-back phrases, in all voices: phrases the model never heard at all
The user's own voice is a fourth set, tested later and never trained on.

Each round, every training clip is put on a new random rough line, so the model sees each
"haan" in many forms. Nothing here is random from run to run: a fixed seed.
"""

from __future__ import annotations

import io
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile
import torch
from torch import nn

from lab.haan_ruko.model import HaanRuko
from lab.haan_ruko.score import Played, ask, hear, mark, separation, zero_miss_line
from lab.haan_ruko.sound import TEST_LINES, Noises
from lab.haan_ruko.speech import Timing
from lab.haan_ruko.windows import Helpers, training_windows

CHECK_VOICE = "inaaya"
SEED = 2026_11_10
# A short "nahi" missed on the user's voice (14 Nov 2026), and such pieces are rare: the user
# agreed to practise them more. 4 is a first try, judged by the test on the user's voice.
NO_PRACTICE = 4
# Clips trained on per round. With all 82 IndicVoices files the training set no longer fits
# one round's windows in this laptop's memory, so each round takes a fresh random share of
# this size: about v3's whole training set (12,225 clips), which ran in about a minute a round
# (14 Nov 2026). A smaller set is used whole, as before.
CLIPS_PER_ROUND = 12_000


def round_share(train: list, rng: np.random.Generator, count: int = CLIPS_PER_ROUND) -> list:
    """The clips one round trains on: all of them when they fit, else a random `count`."""
    if len(train) <= count:
        return train
    return [train[i] for i in sorted(rng.choice(len(train), count, replace=False))]


@dataclass
class Sets:
    train: list[tuple[dict, Timing, np.ndarray]]
    check: list[tuple[dict, Timing, np.ndarray]]
    test: list[tuple[dict, Timing, np.ndarray]]
    # Real people (IndicVoices), split by speaker: never mixed with the machine voices' sets,
    # so each result says plainly which kind of voice it is about.
    real_check: list[tuple[dict, Timing, np.ndarray]] = field(default_factory=list)
    real_test: list[tuple[dict, Timing, np.ndarray]] = field(default_factory=list)


def add_real(sets: Sets, pieces: list[tuple[dict, Timing, np.ndarray]]) -> Sets:
    for item in pieces:
        role = item[0]["role"]
        (sets.train if role == "train" else sets.real_check if role == "check" else sets.real_test).append(item)
    return sets


def practise_more(train: list[tuple[dict, Timing, np.ndarray]], which: Callable[[dict], bool], times: int) -> int:
    """Adds `times - 1` more copies of the training clips `which` picks, so each round puts
    each of them on several different rough lines. Returns how many clips were picked."""
    picked = [item for item in train if which(item[0])]
    train += picked * (times - 1)
    return len(picked)


def split(clips: list[dict], timings: dict[str, Timing], sounds: Path) -> Sets:
    sets = Sets([], [], [])
    for clip in clips:
        timing = timings.get(clip["id"])
        if timing is None or timing.why_left_out or timing.start is None:
            continue
        samples, _ = soundfile.read(io.BytesIO((sounds / clip["file"]).read_bytes()), dtype="float32", always_2d=True)
        item = (clip, timing, samples.mean(axis=1))
        if clip["held_back"]:
            sets.test.append(item)
        elif clip["voice"] == CHECK_VOICE:
            sets.check.append(item)
        else:
            sets.train.append(item)
    return sets


def pick_device() -> torch.device:
    """The GPU when this computer's torch can use it, else the CPU, so the same code trains
    anywhere. On the CPU a training batch took most of a round (11 Nov 2026)."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        # A real tensor, not just is_available(): on 11 Nov 2026 that said yes, then the first
        # tensor failed with "CUDA-capable device(s) is/are busy or unavailable".
        torch.zeros(1, device="cuda")
        return torch.device("cuda")
    except RuntimeError:
        return torch.device("cpu")


def chances_with(model: HaanRuko) -> Callable[[np.ndarray], np.ndarray]:
    def chances(windows: np.ndarray) -> np.ndarray:
        model.eval()
        device = next(model.parameters()).device  # wherever the model is training
        out = []
        with torch.no_grad():
            for at in range(0, len(windows), 512):
                batch = torch.from_numpy(np.ascontiguousarray(windows[at:at + 512])).to(device)[:, None]
                out.append(torch.sigmoid(model(batch)).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    return chances


def judge(played: list[Played]) -> dict:
    line = zero_miss_line(played)
    got = mark(played, line)
    return {"line": line, "wrong_stops": got.wrong_stops, "keep": got.keep, "wrong_share": round(got.wrong_share, 4),
            "separation": round(separation(played), 4)}


def train(sets: Sets, noises: Noises, rounds: int, log: Callable[[str], None] = print,
          helpers: Helpers | None = None) -> tuple[HaanRuko, list[dict]]:
    """`helpers`, when given, make each round's windows on the other cores."""
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(8)
    device = pick_device()
    # Every window has the same shape, so cuDNN may time its ways of doing a convolution once
    # and keep the fastest. The GPU's sums are not bit-for-bit the same from run to run anyway.
    torch.backends.cudnn.benchmark = device.type == "cuda"
    log(f"training on {torch.cuda.get_device_name(device) if device.type == 'cuda' else 'the CPU'}")
    check = hear(sets.check + sets.real_check, TEST_LINES, noises)  # once: the test lines never change
    x, y = training_windows(round_share(sets.train, rng), noises, rng, helpers=helpers)
    model = HaanRuko(float(x.mean()), float(x.std())).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    history: list[dict] = []
    best: tuple[float, float] | None = None
    best_state = None
    for round_ in range(1, rounds + 1):
        began = time.perf_counter()
        if round_ > 1:
            # The last round's windows go first: two rounds' at once would not fit in this laptop's memory.
            x = y = None
            x, y = training_windows(round_share(sets.train, rng), noises, rng, helpers=helpers)
        # Real speech has far more windows than listening sounds: weigh the two kinds equally.
        loss_of = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((y == 0).sum() / max(1.0, (y == 1).sum()), device=device))
        model.train()
        order = rng.permutation(len(x))
        # Summed on the device: float(loss) every batch would make the CPU wait for the GPU each time.
        total = torch.zeros((), device=device)
        for at in range(0, len(order), 128):
            pick = order[at:at + 128]
            optimiser.zero_grad()
            loss = loss_of(model(torch.from_numpy(x[pick]).to(device)[:, None]), torch.from_numpy(y[pick]).to(device))
            loss.backward()
            optimiser.step()
            total += loss.detach() * len(pick)
        mean_loss = float(total) / len(x)
        score = judge(ask(check, chances_with(model)))
        history.append({"round": round_, "windows": len(x), "loss": round(mean_loss, 4), **score})
        log(f"round {round_}: loss {mean_loss:.3f}, left-out voice: wrong stops {score['wrong_stops']}/{score['keep']} "
            f"at the line that misses no speech, separation {score['separation']:.3f} ({time.perf_counter() - began:.0f} s)")
        # The best round is the one that best tells speech from listening on unseen clips.
        # Wrong stops at the no-miss line came first until 11 Nov 2026, but that line is set by
        # the single hardest clip, so it jumped about (1,202 to 1,281 of 1,284) without a trend,
        # and it kept round 6 (separation 0.910) over round 11 (0.934).
        mine = (-score["separation"], score["wrong_share"])
        if best is None or mine < best:
            best, best_state = mine, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    # Back on the CPU, where saving, the ONNX export and a live robot all run it.
    return model.cpu(), history


def save(model: HaanRuko, history: list[dict], folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), folder / "model.pt")
    (folder / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")


def load(folder: Path) -> HaanRuko:
    state = torch.load(folder / "model.pt")
    model = HaanRuko(float(state["mean"]), float(state["std"]))
    model.load_state_dict(state)
    model.eval()
    return model
