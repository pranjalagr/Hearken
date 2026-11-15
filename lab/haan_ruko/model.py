"""The haan vs ruko model: a small network over the last 2 s of the caller's sound (Part 4).

Small on purpose, like a "Hey Google" word spotter: one check must take well under 20 ms on
a normal CPU, so it can be asked every 0.1 s while the robot talks. It looks at a log-mel
picture of the sound and gives one number: how likely this is real speech, so stop.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from lab.haan_ruko.sound import FRAMES, N_MELS


def _block(inp: int, out: int, pool: bool) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Conv2d(inp, out, 3, padding=1), nn.BatchNorm2d(out), nn.ReLU()]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class HaanRuko(nn.Module):
    def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
        super().__init__()
        # The scaling travels inside the model, so the saved model needs nothing else to run.
        self.register_buffer("mean", torch.tensor(float(mean)))
        self.register_buffer("std", torch.tensor(float(std)))
        # Channels kept small: a first try with 32-64-64-96 took about 2 s per training batch
        # on this laptop's CPU; this is about 3 times less work for the same shape of model.
        self.body = nn.Sequential(_block(1, 16, True), _block(16, 32, True), _block(32, 48, True), _block(48, 64, False))
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, log_mel: torch.Tensor) -> torch.Tensor:
        """[batch, 1, N_MELS, FRAMES] -> [batch] stop logits."""
        x = self.body((log_mel - self.mean) / self.std).mean(dim=2)  # average over pitch: [batch, 96, time]
        x = torch.cat([x.mean(dim=2), x.amax(dim=2)], dim=1)  # how much, and the strongest moment
        return self.head(x).squeeze(1)


def export_onnx(model: HaanRuko, path: Path) -> None:
    model.eval()
    dummy = torch.zeros(1, 1, N_MELS, FRAMES)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, dummy, str(path), input_names=["log_mel"], output_names=["stop_logit"],
                      dynamic_axes={"log_mel": {0: "batch"}, "stop_logit": {0: "batch"}}, opset_version=17, dynamo=False)


class OnnxModel:
    """The saved model, as a live robot would run it: one thread, numpy in, a chance out."""

    def __init__(self, path: Path) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])

    def chances(self, windows: np.ndarray) -> np.ndarray:
        # In pieces: all 40,000 test windows at once ran out of memory (11 Nov 2026).
        logits = [self.session.run(None, {"log_mel": windows[at:at + 512].astype(np.float32)[:, None]})[0]
                  for at in range(0, len(windows), 512)]
        return 1.0 / (1.0 + np.exp(-np.concatenate(logits))) if logits else np.zeros(0)
