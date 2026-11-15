"""The end-of-turn models under test, each run exactly as its own model card says.

Only the ONNX files are downloaded: an ONNX file is data, while a pickled PyTorch file
can run code when opened, and one of these repositories offers both.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

MODEL_RATE = 16_000
WINDOW = 8 * MODEL_RATE  # all three read at most the last 8 seconds
Model = Callable[[np.ndarray], float]  # 16 kHz audio -> chance the turn is finished


@dataclass(frozen=True)
class ModelSpec:
    name: str
    repo: str
    file: str
    takes_raw_audio: bool  # True: the ONNX file works out its own sound features
    pad_front: bool  # where silence goes in a clip shorter than the window


# hinglish-turn-v2's card pads at the end, while the other two pad at the front, where
# silence is furthest from the moment being judged. Run both ways so that a padding
# mismatch on our side can't be what makes it lose.
SPECS = (
    ModelSpec("smart-turn-v3.2", "soniqo/Smart-Turn-v3.2-ONNX", "smart-turn-v3.2.onnx", True, True),
    ModelSpec("hinglish-turn-v2", "Shrey160/hinglish-turn-v2", "onnx/s2-004.fp32.onnx", False, False),
    ModelSpec("hinglish-turn-v2 (front pad)", "Shrey160/hinglish-turn-v2", "onnx/s2-004.fp32.onnx", False, True),
    ModelSpec("smart-turn-hinglish", "abhishek-040010/smart-turn-hinglish",
              "smart-turn-v3-hinglish-whisper-tiny-attention.onnx", False, True),
)


def window_of(audio: np.ndarray, pad_front: bool) -> np.ndarray:
    """The last 8 s of the turn, with silence added if it's shorter."""
    tail = np.asarray(audio[-WINDOW:], dtype=np.float32)
    padding = np.zeros(WINDOW - len(tail), dtype=np.float32)
    return np.concatenate([padding, tail] if pad_front else [tail, padding])


def load_model(spec: ModelSpec) -> Model:
    from huggingface_hub import hf_hub_download
    import onnxruntime as ort
    from transformers import WhisperFeatureExtractor

    options = ort.SessionOptions()
    # One thread: many calls share the machine in the real system, so a time measured
    # with every core free would flatter the model.
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(hf_hub_download(spec.repo, spec.file), options, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    features = WhisperFeatureExtractor(chunk_length=8)

    def chance(audio: np.ndarray) -> float:
        window = window_of(audio, spec.pad_front)
        if spec.takes_raw_audio:
            return float(np.ravel(session.run(None, {input_name: window[None]})[0])[0])
        mel = features(window, sampling_rate=MODEL_RATE, return_tensors="np", do_normalize=False).input_features
        logit = float(np.ravel(session.run(None, {input_name: mel.astype(np.float32)})[0])[0])
        return float(1.0 / (1.0 + np.exp(-logit)))

    return chance
