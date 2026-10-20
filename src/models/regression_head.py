"""CPU scoring heads with train-fitted standardization stored in state_dict."""

from pathlib import Path

import numpy as np
import torch
from torch import nn


def features(text, image):
    text, image = np.asarray(text, dtype=np.float32), np.asarray(image, dtype=np.float32)
    if text.ndim != 2 or text.shape != image.shape or text.shape[1] == 0:
        raise ValueError("text/image must have matching [N,D] shapes")
    result = np.concatenate([text, image], axis=1)
    if not np.isfinite(result).all():
        raise ValueError("features must be finite")
    return result


class RegressionHead(nn.Module):
    def __init__(self, dimension, kind="mlp", hidden=64):
        super().__init__()
        if dimension < 1 or hidden < 1 or kind not in {"mean", "ridge", "mlp"}:
            raise ValueError("invalid head configuration")
        self.config = {"dimension": dimension, "kind": kind, "hidden": hidden}
        self.register_buffer("center", torch.zeros(dimension))
        self.register_buffer("scale", torch.ones(dimension))
        self.network = (nn.Sequential(nn.Linear(dimension, hidden), nn.ReLU(), nn.Linear(hidden, 1))
                        if kind == "mlp" else nn.Linear(dimension, 1))

    def forward(self, x):
        raw = self.network((x - self.center) / self.scale).squeeze(-1)
        return 1 + 4 * torch.sigmoid(raw) if self.config["kind"] == "mlp" else raw.clamp(1, 5)

    def predict(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.config["dimension"] or not np.isfinite(x).all():
            raise ValueError("invalid prediction features")
        self.eval()
        with torch.inference_mode():
            result = self(torch.from_numpy(x)).numpy()
        if not np.isfinite(result).all():
            raise ValueError("nonfinite prediction")
        return result


def save_head(model, path):
    with Path(path).open("xb") as stream:
        torch.save({"format_version": 1, "config": model.config, "state_dict": model.state_dict()}, stream)


def load_head(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["format_version"] != 1:
        raise ValueError("unsupported checkpoint format")
    model = RegressionHead(**payload["config"])
    model.load_state_dict(payload["state_dict"], strict=True)
    if not all(torch.isfinite(v).all() for v in model.state_dict().values()) or (model.scale <= 0).any():
        raise ValueError("invalid checkpoint weights/scaler")
    return model.eval()
