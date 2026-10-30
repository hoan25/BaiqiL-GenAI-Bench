"""Offline numerical checks of the reusable encoder using synthetic images."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from src.models.clip_encoder import ClipEncoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    encoder = ClipEncoder(args.model)
    before = encoder.fingerprint()
    prompts = ["a red square", "a blue square " * 100]
    images = [Image.new("RGB", (32, 32), c) for c in ("red", "blue")]
    batch = encoder.encode(prompts, images)
    errors = []
    for i in range(2):
        single = encoder.encode([prompts[i]], [images[i]])
        for name in batch:
            delta = float(np.max(np.abs(single[name][0] - batch[name][i])))
            errors.append(delta)
            assert delta < 1e-4, (name, delta)
    assert batch["text"].shape == batch["image"].shape == (2, 512)
    assert np.all(np.abs(batch["cosine"]) <= 1)
    for name in ("text", "image"):
        np.testing.assert_allclose(np.linalg.norm(batch[name], axis=1), 1, atol=1e-5)
    assert encoder.fingerprint() == before
    assert not any(p.requires_grad for p in encoder.model.parameters())
    assert all(p.grad is None for p in encoder.model.parameters())
    report = {"status": "PASS", "scope": "synthetic_encoder_checks_only", "device": "cpu",
              "shape": [2, 512], "max_batch_single_delta": max(errors),
              "model_state_sha256": before, "frozen_unchanged": True,
              "long_prompt_truncation": True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
