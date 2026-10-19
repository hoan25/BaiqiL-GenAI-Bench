"""Run an offline CLIP CPU smoke check using an existing model directory/cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def select_device(requested: str, cuda_available: bool) -> str:
    """Use CPU fallback only for auto; explicit CUDA must actually be available."""
    if requested not in {"cpu", "cuda", "auto"}:
        raise ValueError(f"Unsupported device: {requested}")
    if requested == "cuda" and not cuda_available:
        raise ValueError("CUDA requested but unavailable")
    return ("cuda" if cuda_available else "cpu") if requested == "auto" else requested


def smoke_clip(model_name: str, requested_device: str = "cpu") -> dict:
    """Check real pretrained inference; never download weights implicitly."""
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    device = select_device(requested_device, torch.cuda.is_available())
    processor = CLIPProcessor.from_pretrained(model_name, local_files_only=True, use_fast=False)
    model = CLIPModel.from_pretrained(model_name, local_files_only=True).to(device).eval()
    # Ảnh tổng hợp chỉ kiểm tra pipeline, không dùng để báo cáo chất lượng mô hình.
    image = Image.new("RGB", (224, 224), (255, 0, 0))
    inputs = processor(text=["a red square"], images=[image], return_tensors="pt", padding=True)
    inputs = inputs.to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    for name in ("image_embeds", "text_embeds", "logits_per_image"):
        if not torch.isfinite(getattr(outputs, name)).all():
            raise ValueError(f"Non-finite output: {name}")
    if outputs.image_embeds.shape != outputs.text_embeds.shape:
        raise ValueError("Text/image embedding shapes differ")
    if outputs.logits_per_image.shape != (1, 1):
        raise ValueError("Expected one image-text pair")
    return {
        "status": "PASS",
        "device": device,
        "requested_device": requested_device,
        "cuda_available": torch.cuda.is_available(),
        "model": model_name,
        "embedding_shape": list(outputs.image_embeds.shape),
        "fixture": "synthetic red square, 224x224; pipeline check only",
        "scope": "Environment inference smoke; not a quality metric or score calibration",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    args = parser.parse_args()
    try:
        result = smoke_clip(args.model, args.device)
    except OSError as exc:
        result = {"status": "BLOCKED", "error": str(exc), "model": args.model}
    except Exception as exc:
        result = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}", "model": args.model}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return {"PASS": 0, "FAIL": 1, "BLOCKED": 2}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
