"""Serving predictor that fuses the CLIP regression head with a BLIP image-text-match gate.

Rule ``gate_sqrt`` (selected in
project_records/reports/results/G7_itm_gate_full_20261030.md):
``score = 1 + (mlp - 1) * sqrt(itm)`` with ``itm = P(match)`` from the ITM head of
``Salesforce/blip-itm-base-coco``. The BLIP artifact is pinned by a receipt JSON
(revision + sha256 of every file); loading refuses anything that differs.
BLIP is English-only, so non-English prompts skip the gate and keep the raw score.
"""

import json
import os
from pathlib import Path

import numpy as np
import torch

from src.data.qa import sha256_file
from src.evaluation.itm_gate import fuse
from src.inference.language import gate_supported
from src.inference.predictor import Predictor
from src.models.regression_head import features

RULE = "gate_sqrt"
NON_ENGLISH_WARNING = "Prompt không phải tiếng Anh; chưa kiểm tra khớp prompt, điểm chỉ từ mô hình hồi quy."


def hf_cache_root():
    """Hub cache directory honouring HF_HUB_CACHE / HF_HOME like huggingface_hub does."""
    explicit = os.getenv("HF_HUB_CACHE")
    if explicit:
        return Path(explicit)
    return Path(os.getenv("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"


def resolve_snapshot(model, revision=None):
    """Return (snapshot_dir, revision) for a local directory or a cached Hub model id.

    Nothing is downloaded: a Hub id is resolved through ``refs/main`` of the local cache.
    """
    path = Path(model)
    if path.is_dir():
        return path.resolve(), revision or path.resolve().name
    if model.count("/") != 1 or any(part.startswith(".") for part in model.split("/")):
        raise ValueError("itm_model must be a local directory or an org/name Hub id")
    repo = hf_cache_root() / f"models--{model.replace('/', '--')}"
    if revision is None:
        ref = repo / "refs" / "main"
        if not ref.is_file():
            raise ValueError(f"{model} is not in the local Hugging Face cache")
        revision = ref.read_text(encoding="utf-8").strip()
    snapshot = repo / "snapshots" / revision
    if not snapshot.is_dir():
        raise ValueError(f"snapshot {revision} of {model} is missing from the cache")
    return snapshot.resolve(), revision


def snapshot_checksums(snapshot):
    """sha256 of every regular file (symlinks followed) directly inside the snapshot."""
    files = sorted(p for p in Path(snapshot).iterdir() if p.is_file())
    if not files:
        raise ValueError("snapshot contains no files")
    return {p.name: sha256_file(p) for p in files}


def verify_receipt(receipt, snapshot, revision):
    """Raise unless the snapshot is byte-identical to what the receipt reviewed."""
    if receipt.get("rule") != RULE:
        raise ValueError("receipt rule differs from the serving rule")
    if receipt.get("revision") != revision:
        raise ValueError("BLIP revision differs from the reviewed receipt")
    expected = receipt.get("files_sha256") or {}
    if not expected:
        raise ValueError("receipt lists no files")
    actual = snapshot_checksums(snapshot)
    if set(actual) != set(expected):
        raise ValueError("BLIP snapshot file set differs from the reviewed receipt")
    for name, digest in expected.items():
        if actual[name] != digest:
            raise ValueError(f"BLIP file {name} differs from the reviewed receipt")


class GatedPredictor(Predictor):
    def __init__(self, model_path, training_path, itm_model, itm_receipt, allow_development=False, device="cpu"):
        super().__init__(model_path, training_path, allow_development)
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        receipt_path = Path(itm_receipt)
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        snapshot, revision = resolve_snapshot(itm_model, receipt.get("revision"))
        verify_receipt(receipt, snapshot, revision)
        from transformers import BlipForImageTextRetrieval, BlipProcessor
        self.itm_device = device
        self.itm_processor = BlipProcessor.from_pretrained(snapshot, local_files_only=True)
        self.itm_model = BlipForImageTextRetrieval.from_pretrained(snapshot, local_files_only=True).to(device).eval()
        self.itm_model.requires_grad_(False)
        self.itm_revision = revision
        self.rule = RULE
        self.version = f"{self.version}-{RULE}-{revision[:12]}"
        self.warnings = [*self.warnings,
                         "Điểm đã nhân với mức khớp prompt (BLIP ITM); prompt không phải tiếng Anh sẽ bỏ qua bước này."]
        self.metrics = {**self.metrics, "gate": {"rule": RULE, "itm_model": str(itm_model), "revision": revision,
                                                 "source_report": receipt.get("source_report")}}

    def match_probability(self, prompt, images):
        inputs = self.itm_processor(images=[i.convert("RGB") for i in images], text=[prompt] * len(images),
                                    return_tensors="pt", padding=True, truncation=True).to(self.itm_device)
        with torch.inference_mode():
            logits = self.itm_model(**inputs, use_itm_head=True).itm_score.float()
        itm = torch.softmax(logits, -1)[:, 1].cpu().numpy()
        if itm.shape != (len(images),) or not np.isfinite(itm).all():
            raise ValueError("invalid ITM output")
        return np.clip(itm, 0, 1)

    def predict(self, prompt, images):
        gated = gate_supported(prompt)
        with self.lock:
            encoded = self.encoder.encode([prompt] * len(images), images)
            mlp = np.clip(self.head.predict(features(encoded["text"], encoded["image"])), 1, 5)
            itm = self.match_probability(prompt, images) if gated else None
        if gated:
            scores = fuse(RULE, mlp, itm)
            return [{"score": float(s), "cosine": float(c), "mlp": float(m), "itm": float(p), "gated": True}
                    for s, c, m, p in zip(scores, encoded["cosine"], mlp, itm)]
        return [{"score": float(m), "cosine": float(c), "mlp": float(m), "itm": None, "gated": False,
                 "warning": NON_ENGLISH_WARNING}
                for m, c in zip(mlp, encoded["cosine"])]
