"""Validation-only M0/M1 evaluation. No test-set evaluation during tuning."""

import numpy as np

from src.data.validate import normalize_prompt
from .metrics import pairwise_accuracy, regression_metrics, spearman


def evaluate_baselines(records, cosine_by_id):
    if not records or len({r["pair_id"] for r in records}) != len(records):
        raise ValueError("manifest must be nonempty with unique pair IDs")
    groups = {}
    for record in records:
        split = record["split"]
        if split not in {"train", "val", "test"}:
            raise ValueError("unknown split")
        group = normalize_prompt(record["prompt"])
        if group in groups and groups[group] != split:
            raise ValueError("prompt leakage between splits")
        groups[group] = split
    train = [r for r in records if r["split"] == "train"]
    val = [r for r in records if r["split"] == "val"]
    if not train or not val:
        raise ValueError("nonempty train and validation required")
    # Do not read test labels or predictions.
    train_target = np.asarray([r["target"] for r in train], dtype=float)
    target = np.asarray([r["target"] for r in val], dtype=float)
    if (not np.isfinite(train_target).all() or not np.isfinite(target).all()
            or np.any((train_target < 1) | (train_target > 5))
            or np.any((target < 1) | (target > 5))):
        raise ValueError("ratings must be finite and in [1, 5]")
    mean = float(train_target.mean())
    prediction = np.full(len(val), mean)
    cosine = np.asarray([cosine_by_id[r["pair_id"]] for r in val], dtype=float)
    if not np.isfinite(cosine).all() or np.any(np.abs(cosine) > 1 + 1e-6):
        raise ValueError("invalid cosine")
    val_groups = [normalize_prompt(r["prompt"]) for r in val]
    return {"split": "val", "train_count": len(train), "val_count": len(val),
            "M0": {"train_mean": mean, **regression_metrics(target, prediction),
                   "pairwise": pairwise_accuracy(target, prediction, val_groups)},
            "M1": {"spearman": spearman(target, cosine),
                   "pairwise": pairwise_accuracy(target, cosine, val_groups),
                   "score_type": "raw_cosine_not_1_to_5"}}
