"""Stable, order-independent assignment of normalized prompt groups."""

import hashlib
import math

from .validate import normalize_prompt


def assign_splits(records, seed=42, ratios=(0.7, 0.15, 0.15)):
    if (len(ratios) != 3 or any(not math.isfinite(r) or r <= 0 for r in ratios)
            or not math.isclose(sum(ratios), 1.0)):
        raise ValueError("three positive finite split ratios must sum to 1")
    groups = {normalize_prompt(record["prompt"]) for record in records}
    if len(groups) < 3:
        raise ValueError("at least three distinct prompt groups required")
    ordered = sorted(groups, key=lambda group: (
        hashlib.sha256(f"{seed}\0{group}".encode("utf-8")).hexdigest(), group))
    # Preserve nonempty evaluation splits even for tiny fixtures.
    n_train = min(max(1, int(len(ordered) * ratios[0])), len(ordered) - 2)
    n_val = min(max(1, int(len(ordered) * ratios[1])), len(ordered) - n_train - 1)
    mapping = {group: "train" if i < n_train else "val" if i < n_train + n_val else "test"
               for i, group in enumerate(ordered)}
    return [{**record, "prompt_group": normalize_prompt(record["prompt"]),
             "split": mapping[normalize_prompt(record["prompt"]) ]} for record in records]
