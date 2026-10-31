"""Fusion of a 1-5 regression score with a zero-shot match probability (ITM/VQA P(match)).

Pure functions; no models. ``gate`` keeps the regression score where the matcher
agrees the image shows the prompt and pulls it toward 1 where it does not.
"""

import numpy as np

from src.evaluation.metrics import pairwise_accuracy, regression_metrics


def _pair(score, match):
    score, match = np.asarray(score, dtype=float), np.asarray(match, dtype=float)
    if score.shape != match.shape or score.ndim != 1 or score.size == 0:
        raise ValueError("score and match must be equal nonempty 1-D arrays")
    if not np.isfinite(score).all() or not np.isfinite(match).all():
        raise ValueError("nonfinite inputs")
    if np.any((score < 1) | (score > 5)) or np.any((match < 0) | (match > 1)):
        raise ValueError("score must be in [1,5] and match in [0,1]")
    return score, match


RULES = {
    "mlp": lambda s, m: s,
    "gate_multiply": lambda s, m: 1 + (s - 1) * m,
    "gate_min": lambda s, m: np.minimum(s, 1 + 4 * m),
    # Softened gates: real GenAI-Bench pairs with low ITM average ~2.5, not 1, so a hard gate overshoots.
    "gate_sqrt": lambda s, m: 1 + (s - 1) * np.sqrt(m),
    "gate_soft25": lambda s, m: 1 + (s - 1) * (0.25 + 0.75 * m),
    "match_only": lambda s, m: 1 + 4 * m,
}


def fuse(rule, score, match):
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule}")
    score, match = _pair(score, match)
    return RULES[rule](score, match)


def evaluate_rules(target, score, match, groups, swap_score=None, swap_match=None):
    """Metrics per rule on original pairs plus prompt-swap sensitivity when swapped inputs are given."""
    target = np.asarray(target, dtype=float)
    report = {}
    for rule in RULES:
        fused = fuse(rule, score, match)
        entry = {**regression_metrics(target, fused), "pairwise": pairwise_accuracy(target, fused, groups)}
        if swap_score is not None:
            swapped = fuse(rule, swap_score, swap_match)
            entry["swap"] = {"original_wins": int((fused > swapped).sum()), "count": int(len(fused)),
                             "mean_drop": float((fused - swapped).mean())}
        report[rule] = entry
    return report
