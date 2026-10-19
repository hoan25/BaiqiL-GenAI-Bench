"""Numerical metrics; cosine is never treated as a calibrated rating."""

from collections import defaultdict
from itertools import combinations
import math

import numpy as np


def paired_arrays(target, prediction):
    target, prediction = np.asarray(target, dtype=float), np.asarray(prediction, dtype=float)
    if target.ndim != 1 or target.size == 0 or target.shape != prediction.shape:
        raise ValueError("expected equal nonempty one-dimensional arrays")
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError("metric inputs must be finite")
    return target, prediction


def average_ranks(values):
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2 + 1
        start = stop
    return ranks


def spearman(target, prediction):
    target, prediction = paired_arrays(target, prediction)
    left, right = average_ranks(target), average_ranks(prediction)
    left, right = left - left.mean(), right - right.mean()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator == 0:
        return None
    return float(np.clip(np.dot(left, right) / denominator, -1, 1))


def regression_metrics(target, prediction):
    target, prediction = paired_arrays(target, prediction)
    error = target - prediction
    return {"count": len(target), "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.square(error).mean())),
            "spearman": spearman(target, prediction)}


def pairwise_accuracy(target, prediction, groups, min_gap=0.5):
    target, prediction = paired_arrays(target, prediction)
    if len(groups) != len(target) or not math.isfinite(min_gap) or min_gap < 0:
        raise ValueError("invalid groups or minimum label gap")
    indices = defaultdict(list)
    for i, group in enumerate(groups):
        indices[group].append(i)
    correct = eligible = label_ties = prediction_ties = 0
    for members in indices.values():
        for left, right in combinations(members, 2):
            truth = target[left] - target[right]
            if truth == 0 or abs(truth) < min_gap:
                label_ties += 1
                continue
            eligible += 1
            predicted = prediction[left] - prediction[right]
            prediction_ties += int(predicted == 0)
            correct += int(predicted * truth > 0)
    return {"accuracy": correct / eligible if eligible else None,
            "eligible_pairs": eligible, "correct_pairs": correct,
            "excluded_label_ties": label_ties, "prediction_ties": prediction_ties,
            "min_label_gap": min_gap, "prediction_tie_credit": 0}
