"""VQAScore-style zero-shot alignment scoring and simple train-only calibration.

The scorer is injected: any callable ``scorer(image_paths, prompts) -> list[float]``
returning one alignment score per (image, prompt) pair. Labels are never passed
to the scorer. Calibration (linear and isotonic) is fit on the train split only
and reported on the validation split; the test split is skipped unless the
caller explicitly asks for it.
"""

from collections import defaultdict
import hashlib
from io import BytesIO
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

from src.data.ingest import iter_rows
from src.data.validate import normalize_prompt
from src.evaluation.metrics import pairwise_accuracy, regression_metrics, spearman

REFERENCE_MLP42_VALIDATION = {
    "source": "project_records/reports/model_card.md, MLP seed42 on GenAI-Bench validation",
    "mae": 0.773777, "rmse": 0.950154, "spearman": 0.480619, "pairwise_accuracy": 0.661340,
}


def _extension(payload):
    with Image.open(BytesIO(payload)) as image:
        fmt = (image.format or "png").lower()
    return "jpg" if fmt == "jpeg" else fmt


def score_pairs(pairs, scorer, batch_size=8, workdir=None, progress=None):
    """pairs: iterable of (key, prompt, image_bytes). Returns {key: score}."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    results, pending = {}, []
    with tempfile.TemporaryDirectory(dir=workdir) as temp:
        temp = Path(temp)

        def flush():
            if not pending:
                return
            paths, prompts = [], []
            for key, prompt, payload in pending:
                path = temp / f"{hashlib.sha256(payload).hexdigest()[:24]}.{_extension(payload)}"
                if not path.exists():
                    path.write_bytes(payload)
                paths.append(str(path))
                prompts.append(prompt)
            scores = [float(s) for s in scorer(paths, prompts)]
            if len(scores) != len(pending) or not np.isfinite(scores).all():
                raise ValueError("scorer returned wrong count or nonfinite scores")
            for (key, _, _), score in zip(pending, scores):
                if key in results:
                    raise ValueError(f"duplicate pair key {key}")
                results[key] = score
            if progress:
                progress(len(results))
            for path in paths:
                Path(path).unlink(missing_ok=True)
            pending.clear()

        for key, prompt, payload in pairs:
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("prompt must be nonempty text")
            if not isinstance(payload, bytes) or not payload:
                raise ValueError("image payload must be nonempty bytes")
            pending.append((key, prompt, payload))
            if len(pending) >= batch_size:
                flush()
        flush()
    return results


def manifest_pairs(records, data_root, splits):
    """Yield (pair_id, prompt, image_bytes) for manifest records in the requested splits."""
    wanted = {}
    for record in records:
        if record["split"] not in splits:
            continue
        key = (record["shard"], record["row"], record["image_model"])
        if key in wanted:
            raise ValueError("duplicate source coordinate in manifest")
        wanted[key] = record
    root = Path(data_root).resolve()
    paths = []
    for shard in sorted({k[0] for k in wanted}):
        path = (root / shard).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"shard missing or outside data root: {shard}")
        paths.append(path)
    seen = 0
    for row in iter_rows(root, paths):
        for model, payload in row["images"].items():
            record = wanted.get((row["shard"], row["row"], model))
            if record is None:
                continue
            if row["prompt"] != record["prompt"]:
                raise ValueError("source prompt differs from manifest")
            if not isinstance(payload, bytes) or hashlib.sha256(payload).hexdigest() != record["image_sha256"]:
                raise ValueError("source image differs from manifest")
            seen += 1
            yield record["pair_id"], record["prompt"], payload
    if seen != len(wanted):
        raise ValueError("manifest contains missing source rows or models")


def fit_linear(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 2 or np.ptp(x) == 0:
        raise ValueError("need at least two distinct scores to fit")
    slope, intercept = np.polyfit(x, y, 1)
    return {"kind": "linear", "slope": float(slope), "intercept": float(intercept)}


def fit_isotonic(x, y):
    """Pool-adjacent-violators, nondecreasing; ties in x are averaged first."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 2:
        raise ValueError("need at least two points")
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order]
    knots, means, weights = [], [], []
    start = 0
    while start < len(xs):
        stop = start + 1
        while stop < len(xs) and xs[stop] == xs[start]:
            stop += 1
        knots.append(xs[start]); means.append(ys[start:stop].mean()); weights.append(stop - start)
        start = stop
    blocks = [[k, k, m, w] for k, m, w in zip(knots, means, weights)]  # [x_lo, x_hi, mean, weight]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][2] > blocks[i + 1][2]:
            a, b = blocks[i], blocks[i + 1]
            w = a[3] + b[3]
            blocks[i] = [a[0], b[1], (a[2] * a[3] + b[2] * b[3]) / w, w]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    return {"kind": "isotonic",
            "x": [float(b[0]) for b in blocks] if len(blocks) > 1 else [float(blocks[0][0]), float(blocks[0][1]) + 1e-9],
            "y": [float(b[2]) for b in blocks] if len(blocks) > 1 else [float(blocks[0][2])] * 2}


def apply_calibration(model, x):
    x = np.asarray(x, dtype=float)
    if model["kind"] == "linear":
        return np.clip(model["slope"] * x + model["intercept"], 1, 5)
    if model["kind"] == "isotonic":
        return np.clip(np.interp(x, model["x"], model["y"]), 1, 5)
    raise ValueError("unknown calibration")


def evaluate_scores(records, scores, splits=("train", "val")):
    """Raw-score ranking metrics per split plus train-fit calibrations reported on each split."""
    by_split = defaultdict(list)
    for record in records:
        if record["split"] in splits:
            if record["pair_id"] not in scores:
                raise ValueError(f"missing score for {record['pair_id']}")
            by_split[record["split"]].append(record)
    report = {"splits": {}, "calibration": {}, "reference_mlp42_validation": REFERENCE_MLP42_VALIDATION}
    for split, items in by_split.items():
        target = np.asarray([r["target"] for r in items], dtype=float)
        raw = np.asarray([scores[r["pair_id"]] for r in items], dtype=float)
        groups = [normalize_prompt(r["prompt"]) for r in items]
        report["splits"][split] = {
            "count": len(items), "raw_mean": float(raw.mean()), "raw_std": float(raw.std()),
            "raw_spearman": spearman(target, raw),
            "raw_pairwise": pairwise_accuracy(target, raw, groups)}
    if "train" in by_split:
        train = by_split["train"]
        x = [scores[r["pair_id"]] for r in train]
        y = [r["target"] for r in train]
        for fit in (fit_linear, fit_isotonic):
            model = fit(x, y)
            report["calibration"][model["kind"]] = {"model": model, "metrics": {}}
            for split, items in by_split.items():
                target = np.asarray([r["target"] for r in items], dtype=float)
                raw = np.asarray([scores[r["pair_id"]] for r in items], dtype=float)
                report["calibration"][model["kind"]]["metrics"][split] = regression_metrics(
                    target, apply_calibration(model, raw))
    return report
