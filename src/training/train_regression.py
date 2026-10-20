"""Train-only fit and validation-only selection of M0/Ridge/MLP."""

from copy import deepcopy
import math

import numpy as np
import torch

from src.data.validate import normalize_prompt
from src.evaluation.baselines import evaluate_baselines
from src.evaluation.metrics import regression_metrics, pairwise_accuracy
from src.models.regression_head import RegressionHead, features


def fit_heads(x_train, y_train, x_val, y_val, *, seed=42, alpha=1.0,
              hidden=64, epochs=100, patience=15, learning_rate=0.001):
    x_train, x_val = np.asarray(x_train, dtype=np.float32), np.asarray(x_val, dtype=np.float32)
    y_train, y_val = np.asarray(y_train, dtype=np.float32), np.asarray(y_val, dtype=np.float32)
    if (x_train.ndim != 2 or x_val.ndim != 2 or x_train.shape[1] != x_val.shape[1]
            or len(x_train) == 0 or len(x_val) == 0 or x_train.shape[1] == 0
            or y_train.shape != (len(x_train),) or y_val.shape != (len(x_val),)):
        raise ValueError("invalid train/validation shapes")
    if not all(np.isfinite(a).all() for a in (x_train, x_val, y_train, y_val)):
        raise ValueError("nonfinite training data")
    if any(np.any((y < 1) | (y > 5)) for y in (y_train, y_val)):
        raise ValueError("targets must be in [1,5]")
    if (not math.isfinite(alpha) or alpha <= 0 or epochs < 1 or patience < 1
            or hidden < 1 or not math.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError("invalid training configuration")
    center = x_train.astype(float).mean(axis=0)
    scale = x_train.astype(float).std(axis=0)
    scale[scale < 1e-8] = 1
    standardized = (x_train - center) / scale
    mean = float(y_train.mean())
    target = y_train.astype(float) - mean
    # Solve the smaller system; intercept is not penalized.
    if len(x_train) < x_train.shape[1]:
        weight = standardized.T @ np.linalg.solve(
            standardized @ standardized.T + alpha * np.eye(len(x_train)), target)
    else:
        weight = np.linalg.solve(standardized.T @ standardized + alpha * np.eye(x_train.shape[1]),
                                 standardized.T @ target)
    models = {}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for kind in ("mean", "ridge", "mlp"):
            model = RegressionHead(x_train.shape[1], kind, hidden)
            model.center.copy_(torch.tensor(center, dtype=torch.float32))
            model.scale.copy_(torch.tensor(scale, dtype=torch.float32))
            if kind != "mlp":
                with torch.no_grad():
                    model.network.weight.copy_(torch.tensor(weight if kind == "ridge" else np.zeros_like(weight)).float()[None])
                    model.network.bias.fill_(mean)
            models[kind] = model
        model = models["mlp"]
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        x, y = torch.from_numpy(x_train), torch.from_numpy(y_train)
        history, best, best_epoch, stale = [], float("inf"), 0, 0
        best_state = None
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            loss = torch.mean((model(x) - y) ** 2)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite loss")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError("nonfinite gradient")
            optimizer.step()
            prediction = model.predict(x_val)
            val_mae = regression_metrics(y_val, prediction)["mae"]
            history.append({"epoch": epoch, "train_mse_before_step": float(loss.detach()), "val_mae": val_mae})
            if val_mae < best:
                best, best_epoch, stale = val_mae, epoch, 0
                best_state = deepcopy(model.state_dict())
            else:
                stale += 1
            if stale >= patience:
                break
        model.load_state_dict(best_state)
    metrics = {kind: regression_metrics(y_val, model.predict(x_val)) for kind, model in models.items()}
    selected = min(models, key=lambda kind: metrics[kind]["mae"])
    return models, {"selected": selected, "selection_metric": "validation_mae",
                    "metrics": metrics, "history": history, "best_epoch": best_epoch,
                    "epochs_run": len(history), "train_count": len(y_train), "val_count": len(y_val)}


def train_from_embeddings(records, arrays, **config):
    ids = [r["pair_id"] for r in records]
    if arrays["pair_ids"].tolist() != ids:
        raise ValueError("embedding pair IDs do not match manifest order")
    # Also validates split names, unique IDs and normalized prompt disjointness.
    baseline = evaluate_baselines(records, dict(zip(ids, arrays["cosine"].tolist())))
    text, image = arrays["text"], arrays["image"]
    if text.ndim != 2 or image.shape != text.shape or len(text) != len(records):
        raise ValueError("embedding shape mismatch")
    partitions = {}
    for split in ("train", "val"):
        indices = [i for i, r in enumerate(records) if r["split"] == split]
        partitions[split] = (features(text[indices], image[indices]),
                             np.asarray([records[i]["target"] for i in indices]))
    models, report = fit_heads(*partitions["train"], *partitions["val"], **config)
    report["baselines"] = baseline
    x_val, y_val = partitions["val"]
    groups = [normalize_prompt(r["prompt"]) for r in records if r["split"] == "val"]
    for kind, model in models.items():
        report["metrics"][kind]["pairwise"] = pairwise_accuracy(y_val, model.predict(x_val), groups)
    return models, report, x_val
