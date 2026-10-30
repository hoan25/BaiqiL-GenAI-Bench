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
              hidden=64, epochs=100, patience=15, learning_rate=0.001, device="cpu"):
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; refusing CPU fallback")
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
    if device == "cpu":
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
    if device == "cuda":
        # Fit statistics and the Ridge solve on GPU in float64.
        gx = torch.as_tensor(x_train, dtype=torch.float64, device=device)
        gy = torch.as_tensor(y_train, dtype=torch.float64, device=device)
        gc, gs = gx.mean(0), gx.std(0, correction=0)
        gs = torch.where(gs < 1e-8, torch.ones_like(gs), gs)
        gz = (gx - gc) / gs
        gm = gy.mean()
        if len(x_train) < x_train.shape[1]:
            gw = gz.T @ torch.linalg.solve(gz @ gz.T + alpha * torch.eye(len(gx), device=device, dtype=gx.dtype), gy-gm)
        else:
            gw = torch.linalg.solve(gz.T @ gz + alpha * torch.eye(gz.shape[1], device=device, dtype=gx.dtype), gz.T @ (gy-gm))
        center, scale, weight, mean = gc.cpu().numpy(), gs.cpu().numpy(), gw.cpu().numpy(), float(gm.item())
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device == "cuda" else []):
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
        model = models["mlp"].to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        x, y = torch.from_numpy(x_train).to(device), torch.from_numpy(y_train).to(device)
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
    device_prediction = models["mlp"].predict(x_val)
    models["mlp"].cpu()
    parity_delta = float(np.max(np.abs(models["mlp"].predict(x_val) - device_prediction)))
    if parity_delta > 1e-4:
        raise ValueError("GPU to CPU inference parity exceeded 1e-4")
    metrics = {kind: regression_metrics(y_val, model.predict(x_val)) for kind, model in models.items()}
    selected = min(models, key=lambda kind: metrics[kind]["mae"])
    return models, {"selected": selected, "selection_metric": "validation_mae",
                    "metrics": metrics, "history": history, "best_epoch": best_epoch,
                    "epochs_run": len(history), "train_count": len(y_train), "val_count": len(y_val),
                    "training_device": device, "cuda_version": torch.version.cuda if device == "cuda" else None,
                    "device_name": torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
                    "export_device": "cpu", "device_to_cpu_max_delta": parity_delta}


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
