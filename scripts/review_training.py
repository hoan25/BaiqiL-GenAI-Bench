"""Verify a Kaggle training bundle and replay validation on CPU, never holdout."""

import argparse
from collections import Counter
import json
from pathlib import Path, PurePosixPath
import statistics
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from src.data.acceptance import sha256, validate_lock
from src.evaluation.baselines import evaluate_baselines
from src.evaluation.metrics import regression_metrics, pairwise_accuracy
from src.models.regression_head import load_head, features


def compare(actual, expected, tolerance=1e-5):
    if isinstance(expected, dict):
        for key, value in expected.items():
            compare(actual[key], value, tolerance)
    elif isinstance(expected, (int, float)):
        if not np.isfinite(actual) or abs(actual - expected) > tolerance:
            raise ValueError(f"metric mismatch: {actual} versus {expected}")
    elif actual != expected:
        raise ValueError(f"metadata mismatch: {actual} versus {expected}")


def review(archive, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        if sum(i.file_size for i in z.infolist()) > 256 * 1024 * 1024:
            raise ValueError("Unexpectedly large training bundle")
        if z.testzip() is not None:
            raise ValueError("Archive CRC failure")
        for item in z.infolist():
            name = PurePosixPath(item.filename)
            target = (output / item.filename).resolve()
            if (name.is_absolute() or ".." in name.parts or "\\" in item.filename
                    or not target.is_relative_to(output)):
                raise ValueError("Unsafe archive member")
            if item.is_dir():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(z.read(item))
    read = lambda name: json.loads((output / name).read_text(encoding="utf-8"))
    run = read("result.json")
    if run["status"] != "PASS" or any(c["returncode"] for c in run["commands"]):
        raise ValueError("Training run did not pass")
    if run["primary_seed"] != 42 or run["primary_training"] != "training-seed42":
        raise ValueError("Primary seed policy changed")
    manifest = output / "data/manifest.jsonl"
    lock_path = output / "data/lock.json"
    validate_lock(lock_path, manifest)
    if sha256(lock_path) != run["data_lock_sha256"]:
        raise ValueError("Data lock checksum mismatch")
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    extraction = read("extraction/result.json")
    embedding = output / "extraction/embeddings.npz"
    if (extraction["status"] != "PASS" or not extraction["frozen_state_unchanged"]
            or extraction["manifest_sha256"] != sha256(manifest)
            or extraction["embeddings_sha256"] != sha256(embedding)
            or extraction["data_lock_sha256"] != sha256(lock_path)):
        raise ValueError("Extraction provenance mismatch")
    with np.load(embedding, allow_pickle=False) as data:
        if data["pair_ids"].tolist() != [r["pair_id"] for r in records]:
            raise ValueError("Embedding IDs mismatch")
        for key in ("text", "image"):
            if data[key].shape != (9598, 512) or not np.isfinite(data[key]).all():
                raise ValueError("Invalid embedding dimensions or values")
            if not np.allclose(np.linalg.norm(data[key], axis=1), 1, atol=1e-4):
                raise ValueError("Non-unit embeddings")
        if not np.allclose(data["cosine"], np.sum(data["text"] * data["image"], axis=1), atol=1e-5):
            raise ValueError("Cosine mismatch")
        baseline = evaluate_baselines(records, dict(zip(data["pair_ids"], data["cosine"].tolist())))
        compare(baseline, extraction["validation"])
        train = [i for i, r in enumerate(records) if r["split"] == "train"]
        val = [i for i, r in enumerate(records) if r["split"] == "val"]
        x_train = features(data["text"][train], data["image"][train])
        x_val = features(data["text"][val], data["image"][val])
    # Only train and validation targets are accessed. Test metrics remain absent.
    y_val = [records[i]["target"] for i in val]
    groups = [records[i]["prompt_group"] for i in val]
    center, scale = x_train.astype(float).mean(axis=0), x_train.astype(float).std(axis=0)
    scale[scale < 1e-8] = 1
    checked = {}
    for seed in (42, 43, 44):
        folder = f"training-seed{seed}"
        result = read(f"{folder}/result.json")
        if (result["status"] != "PASS" or result["config"]["seed"] != seed
                or result["scope"] != "reviewed_full_data_validation_only"
                or result["manifest_sha256"] != sha256(manifest)
                or result["data_lock_sha256"] != sha256(lock_path)
                or result["embeddings_sha256"] != sha256(embedding)
                or result["encoder_state_sha256"] != extraction["model_state_sha256"]):
            raise ValueError("Training provenance mismatch")
        metrics = {}
        for kind in ("mean", "ridge", "mlp"):
            checkpoint = result["checkpoints"][kind]
            if checkpoint["file"] != f"{kind}.pt":
                raise ValueError("Unexpected checkpoint name")
            path = output / folder / checkpoint["file"]
            if sha256(path) != checkpoint["sha256"]:
                raise ValueError("Checkpoint checksum mismatch")
            model = load_head(path)  # CPU, weights_only=True
            if (model.config["kind"] != kind or not np.allclose(model.center.numpy(), center, atol=1e-6)
                    or not np.allclose(model.scale.numpy(), scale, atol=1e-6)):
                raise ValueError("Scaler was not fitted on train embeddings")
            prediction = model.predict(x_val)
            metrics[kind] = {**regression_metrics(y_val, prediction),
                             "pairwise": pairwise_accuracy(y_val, prediction, groups)}
            compare(metrics[kind], result["metrics"][kind])
        compare(baseline, result["baselines"])
        if result["selected"] != min(metrics, key=lambda k: metrics[k]["mae"]):
            raise ValueError("Selected head is not minimum validation MAE")
        best = min(result["history"], key=lambda h: h["val_mae"])
        if best["epoch"] != result["best_epoch"]:
            raise ValueError("Early stopping epoch mismatch")
        checked[str(seed)] = metrics
    aggregate = {kind: {"mean": statistics.mean(checked[str(s)][kind]["mae"] for s in (42, 43, 44)),
                       "std_population": statistics.pstdev(checked[str(s)][kind]["mae"] for s in (42, 43, 44))}
                 for kind in ("mean", "ridge", "mlp")}
    compare(aggregate, run["validation_mae_across_seeds"])
    primary = read("training-seed42/result.json")
    if run["selected"] != primary["selected"]:
        raise ValueError("Primary selection mismatch")
    report = {"status": "PASS", "scope": "validation_replay_cpu_not_holdout_or_release",
              "archive_sha256": sha256(archive), "source_commit": run["git_commit"],
              "manifest_sha256": sha256(manifest), "split_counts": dict(Counter(r["split"] for r in records)),
              "baseline": baseline, "cpu_metrics": checked, "aggregate": aggregate,
              "selected": primary["selected"], "primary_seed": 42,
              "checks": ["ZIP CRC", "data lock", "manifest/embedding/checkpoint checksums",
                         "9598x512 finite normalized embeddings", "train-only scaler",
                         "CPU replay of nine heads", "validation selection and best epoch"],
              "test_holdout": "NOT EVALUATED"}
    (output / "review.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    receipt = {"status": "VALIDATED_FOR_LOCAL_DEMO", "scope": "reviewed_full_data_validation_only",
               "training_report_sha256": sha256(output / "training-seed42/result.json"),
               "model_files_sha256": extraction["model_files_sha256"],
               "archive_sha256": report["archive_sha256"]}
    (output / "training-seed42/artifact_review.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    result = review(args.archive, args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
