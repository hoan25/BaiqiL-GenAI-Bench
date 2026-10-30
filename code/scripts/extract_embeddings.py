"""Frozen extraction for development fixtures or explicitly reviewed data."""

import argparse
import json
from pathlib import Path
import sys
import time
import subprocess

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import transformers

from src.data.qa import sha256_file
from src.evaluation.baselines import evaluate_baselines
from src.models.clip_encoder import ClipEncoder
from src.training.extract_embeddings import extract_embeddings
from src.data.acceptance import validate_lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--development", action="store_true")
    mode.add_argument("--data-lock", type=Path, help="Explicitly reviewed full-data manifest lock")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "FAIL", "scope": "development_only_not_full_data_acceptance",
              "device": args.device, "batch_size": args.batch_size,
              "torch": torch.__version__, "transformers": transformers.__version__}
    started = time.perf_counter()
    try:
        if args.data_lock:
            validate_lock(args.data_lock, args.manifest, args.data_root)
            report.update(scope="reviewed_full_data_validation_only",
                          data_lock_sha256=sha256_file(args.data_lock))
        report["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True).strip()
        report["model_files_sha256"] = {
            p.name: sha256_file(p) for p in sorted(args.model.iterdir()) if p.is_file()}
        records = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        report["manifest_sha256"] = sha256_file(args.manifest)
        encoder = ClipEncoder(args.model, args.device)
        before = encoder.fingerprint()
        arrays = extract_embeddings(records, args.data_root, encoder, args.batch_size)
        after = encoder.fingerprint()
        if before != after:
            raise ValueError("frozen model state changed")
        metrics = evaluate_baselines(records, dict(zip(arrays["pair_ids"], arrays["cosine"].tolist())))
        np.savez_compressed(args.output / "embeddings.npz", **arrays)
        report.update(status="PASS", count=len(records), dimension=arrays["text"].shape[1],
                      model_state_sha256=before, frozen_state_unchanged=True, validation=metrics,
                      embeddings_sha256=sha256_file(args.output / "embeddings.npz"))
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    report["elapsed_seconds"] = time.perf_counter() - started
    (args.output / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
