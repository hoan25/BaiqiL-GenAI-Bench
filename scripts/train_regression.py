"""Development training CLI; no download or test-holdout evaluation."""

import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.data.qa import sha256_file
from src.models.regression_head import save_head, load_head
from src.training.train_regression import train_from_embeddings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--extraction", type=Path, required=True, help="G2 output directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development", action="store_true", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = {k: getattr(args, k) for k in ("seed", "epochs", "patience", "hidden", "alpha", "learning_rate")}
    result = {"status": "FAIL", "scope": "development_only_not_dataset_benchmark", "config": config,
              "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__}
    try:
        root = Path(__file__).resolve().parents[1]
        for key, command in (("git_commit", ["git", "rev-parse", "HEAD"]),
                             ("git_status", ["git", "status", "--porcelain"])):
            process = subprocess.run(command, cwd=root, capture_output=True, text=True)
            result[key] = process.stdout.strip() if process.returncode == 0 else None
        provenance = json.loads((args.extraction / "result.json").read_text(encoding="utf-8"))
        result["manifest_sha256"] = sha256_file(args.manifest)
        result["embeddings_sha256"] = sha256_file(args.extraction / "embeddings.npz")
        if (provenance["status"] != "PASS" or result["manifest_sha256"] != provenance["manifest_sha256"]
                or result["embeddings_sha256"] != provenance["embeddings_sha256"]):
            raise ValueError("extraction provenance/checksum mismatch")
        result["encoder_state_sha256"] = provenance["model_state_sha256"]
        records = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        with np.load(args.extraction / "embeddings.npz", allow_pickle=False) as arrays:
            models, report, x_val = train_from_embeddings(records, arrays, **config)
        result.update(report)
        result["checkpoints"] = {}
        for kind, model in models.items():
            path = args.output / f"{kind}.pt"
            save_head(model, path)
            delta = float(np.max(np.abs(load_head(path).predict(x_val) - model.predict(x_val))))
            if delta >= 1e-5:
                raise ValueError("checkpoint reload mismatch")
            result["checkpoints"][kind] = {"file": path.name, "sha256": sha256_file(path), "reload_max_delta": delta}
        result["status"] = "PASS"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    (args.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
