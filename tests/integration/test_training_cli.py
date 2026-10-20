import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from src.data.qa import sha256_file


def test_cli_provenance_failure_success_and_no_overwrite(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    records = [{"pair_id": str(i), "prompt": f"prompt {i}", "split": split,
                **({"target": 2 + i} if split != "test" else {})}
               for i, split in enumerate(["train", "val", "test"])]
    manifest.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    extraction = tmp_path / "extraction"
    extraction.mkdir()
    np.savez(extraction / "embeddings.npz", pair_ids=np.array(["0", "1", "2"]),
             text=np.array([[0.], [1.], [np.nan]]), image=np.array([[1.], [0.], [np.nan]]),
             cosine=np.array([0., 0.5, np.nan]))
    provenance = {"status": "PASS", "manifest_sha256": "wrong",
                  "embeddings_sha256": sha256_file(extraction / "embeddings.npz"),
                  "model_state_sha256": "synthetic_fixture_not_clip"}
    provenance_path = extraction / "result.json"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    command = [sys.executable, "-X", "utf8", str(Path(__file__).resolve().parents[2] / "scripts/train_regression.py"),
               "--manifest", str(manifest), "--extraction", str(extraction), "--development", "--epochs", "2"]
    failed = subprocess.run(command + ["--output", str(tmp_path / "failed")], capture_output=True)
    assert failed.returncode == 1
    assert "checksum mismatch" in json.loads((tmp_path / "failed/result.json").read_text())["error"]
    provenance["manifest_sha256"] = sha256_file(manifest)
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    output = tmp_path / "ok"
    succeeded = subprocess.run(command + ["--output", str(output)], capture_output=True)
    assert succeeded.returncode == 0, succeeded.stderr
    result = json.loads((output / "result.json").read_text())
    assert result["status"] == "PASS"
    assert all(c["reload_max_delta"] < 1e-5 for c in result["checkpoints"].values())
    original = (output / "result.json").read_bytes()
    repeated = subprocess.run(command + ["--output", str(output)], capture_output=True)
    assert repeated.returncode != 0
    assert (output / "result.json").read_bytes() == original
