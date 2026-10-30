import json
from pathlib import Path
import subprocess

import pytest

from scripts.clean_python import python_command


@pytest.mark.parametrize("script", ["extract_embeddings", "train_regression"])
def test_official_cli_rejects_unapproved_lock_before_work(tmp_path, script):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("{}\n")
    lock = tmp_path / "lock.json"
    lock.write_text("{}")
    extraction = tmp_path / "extraction"
    extraction.mkdir()
    (extraction / "result.json").write_text("{}")
    output = tmp_path / "result"
    command = python_command() + [str(Path("scripts") / f"{script}.py"),
                                  "--manifest", str(manifest), "--data-lock", str(lock),
                                  "--output", str(output)]
    if script == "extract_embeddings":
        command += ["--data-root", str(tmp_path), "--model", str(tmp_path / "absent_model")]
    else:
        command += ["--extraction", str(extraction)]
    proc = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
    assert proc.returncode == 1, proc.stderr
    result = json.loads((output / "result.json").read_text())
    assert result["status"] == "FAIL"
    assert "data lock" in result["error"]
    assert not list(output.glob("*.pt"))
    assert not (output / "embeddings.npz").exists()
