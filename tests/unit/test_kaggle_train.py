import argparse
import json
from pathlib import Path
import subprocess
import zipfile

import pytest

from scripts import kaggle_train


def args(tmp_path, accept=False):
    return argparse.Namespace(workdir=tmp_path, qa_archive=tmp_path / "qa.zip",
                              reviewer="Fixture reviewer", accept_data_review=accept,
                              data_root=None, download_dataset=False, device="cuda", batch_size=16)


def test_no_review_stops_before_install_or_download(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("must not execute"))
    assert kaggle_train.run(args(tmp_path)) == 1
    with zipfile.ZipFile(next(tmp_path.glob("training-*.zip"))) as z:
        result = json.loads(z.read("result.json"))
        assert result["status"] == "FAIL" and result["commands"] == []
        assert "accept" in result["error"]


def setup_fake_lock(archive, output, reviewer):
    output.mkdir()
    (output / "manifest.jsonl").write_text("fixture")
    lock = output / "lock.json"
    lock.write_text(json.dumps({"dataset_revision": "fixture", "sources": []}))
    return lock


def test_stage_failure_keeps_zip_and_stops_training(tmp_path, monkeypatch):
    monkeypatch.setattr(kaggle_train, "accept_bundle", setup_fake_lock)
    monkeypatch.setattr(subprocess, "check_output",
                        lambda command, **kw: "" if "status" in command else "fixture-commit")
    def fail(command, **kw):
        assert "-I" in command and "-S" in command
        assert "PYTHONPATH" not in kw["env"]
        kw["stdout"].write("fixture install failure")
        return subprocess.CompletedProcess(command, 1)
    monkeypatch.setattr(subprocess, "run", fail)
    assert kaggle_train.run(args(tmp_path, True)) == 1
    with zipfile.ZipFile(next(tmp_path.glob("training-*.zip"))) as z:
        report = json.loads(z.read("result.json"))
        assert report["failed_stage"] == "create_venv"
        assert len(report["commands"]) == 1
        assert "fixture install failure" in report["diagnostic_tail"]


def test_success_uses_fixed_primary_seed_and_no_holdout(tmp_path, monkeypatch):
    monkeypatch.setattr(kaggle_train, "accept_bundle", setup_fake_lock)
    monkeypatch.setattr(subprocess, "check_output",
                        lambda command, **kw: "" if "status" in command else "fixture-commit")
    commands = []
    def run(command, **kw):
        commands.append(command)
        if "scripts/train_regression.py" in command:
            output = Path(command[command.index("--output") + 1])
            output.mkdir()
            (output / "result.json").write_text(json.dumps({
                "selected": "ridge", "metrics": {kind: {"mae": 1.0} for kind in ("mean", "ridge", "mlp")}}))
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(subprocess, "run", run)
    assert kaggle_train.run(args(tmp_path, True)) == 0
    report = json.loads(next(tmp_path.glob("training-*/result.json")).read_text())
    assert report["primary_training"] == "training-seed42"
    assert report["seeds"] == [42, 43, 44]
    assert not any("--development" in command for command in commands)
    assert not any("scripts/download_dataset.py" in command for command in commands)
    assert all(Path(p).is_file() for c in commands if "pytest" in c for p in c if p.startswith("tests/"))
