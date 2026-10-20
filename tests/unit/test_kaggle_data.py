import json
from pathlib import Path
import subprocess
import zipfile

import pytest

from scripts.kaggle_data import run_qa


def test_explicit_data_source_required(tmp_path):
    with pytest.raises(ValueError):
        run_qa(tmp_path)
    with pytest.raises(ValueError):
        run_qa(tmp_path, tmp_path, download=True)


def test_bootstrap_failure_exports_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: "fixture-commit")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 1))
    assert run_qa(tmp_path, data_root=tmp_path) == 1
    archives = list(tmp_path.glob("data-qa-*.zip"))
    assert len(archives) == 1
    with zipfile.ZipFile(archives[0]) as archive:
        report = json.loads(archive.read("result.json"))
        assert report["status"] == "FAIL"
        assert report["commands"][0]["name"] == "create_venv"
        assert "create_venv.log" in archive.namelist()


def test_qa_review_exit_is_not_training_success(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: "fixture-commit")
    def execute(command, **kwargs):
        assert "PYTHONPATH" not in kwargs["env"]
        if "scripts/prepare_data.py" in command:
            output = Path(command[command.index("--output") + 1])
            output.mkdir()
            (output / "qa.json").write_text(json.dumps({"status": "REVIEW_REQUIRED"}))
            return subprocess.CompletedProcess(command, 2)
        assert "scripts/download_dataset.py" not in command
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(subprocess, "run", execute)
    assert run_qa(tmp_path, data_root=tmp_path) == 0
    report = json.loads(next(tmp_path.glob("data-qa-*/result.json")).read_text())
    assert report["status"] == "REVIEW_REQUIRED"
    assert report["scope"] == "data_qa_only_no_training"
