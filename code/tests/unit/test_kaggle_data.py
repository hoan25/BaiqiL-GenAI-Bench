import json
from pathlib import Path
import subprocess
import zipfile

import pytest

from scripts.kaggle_data import run_qa, diagnostic_tail


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


def test_pip_failure_prints_actual_diagnostic_and_archives(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: "fixture-commit")
    def execute(command, **kwargs):
        if "--pip-source" in command:
            kwargs["stdout"].write("ERROR: fixture pip bootstrap failure\n")
            return subprocess.CompletedProcess(command, 1)
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(subprocess, "run", execute)
    assert run_qa(tmp_path, data_root=tmp_path) == 1
    assert "ERROR: fixture pip bootstrap failure" in capsys.readouterr().out
    with zipfile.ZipFile(next(tmp_path.glob("*.zip"))) as archive:
        report = json.loads(archive.read("result.json"))
        assert report["failed_stage"] == "bootstrap_pip"
        assert report["failed_returncode"] == 1
        assert "fixture pip bootstrap failure" in report["diagnostic_tail"]


def test_zero_exit_startup_error_is_not_hidden(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: "fixture-commit")
    def execute(command, **kwargs):
        kwargs["stdout"].write("Error in sitecustomize: fixture module missing\n")
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(subprocess, "run", execute)
    assert run_qa(tmp_path, data_root=tmp_path) == 1
    assert "fixture module missing" in capsys.readouterr().out


def test_diagnostic_tail_is_bounded_and_redacted():
    log = "https://name:password@example.org/simple?token=secret\nAuthorization: Bearer hidden\n"
    output = diagnostic_tail(log)
    assert "password" not in output and "secret" not in output and "hidden" not in output
    assert len(diagnostic_tail("x" * 10000)) == 6000
