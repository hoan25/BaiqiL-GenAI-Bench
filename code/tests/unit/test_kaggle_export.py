"""A failed remote setup must still leave an archive with diagnostic evidence."""

import json
import subprocess
import sys
import zipfile

import pytest

from scripts import kaggle_g0


@pytest.mark.parametrize("failure", [OSError("setup unavailable"), subprocess.CalledProcessError(1, ["setup"])])
def test_setup_failure_exports_logs(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(sys, "argv", ["kaggle_g0.py", "--workdir", str(tmp_path), "--device", "cpu"])
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: "abc123\n")

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(subprocess, "run", fail)
    assert kaggle_g0.main() == 1
    archives = list(tmp_path.glob("g0-*.zip"))
    assert len(archives) == 1
    with zipfile.ZipFile(archives[0]) as archive:
        assert json.loads(archive.read("result.json"))["status"] == "FAIL"
        assert json.loads(archive.read("provenance.json"))["commit"] == "abc123"
        assert archive.read("error.txt")
        assert "create_venv.log" in archive.namelist()


def test_bootstrap_only_avoids_ensurepip_and_ml(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["kaggle_g0.py", "--workdir", str(tmp_path), "--bootstrap-only"])
    monkeypatch.setenv("PYTHONPATH", "external-injection")
    monkeypatch.setenv("PYTHONHOME", "external-python")
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: "abc123\n")
    commands = []

    def run(command, **kwargs):
        assert "PYTHONPATH" not in kwargs["env"]
        assert "PYTHONHOME" not in kwargs["env"]
        commands.append(command)

    monkeypatch.setattr(subprocess, "run", run)
    assert kaggle_g0.main() == 0
    assert "--without-pip" in commands[0]
    assert "--pip-source" in commands[1]
    assert "--python" not in commands[1]
    assert all("-S" in c and "-I" in c for c in commands)
    assert not any("torch" in str(c) or "download_clip.py" in str(c) for c in commands)
    with zipfile.ZipFile(next(tmp_path.glob("g0-*.zip"))) as archive:
        assert "bootstrap only" in json.loads(archive.read("result.json"))["scope"]


def test_zero_exit_startup_error_is_archived_as_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["kaggle_g0.py", "--workdir", str(tmp_path), "--bootstrap-only"])
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: "abc123\n")

    def startup_error(command, **kwargs):
        kwargs["stdout"].write("Error in sitecustomize: missing wrapt\n")

    monkeypatch.setattr(subprocess, "run", startup_error)
    assert kaggle_g0.main() == 1
    with zipfile.ZipFile(next(tmp_path.glob("g0-*.zip"))) as archive:
        assert json.loads(archive.read("result.json"))["status"] == "FAIL"
