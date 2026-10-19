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
