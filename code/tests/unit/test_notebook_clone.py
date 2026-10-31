"""Exercise the notebook's Git setup without contacting GitHub or Kaggle."""

import json
from pathlib import Path
import subprocess

import pytest

CODE_ROOT = Path(__file__).resolve().parents[2]


def test_notebook_clone_resume_and_preserve_changes(tmp_path):
    source = tmp_path / "source"
    checkout = tmp_path / "checkout"
    source.mkdir()
    subprocess.run(["git", "init", "-b", "main", str(source)], check=True, capture_output=True)
    (source / "example.txt").write_text("original", encoding="utf-8")
    subprocess.run(["git", "add", "example.txt"], cwd=source, check=True)
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "fixture"], cwd=source, check=True, capture_output=True)
    notebook = json.loads((CODE_ROOT / "notebooks/00_environment.ipynb").read_text(encoding="utf-8"))
    cell = next(c for c in notebook["cells"] if c["id"] == "github")
    code = "".join(cell["source"]).replace("https://github.com/hoan25/BaiqiL-GenAI-Bench.git", source.as_posix()).replace("/kaggle/working/promptalign-ai", checkout.as_posix())
    exec(compile(code, "notebook_clone", "exec"), {})
    assert (checkout / "example.txt").read_text() == "original"
    exec(compile(code, "notebook_resume", "exec"), {})
    (checkout / "example.txt").write_text("user edit", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Working tree has changes"):
        exec(compile(code, "notebook_dirty", "exec"), {})
    assert (checkout / "example.txt").read_text() == "user edit"
