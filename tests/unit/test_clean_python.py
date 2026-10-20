"""Reproduce zero-exit customization failure using a real disposable venv."""

import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.clean_python import python_command, pip_source


def test_broken_host_hook_is_skipped_without_losing_venv(tmp_path):
    env_dir = tmp_path / "isolated"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    subprocess.run(python_command() + ["-m", "venv", "--without-pip", str(env_dir)],
                   env=env, check=True, capture_output=True)
    python = env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    probe = "import sys,sysconfig,json; print(json.dumps([sys.prefix,sys.base_prefix,sysconfig.get_path('purelib')]))"
    def run(args):
        return subprocess.run(args, env=env, capture_output=True, text=True, encoding="utf-8")
    clean = run(python_command(python) + ["-c", probe])
    assert clean.returncode == 0, clean.stderr
    prefix, base, packages = json.loads(clean.stdout)
    assert Path(prefix).resolve() == env_dir.resolve()
    assert prefix != base
    package_dir = Path(packages)
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "sitecustomize.py").write_text(
        "raise ModuleNotFoundError(\"No module named 'wrapt'\")\n", encoding="utf-8")
    ordinary = run([str(python), "-I", "-c", probe])
    assert ordinary.returncode == 0
    assert "Error in sitecustomize" in ordinary.stderr
    assert "wrapt" in ordinary.stderr
    isolated = run(python_command(python) + ["-c", probe])
    assert isolated.returncode == 0 and isolated.stderr == ""
    assert json.loads(isolated.stdout) == [prefix, base, packages]
    # Installed packages remain importable; host site-packages are not exposed.
    (package_dir / "qa_probe.py").write_text("VALUE = 42\n", encoding="utf-8")
    imported = run(python_command(python) + ["-c", "import qa_probe; assert qa_probe.VALUE == 42"])
    assert imported.returncode == 0 and imported.stderr == ""
    bootstrap = run(python_command(python) + ["--pip-source", pip_source(), "--version"])
    assert bootstrap.returncode == 0 and bootstrap.stderr == "", bootstrap.stderr
    assert "pip " in bootstrap.stdout
