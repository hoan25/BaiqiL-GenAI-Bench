"""Isolated CPU data QA handoff. Never trains before full-data review."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.clean_python import python_command, pip_source


def diagnostic_tail(log, limit=6000):
    """Keep notebook errors bounded and remove common credential representations."""
    log = re.sub(r"(https?://)[^\s/@]+:[^\s/@]+@", r"\1[REDACTED]@", log)
    log = re.sub(r"(?i)(authorization\s*[:=]\s*)([^\r\n]+)", r"\1[REDACTED]", log)
    log = re.sub(r"(?i)([?&](?:token|access_token|key)=)[^&\s]+", r"\1[REDACTED]", log)
    return log[-limit:]


def run_qa(workdir, data_root=None, download=False):
    if (data_root is None) == (not download):
        raise ValueError("choose exactly one: local data root or explicit download")
    root = Path(__file__).resolve().parents[1]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run = Path(workdir) / f"data-qa-{stamp}"
    run.mkdir(parents=True, exist_ok=False)
    env_dir = Path(workdir) / f".promptalign-data-{stamp}"
    python = env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    environment = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME"):
        environment.pop(name, None)
    report = {"status": "FAIL", "scope": "data_qa_only_no_training", "started_utc": stamp, "commands": []}

    def execute(name, command, accepted=(0,)):
        command = python_command(command[0]) + list(map(str, command[1:]))
        print(f"{name}: {run / (name + '.log')}", flush=True)
        with (run / f"{name}.log").open("w", encoding="utf-8") as stream:
            result = subprocess.run(command, cwd=root, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        report["commands"].append({"name": name, "returncode": result.returncode})
        log = (run / f"{name}.log").read_text(encoding="utf-8")
        if result.returncode not in accepted or "Error in sitecustomize" in log or "Error in usercustomize" in log:
            report["failed_stage"] = name
            report["failed_returncode"] = result.returncode
            report["diagnostic_tail"] = diagnostic_tail(log)
            print(f"FAILED STAGE: {name}; exit code: {result.returncode}\n"
                  f"--- diagnostic log tail ---\n{report['diagnostic_tail']}\n"
                  f"--- full log: {run / (name + '.log')} ---", flush=True)
            raise RuntimeError(f"{name} failed; see diagnostic output and archived log")

    try:
        report["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        execute("create_venv", [sys.executable, "-I", "-m", "venv", "--without-pip", env_dir])
        execute("bootstrap_pip", [python, "--pip-source", pip_source(), "install", "pip==26.2.1"])
        execute("startup", [python, "-I", "-c", "import sys; print(sys.version); assert sys.prefix != sys.base_prefix"])
        execute("dependencies", [python, "-m", "pip", "install", "-r", "requirements-kaggle-data.in", "--progress-bar", "off"])
        execute("pip_check", [python, "-m", "pip", "check"])
        execute("dependency_imports", [python, "-c", "import pyarrow.parquet; from PIL import Image; from huggingface_hub import snapshot_download; import pytest; print('Data QA imports PASS')"])
        execute("versions", [python, "-m", "pip", "freeze"])
        execute("data_tests", [python, "-m", "pytest", "tests/unit/test_data.py", "tests/integration/test_parquet_pipeline.py", "-q", "-p", "no:cacheprovider", "--basetemp", env_dir / "test-temp"])
        if download:
            execute("download_dataset", [python, "scripts/download_dataset.py", "--allow-download", "--destination", Path(workdir) / ".cache/genai-bench", "--output", run / "dataset.json"])
            data_root = json.loads((run / "dataset.json").read_text(encoding="utf-8"))["data_root"]
        execute("prepare_data", [python, "scripts/prepare_data.py", "--input", data_root, "--output", run / "qa"], accepted=(0, 2))
        report["qa_status"] = json.loads((run / "qa/qa.json").read_text(encoding="utf-8"))["status"]
        report["status"] = "REVIEW_REQUIRED"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(report["error"], flush=True)
    finally:
        (run / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        archive = shutil.make_archive(str(run), "zip", root_dir=run)
        print(f"{report['status']}\nSEND BACK: {archive}", flush=True)
    return 1 if report["status"] == "FAIL" else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=Path("/kaggle/working"))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data-root", type=Path)
    source.add_argument("--download-dataset", action="store_true")
    args = parser.parse_args()
    return run_qa(args.workdir, args.data_root, args.download_dataset)


if __name__ == "__main__":
    raise SystemExit(main())
