"""Prepare isolated Kaggle dependencies, run G0, and archive evidence on failure too."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--workdir", type=Path, default=Path("/kaggle/working"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = args.workdir / f"g0-{stamp}"
    run.mkdir(parents=True, exist_ok=False)
    env_dir = args.workdir / ".promptalign-g0-venv"
    python = env_dir / "bin/python"
    model = args.workdir / ".cache/clip-vit-base-patch32"

    def execute(name: str, command: list[str]) -> None:
        print(f"Running {name}; log: {run / (name + '.log')}", flush=True)
        with (run / f"{name}.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)

    status = "FAIL"
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        (run / "provenance.json").write_text(json.dumps({"commit": commit, "device_requested": args.device, "started_utc": stamp}, indent=2), encoding="utf-8")
        if not python.exists():
            execute("create_venv", [sys.executable, "-m", "venv", str(env_dir)])
        index = "cu126" if args.device == "cuda" else "cpu"
        execute("install_torch", [str(python), "-m", "pip", "install", f"torch==2.8.0+{index}", "--index-url", f"https://download.pytorch.org/whl/{index}", "--progress-bar", "off"])
        execute("install_dependencies", [str(python), "-m", "pip", "install", "-r", "requirements-g0.in", "--progress-bar", "off"])
        execute("download_clip", [str(python), "scripts/download_clip.py", "--destination", str(model), "--output", str(run / "clip_download.json")])
        command = [str(python), "scripts/run_g0_checks.py", "--root", str(root), "--model", str(model), "--output", str(run / "checks")]
        if args.device == "cuda":
            command.append("--require-cuda")
        execute("checks", command)
        status = "PASS"
    except Exception as exc:
        (run / "error.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        print(f"G0 failed: {exc}", flush=True)
    finally:
        (run / "result.json").write_text(json.dumps({"status": status, "scope": "G0 execution only; human review still required"}, indent=2), encoding="utf-8")
        archive = shutil.make_archive(str(run), "zip", root_dir=run)
        print(f"RESULT: {status}\nSEND BACK: {archive}", flush=True)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
