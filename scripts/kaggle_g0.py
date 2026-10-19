"""Prepare isolated Kaggle dependencies, run G0, and archive evidence on failure too."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--workdir", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--bootstrap-only", action="store_true", help="Verify fresh venv startup without rerunning ML checks")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run = args.workdir / f"g0-{stamp}"
    run.mkdir(parents=True, exist_ok=False)
    env_dir = args.workdir / (f".promptalign-bootstrap-{stamp}" if args.bootstrap_only else ".promptalign-g0-venv")
    python = env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    model = args.workdir / ".cache/clip-vit-base-patch32"
    process_env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME"):
        process_env.pop(key, None)

    def execute(name: str, command: list[str]) -> None:
        command = [command[0], "-X", "utf8", *command[1:]]
        print(f"Running {name}; log: {run / (name + '.log')}", flush=True)
        with (run / f"{name}.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, cwd=root, env=process_env, stdout=log, stderr=subprocess.STDOUT, check=True)
        content = (run / f"{name}.log").read_text(encoding="utf-8")
        if "Error in sitecustomize" in content or "Error in usercustomize" in content:
            raise RuntimeError(f"Python startup error in {name}.log")

    status = "FAIL"
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        (run / "provenance.json").write_text(json.dumps({"commit": commit, "device_requested": args.device, "started_utc": stamp, "bootstrap_only": args.bootstrap_only}, indent=2), encoding="utf-8")
        if not python.exists():
            execute("create_venv", [sys.executable, "-I", "-m", "venv", "--without-pip", str(env_dir)])
        execute("bootstrap_pip", [sys.executable, "-I", "-m", "pip", "--python", str(python), "install", "pip==26.2.1", "--disable-pip-version-check"])
        execute("startup", [str(python), "-I", "-c", "import sys; print(sys.version); print(sys.prefix); assert sys.prefix != sys.base_prefix"])
        execute("pip_check", [str(python), "-I", "-m", "pip", "check"])
        if args.bootstrap_only:
            status = "PASS"
            return 0
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
        scope = "Fresh venv bootstrap only; does not rerun or certify GPU inference" if args.bootstrap_only else "G0 execution only; human review still required"
        (run / "result.json").write_text(json.dumps({"status": status, "scope": scope}, indent=2), encoding="utf-8")
        archive = shutil.make_archive(str(run), "zip", root_dir=run)
        print(f"RESULT: {status}\nSEND BACK: {archive}", flush=True)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
