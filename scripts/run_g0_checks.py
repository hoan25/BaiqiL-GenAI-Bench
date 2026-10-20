"""Run G0 checks in fresh processes and retain logs for local/Kaggle review."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.clean_python import python_command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    root, model, output = args.root.resolve(), args.model.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    python = python_command() if os.environ.get("PROMPTALIGN_CLEAN_PYTHON") == "1" else [sys.executable, "-X", "utf8"]
    commands = {
        "environment": python + ["scripts/check_environment.py", "--output", str(output / "environment.json")],
        "pytest": python + ["-m", "pytest", "tests/unit", "-q", "-p", "no:cacheprovider"],
        "clip_cpu": python + ["scripts/smoke_clip.py", "--model", str(model), "--device", "cpu", "--output", str(output / "clip_cpu.json")],
        "clip_auto": python + ["scripts/smoke_clip.py", "--model", str(model), "--device", "cuda" if args.require_cuda else "auto", "--output", str(output / "clip_auto.json")],
        "versions": python + ["-m", "pip", "freeze", "--all"],
    }
    results = []
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1")
    for name, command in commands.items():
        result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, encoding="utf-8")
        (output / f"{name}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        results.append({"name": name, "command": command, "exit_code": result.returncode})
        (output / "commands.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"{name}: exit {result.returncode}", flush=True)
        if result.returncode:
            print(result.stdout + result.stderr)
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
