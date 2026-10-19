"""Collect G0 evidence without downloading data or model weights."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import subprocess
import sys
from pathlib import Path


def check_module(name: str, attributes: tuple[str, ...] = ()) -> dict:
    """Import a dependency and required symbols, recording real failures."""
    try:
        module = importlib.import_module(name)
        for attribute in attributes:
            getattr(module, attribute)
        return {"status": "PASS", "version": getattr(module, "__version__", "unknown")}
    except Exception as exc:
        return {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}


def collect_environment() -> dict:
    """Check local imports only; this does not certify model inference or Kaggle."""
    pip_check = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True, text=True, check=False,
    )
    checks = {
        "torch": check_module("torch"),
        "transformers": check_module("transformers", ("CLIPModel", "CLIPProcessor")),
        "huggingface_hub": check_module("huggingface_hub"),
        "pytest": check_module("pytest"),
        "pip_check": {
            "status": "PASS" if pip_check.returncode == 0 and not any(
                marker in pip_check.stdout + pip_check.stderr
                for marker in ("Error in sitecustomize", "Error in usercustomize")
            ) else "FAIL",
            "output": (pip_check.stdout + pip_check.stderr).strip(),
        },
    }
    return {
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "checks": checks,
        "status": "PASS" if all(c["status"] == "PASS" for c in checks.values()) else "FAIL",
        "scope": "Local dependency imports only; ENV-02 through ENV-06 are not certified.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write UTF-8 JSON evidence")
    args = parser.parse_args()
    result = collect_environment()
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
