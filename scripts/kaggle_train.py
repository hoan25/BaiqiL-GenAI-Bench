"""Reviewed data -> frozen GPU embeddings -> CPU heads, with failure archives."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.clean_python import python_command, pip_source
from scripts.kaggle_data import diagnostic_tail
from src.data.acceptance import accept_bundle, sha256

MODEL_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"


def run(args):
    root = Path(__file__).resolve().parents[1]
    work = args.workdir.resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = work / f"training-{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    env_dir = work / f".promptalign-training-{stamp}"
    python = env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME"):
        env.pop(key, None)
    env.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    report = {"status": "FAIL", "scope": "validation_only_no_holdout_no_release",
              "commands": [], "seeds": [42, 43, 44], "primary_seed": 42}

    def execute(name, arguments, interpreter=python):
        command = python_command(interpreter) + list(map(str, arguments))
        print(f"{name}: {output / (name + '.log')}", flush=True)
        with (output / f"{name}.log").open("w", encoding="utf-8") as stream:
            proc = subprocess.run(command, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT)
        content = (output / f"{name}.log").read_text(encoding="utf-8")
        report["commands"].append({"name": name, "returncode": proc.returncode})
        if proc.returncode or any(m in content for m in ("Error in sitecustomize", "Error in usercustomize")):
            report.update(failed_stage=name, diagnostic_tail=diagnostic_tail(content))
            print(report["diagnostic_tail"], flush=True)
            raise RuntimeError(f"{name} failed")

    try:
        if not args.accept_data_review:
            raise ValueError("Read the data review and explicitly accept its decisions before running")
        lock_path = accept_bundle(args.qa_archive, output / "data", args.reviewer)
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        manifest = output / "data/manifest.jsonl"
        report.update(data_lock_sha256=sha256(lock_path),
                      git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip())
        if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True).strip():
            raise ValueError("Tracked source modifications must be committed before official experiments")
        data_root = args.data_root or work / ".cache/genai-bench" / lock["dataset_revision"] / "data"
        missing = any(not (data_root / s["shard"]).is_file() for s in lock["sources"])
        if missing and not args.download_dataset:
            raise ValueError("Reviewed dataset missing: mount all shards or explicitly enable download")
        if missing and args.data_root:
            raise ValueError("Explicit data-root is incomplete; fix it or omit it to use the pinned download cache")
        execute("create_venv", ["-m", "venv", "--without-pip", env_dir], sys.executable)
        execute("bootstrap_pip", ["--pip-source", pip_source(), "install", "pip==26.2.1"])
        execute("startup", ["-c", "import sys; print(sys.version); assert sys.prefix != sys.base_prefix"])
        index = "cu126" if args.device == "cuda" else "cpu"
        execute("torch", ["-m", "pip", "install", f"torch==2.8.0+{index}",
                         "--index-url", f"https://download.pytorch.org/whl/{index}", "--progress-bar", "off"])
        execute("dependencies", ["-m", "pip", "install", "-r", "requirements-train.txt", "--progress-bar", "off"])
        execute("pip_check", ["-m", "pip", "check"])
        execute("imports", ["scripts/check_environment.py", "--output", output / "environment.json"])
        execute("versions", ["-m", "pip", "freeze", "--all"])
        execute("device", ["-c", "import torch,pyarrow,numpy; from PIL import Image; "
                 "print(torch.__version__, torch.version.cuda); "
                 + ("assert torch.cuda.is_available(), 'Enable a Kaggle GPU'; print(torch.cuda.get_device_name(0))"
                    if args.device == "cuda" else "print('CPU explicitly requested')")])
        execute("pipeline_tests", ["-m", "pytest", "tests/unit/test_metrics.py", "tests/unit/test_regression.py",
                                  "tests/unit/test_acceptance.py", "tests/integration/test_embeddings.py",
                                  "-q", "-p", "no:cacheprovider", "--basetemp", env_dir / "tests"])
        if missing:
            execute("download_dataset", ["scripts/download_dataset.py", "--allow-download",
                     "--revision", lock["dataset_revision"], "--destination", work / ".cache/genai-bench",
                     "--output", output / "dataset_download.json"])
        model = work / ".cache/models/clip-vit-base-patch32" / MODEL_REVISION
        execute("download_clip", ["scripts/download_clip.py", "--revision", MODEL_REVISION,
                                 "--destination", model, "--output", output / "clip_download.json"])
        execute("extract", ["scripts/extract_embeddings.py", "--manifest", manifest,
                "--data-root", data_root, "--model", model, "--output", output / "extraction",
                "--device", args.device, "--batch-size", args.batch_size, "--data-lock", lock_path])
        seed_results = []
        for seed in report["seeds"]:
            folder = output / f"training-seed{seed}"
            execute(f"train_seed{seed}", ["scripts/train_regression.py", "--manifest", manifest,
                    "--extraction", output / "extraction", "--output", folder, "--data-lock", lock_path,
                    "--seed", seed, "--epochs", "100", "--patience", "15"])
            seed_results.append(json.loads((folder / "result.json").read_text(encoding="utf-8")))
        report["validation_mae_across_seeds"] = {
            kind: {"mean": statistics.mean(r["metrics"][kind]["mae"] for r in seed_results),
                   "std_population": statistics.pstdev(r["metrics"][kind]["mae"] for r in seed_results)}
            for kind in ("mean", "ridge", "mlp")}
        report.update(status="PASS", primary_training="training-seed42",
                      selected=seed_results[0]["selected"],
                      selection_policy="Fixed primary seed 42; select head by validation MAE; seeds 43/44 report stability only")
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(report["error"], flush=True)
    finally:
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        archive = shutil.make_archive(str(output), "zip", root_dir=output)
        print(f"{report['status']}\nDownload from Kaggle Output / Files: {Path(archive).name}", flush=True)
    return 0 if report["status"] == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-archive", type=Path, required=True, help="Reviewed ZIP or its unpacked root directory")
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--accept-data-review", action="store_true")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--download-dataset", action="store_true")
    parser.add_argument("--workdir", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
