"""Explicit opt-in download of GenAI-Bench Parquet only, pinned to resolved revision."""

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-download", action="store_true", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    revision = HfApi().dataset_info("BaiqiL/GenAI-Bench").sha
    # Revision directory preserves any earlier dataset version.
    destination = args.destination / revision
    snapshot_download("BaiqiL/GenAI-Bench", repo_type="dataset", revision=revision,
                      local_dir=destination, allow_patterns=["data/*.parquet"], max_workers=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump({"dataset": "BaiqiL/GenAI-Bench", "revision": revision,
                   "data_root": str(destination / "data")}, stream, indent=2)


if __name__ == "__main__":
    main()
