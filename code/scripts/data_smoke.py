"""Offline G1 contract smoke. Synthetic data only, never a dataset QA report."""

import argparse
from io import BytesIO
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image
from src.data.manifest import build_manifest
from src.data.split import assign_splits
from src.data.validate import MODELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    buffer = BytesIO()
    Image.new("RGB", (8, 8), "red").save(buffer, format="PNG")
    rows = [{"shard": "synthetic", "row": i, "prompt": f"synthetic prompt {i}",
             "images": dict.fromkeys(MODELS, buffer.getvalue()),
             "ratings": {model: [2, 4, 3] for model in MODELS}} for i in range(10)]
    records, report = build_manifest(rows)
    records = assign_splits(records)
    result = {"scope": "synthetic_contract_only", "seed": 42, **report,
              "split_counts": {name: sum(r["split"] == name for r in records)
                               for name in ("train", "val", "test")}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Evidence files are never silently overwritten.
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
