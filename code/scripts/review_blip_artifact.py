"""Write the artifact receipt that pins the BLIP ITM model used by GatedPredictor.

Reads the already-cached snapshot (no download), records its revision and the
sha256 of every file, and stores the fusion rule plus the report that selected it.
"""

import argparse
import datetime as dt
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.inference.gated_predictor import RULE, resolve_snapshot, snapshot_checksums

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "configs/blip_itm_base_coco_receipt.json"
SOURCE_REPORT = "project_records/reports/results/G7_itm_gate_full_20261030.md"


def build_receipt(model, revision=None, source_report=SOURCE_REPORT):
    snapshot, revision = resolve_snapshot(model, revision)
    return {"model": model, "revision": revision, "rule": RULE, "source_report": source_report,
            "status": "REVIEWED_FOR_LOCAL_GATE", "reviewed_at": dt.date.today().isoformat(),
            "snapshot_path": str(snapshot), "files_sha256": snapshot_checksums(snapshot)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Salesforce/blip-itm-base-coco")
    parser.add_argument("--revision", default=None, help="snapshot commit; default refs/main of the cache")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true", help="overwrite an existing receipt")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"{args.output} exists; pass --force to rewrite it")
    receipt = build_receipt(args.model, args.revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"receipt written: {args.output} (revision {receipt['revision']}, {len(receipt['files_sha256'])} files)")


if __name__ == "__main__":
    main()
