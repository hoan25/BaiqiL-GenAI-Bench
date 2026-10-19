"""Validate local Parquet shards and export a candidate split manifest."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.qa import prepare_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new run directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-shards", type=int, default=12)
    parser.add_argument("--expected-rows", type=int, default=1600)
    parser.add_argument("--near-threshold", type=float, default=0.95)
    args = parser.parse_args()
    try:
        report = prepare_data(args.input, args.output, args.seed, args.expected_shards,
                              args.expected_rows, args.near_threshold)
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"{report['status']}: {report['valid_pairs']} valid pairs; output: {args.output}")
    return 0 if report["status"] == "CHECKS_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
