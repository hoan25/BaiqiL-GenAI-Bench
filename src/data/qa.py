"""QA artifacts for local data. Review candidates never imply acceptance."""

from collections import Counter, defaultdict
from difflib import SequenceMatcher
import hashlib
from itertools import combinations
import json
from pathlib import Path
import platform
import subprocess

import pyarrow
from PIL import __version__ as pillow_version

from .ingest import discover_shards, iter_rows
from .manifest import build_manifest
from .split import assign_splits


def sha256_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def duplicate_report(records, threshold=0.95):
    if not 0 < threshold <= 1:
        raise ValueError("near-duplicate threshold must be in (0, 1]")
    groups, exact = defaultdict(set), defaultdict(set)
    splits = defaultdict(set)
    for record in records:
        source = (record["shard"], record["row"])
        groups[record["prompt_group"]].add(source)
        exact[record["prompt"]].add(source)
        splits[record["prompt_group"]].add(record["split"])
    near = []
    for left, right in combinations(sorted(groups), 2):
        matcher = SequenceMatcher(None, left, right, autojunk=False)
        if matcher.quick_ratio() < threshold:
            continue
        score = matcher.ratio()
        if score >= threshold:
            near.append({"left": left, "right": right, "similarity": score,
                         "cross_split": splits[left] != splits[right]})
    return {
        "exact_duplicates": [{"prompt": p, "sources": sorted(s)} for p, s in sorted(exact.items()) if len(s) > 1],
        "normalized_duplicates": [{"prompt_group": p, "sources": sorted(s)} for p, s in sorted(groups.items()) if len(s) > 1],
        "normalized_leakage": sorted(p for p, s in splits.items() if len(s) > 1),
        "near_method": "difflib.SequenceMatcher(autojunk=False)",
        "near_threshold": threshold, "near_candidates": near,
        "near_limitations": "Lexical heuristic only; requires human review, not semantic deduplication.",
    }


def prepare_data(root, output, seed=42, expected_shards=12, expected_rows=1600,
                 near_threshold=0.95):
    root, output = Path(root).resolve(), Path(output)
    if expected_shards < 1 or expected_rows < 1:
        raise ValueError("expected counts must be positive")
    if not 0 < near_threshold <= 1:
        raise ValueError("near-duplicate threshold must be in (0, 1]")
    # Reserve a new run directory; never overwrite an existing manifest.
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "FAIL", "scope": "local_files_qa_not_full_acceptance",
              "seed": seed, "ratios": [0.7, 0.15, 0.15],
              "expected_shards": expected_shards, "expected_rows": expected_rows}
    try:
        repository = Path(__file__).resolve().parents[2]
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository,
                                  capture_output=True, text=True, check=False)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=normal"],
                               cwd=repository, capture_output=True, text=True, check=False)
        report["provenance"] = {
            "git_commit": revision.stdout.strip() if revision.returncode == 0 else None,
            "working_tree_dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
            "python": platform.python_version(), "pyarrow": pyarrow.__version__,
            "pillow": pillow_version,
        }
        paths = discover_shards(root)
        report["sources"] = [{"shard": p.relative_to(root).as_posix(),
                              "bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in paths]
        records, counts = build_manifest(iter_rows(root, paths))
        report.update(counts)
        records = assign_splits(records, seed=seed)
        report.update(duplicate_report(records, near_threshold))
        report["split_pairs"] = dict(Counter(r["split"] for r in records))
        report["split_groups"] = dict(Counter(split for _, split in
            {(r["prompt_group"], r["split"]) for r in records}))
        report["model_pairs"] = dict(Counter(r["image_model"] for r in records))
        issues = []
        if len(paths) != expected_shards:
            issues.append("shard_count_mismatch")
        if counts["source_rows"] != expected_rows:
            issues.append("row_count_mismatch")
        if counts["rejected_pairs"]:
            issues.append("rejected_pairs_require_review")
        if report["near_candidates"]:
            issues.append("near_duplicates_require_review")
        if report["normalized_leakage"]:
            issues.append("normalized_prompt_leakage")
        report["issues"] = issues
        report["status"] = "REVIEW_REQUIRED" if issues else "CHECKS_PASS"
        manifest = output / "manifest.jsonl"
        with manifest.open("x", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        report["manifest_sha256"] = sha256_file(manifest)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        with (output / "qa.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False)
    return report
