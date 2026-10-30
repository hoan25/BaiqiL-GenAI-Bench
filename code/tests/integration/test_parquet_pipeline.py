from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.data.ingest import iter_rows
from src.data.qa import prepare_data
from src.data.validate import MODELS
from scripts.clean_python import python_command


def write_shard(path, prompts, corrupt=False):
    buffer = BytesIO()
    Image.new("RGB", (5, 7), "blue").save(buffer, format="PNG")
    rows = []
    for i, prompt in enumerate(prompts):
        row = {"Prompt": prompt, "HumanRatings": {m: [2, 3, 4] for m in MODELS}}
        row.update({m: {"bytes": buffer.getvalue(), "path": None} for m in MODELS})
        if corrupt and i == 0:
            row[MODELS[0]]["bytes"] = b"corrupt"
        rows.append(row)
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=1)


def test_read_multiple_shards_and_stable_manifest(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    write_shard(source / "a.parquet", ["red sun", "blue ocean"])
    write_shard(source / "b.parquet", ["green forest", " RED  SUN "])
    assert [r["row"] for r in iter_rows(source, batch_size=1)] == [0, 1, 0, 1]
    first = prepare_data(source, tmp_path / "one", expected_shards=2, expected_rows=4)
    second = prepare_data(source, tmp_path / "two", expected_shards=2, expected_rows=4)
    assert first["status"] == "CHECKS_PASS"
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert first["valid_pairs"] == 24
    assert len(first["normalized_duplicates"]) == 1
    assert not first["normalized_leakage"]
    assert len(first["sources"]) == 2
    original = (tmp_path / "one" / "manifest.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        prepare_data(source, tmp_path / "one")
    assert (tmp_path / "one" / "manifest.jsonl").read_bytes() == original


def test_rejections_counts_and_near_duplicates(tmp_path):
    write_shard(tmp_path / "a.parquet", ["a little cat sits on the wooden chair",
                 "a little cat sits on a wooden chair", "ocean waves"], corrupt=True)
    report = prepare_data(tmp_path, tmp_path / "out", near_threshold=0.9)
    assert report["status"] == "REVIEW_REQUIRED"
    assert report["valid_pairs"] == 17
    assert report["rejected_pairs"] == 1
    assert "shard_count_mismatch" in report["issues"]
    assert "row_count_mismatch" in report["issues"]
    assert report["near_candidates"]


def test_missing_column_failure_has_report(tmp_path):
    pq.write_table(pa.table({"Prompt": ["hello"]}), tmp_path / "bad.parquet")
    with pytest.raises(ValueError, match="missing columns"):
        prepare_data(tmp_path, tmp_path / "out")
    report = json.loads((tmp_path / "out" / "qa.json").read_text())
    assert report["status"] == "FAIL"
    assert "missing columns" in report["error"]


def test_cli_and_review_exit_code(tmp_path):
    write_shard(tmp_path / "a.parquet", ["sun", "ocean", "forest"])
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(python_command() + [str(root / "scripts/prepare_data.py"),
        "--input", str(tmp_path), "--output", str(tmp_path / "cli")],
        capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 2, result.stderr
    assert "REVIEW_REQUIRED" in result.stdout


def test_external_image_path_is_not_read(tmp_path):
    write_shard(tmp_path / "a.parquet", ["sun", "ocean", "forest"])
    rows = pq.read_table(tmp_path / "a.parquet").to_pylist()
    rows[0][MODELS[0]] = {"bytes": None, "path": "../../private.png"}
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "a.parquet")
    report = prepare_data(tmp_path, tmp_path / "out", expected_shards=1, expected_rows=3)
    assert report["rejected_pairs"] == 1
    assert "encoded bytes" in report["rejections"][0]["reason"]
