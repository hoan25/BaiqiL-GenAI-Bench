from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.data.manifest import build_manifest
from src.data.ingest import iter_rows
from src.data.split import assign_splits
from src.data.validate import MODELS
from src.evaluation.vqascore import (apply_calibration, evaluate_scores, fit_isotonic, fit_linear,
                                     manifest_pairs, score_pairs)

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "score_vqascore.py"


def png(color):
    buffer = BytesIO()
    Image.new("RGB", (4, 4), color).save(buffer, format="PNG")
    return buffer.getvalue()


def write_shard(path, prompts):
    rows = []
    for i, prompt in enumerate(prompts):
        row = {"Prompt": prompt, "HumanRatings": {m: [1 + (i + j) % 5] * 3 for j, m in enumerate(MODELS)}}
        row.update({m: {"bytes": png((20 * i, 30 * j, 0)), "path": None} for j, m in enumerate(MODELS)})
        rows.append(row)
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=1)


def make_records(tmp_path, prompts):
    write_shard(tmp_path / "a.parquet", prompts)
    records, _ = build_manifest(iter_rows(tmp_path))
    return assign_splits(records, seed=1)


def fake_scorer(paths, prompts):
    # Deterministic score from the image bytes and prompt; proves pairing is preserved.
    out = []
    for path, prompt in zip(paths, prompts):
        data = open(path, "rb").read()
        out.append((sum(data) % 97) / 97 + len(prompt) * 1e-3)
    return out


def test_score_pairs_batches_preserve_keys_and_clean_temp(tmp_path):
    pairs = [(f"k{i}", f"prompt {i}", png((i, i, i))) for i in range(5)]
    expected = {}
    for key, prompt, payload in pairs:
        file = tmp_path / f"{key}.png"
        file.write_bytes(payload)
        expected[key] = fake_scorer([str(file)], [prompt])[0]
    seen = []
    result = score_pairs(pairs, fake_scorer, batch_size=2, workdir=tmp_path, progress=seen.append)
    assert result == pytest.approx(expected)
    assert seen == [2, 4, 5]
    assert not [p for p in tmp_path.iterdir() if p.is_dir()]


def test_score_pairs_rejects_bad_scorer_and_inputs():
    pairs = [("a", "x", png("red")), ("b", "y", png("blue"))]
    with pytest.raises(ValueError):
        score_pairs(pairs, lambda p, t: [1.0], batch_size=8)
    with pytest.raises(ValueError):
        score_pairs(pairs, lambda p, t: [float("nan")] * len(p), batch_size=8)
    with pytest.raises(ValueError):
        score_pairs([("a", "", png("red"))], fake_scorer)
    with pytest.raises(ValueError):
        score_pairs(pairs + [("a", "z", png("red"))], fake_scorer, batch_size=1)


def test_manifest_pairs_filters_splits_and_verifies_bytes(tmp_path):
    records = make_records(tmp_path, ["one cat", "two dogs", "red car", "blue sky"])
    chosen = {r["split"] for r in records}
    assert chosen >= {"train", "val"}
    pairs = list(manifest_pairs(records, tmp_path, ("train",)))
    ids = {r["pair_id"] for r in records if r["split"] == "train"}
    assert {k for k, _, _ in pairs} == ids
    tampered = [dict(r, image_sha256="0" * 64) for r in records]
    with pytest.raises(ValueError):
        list(manifest_pairs(tampered, tmp_path, ("train",)))


def test_evaluate_scores_calibrates_on_train_only(tmp_path):
    records = make_records(tmp_path, ["one cat", "two dogs", "red car", "blue sky", "green hat", "old man"])
    scores = {r["pair_id"]: r["target"] * 0.1 + 0.05 for r in records}  # perfectly monotone
    report = evaluate_scores(records, scores, ("train", "val"))
    assert set(report["splits"]) == {"train", "val"}
    assert report["splits"]["train"]["raw_spearman"] == pytest.approx(1)
    assert report["splits"]["train"]["raw_pairwise"]["accuracy"] in (None, 1)
    for kind in ("linear", "isotonic"):
        assert report["calibration"][kind]["metrics"]["train"]["mae"] < 1e-6
        assert report["calibration"][kind]["metrics"]["val"]["mae"] < 0.5
    with pytest.raises(ValueError):
        evaluate_scores(records, {}, ("train",))


def test_isotonic_is_monotone_and_pools_violators():
    model = fit_isotonic([1, 2, 3, 4], [1, 3, 2, 5])
    y = apply_calibration(model, [0, 1, 2, 2.5, 3, 4, 9])
    assert all(np.diff(y) >= 0)
    assert y[1] == pytest.approx(1) and y[2] == pytest.approx(2.5) and y[-1] == pytest.approx(5)
    linear = fit_linear([0, 1], [1, 5])
    assert apply_calibration(linear, [0.5, 2]).tolist() == pytest.approx([3, 5])
    with pytest.raises(ValueError):
        fit_linear([1, 1], [1, 2])


def test_cli_rejects_incomplete_arguments(tmp_path):
    proc = subprocess.run([sys.executable, str(SCRIPT), "--output", tmp_path / "o",
                           "--manifest", "x"], capture_output=True, text=True)
    assert proc.returncode == 2
    assert "go together" in proc.stderr


FAKE_TORCH = """
import contextlib
class _Cuda:
    @staticmethod
    def is_available():
        return False
cuda = _Cuda()
inference_mode = contextlib.nullcontext
"""
FAKE_T2V = """
class _Out:
    def __init__(self, values):
        self.values = values
    def reshape(self, *_):
        return self
    def float(self):
        return self
    def cpu(self):
        return self
    def tolist(self):
        return self.values
class VQAScore:
    def __init__(self, model, device):
        assert model and device == "cpu"
    def batch_forward(self, dataset, batch_size):
        return _Out([len(d["texts"][0]) / 100 + len(open(d["images"][0], "rb").read()) % 7 / 10 for d in dataset])
"""


def test_cli_end_to_end_with_stub_scorer(tmp_path, monkeypatch):
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    (stubs / "torch.py").write_text(FAKE_TORCH)
    (stubs / "t2v_metrics.py").write_text(FAKE_T2V)
    data = tmp_path / "data"
    data.mkdir()
    records = make_records(data, ["one cat", "two dogs", "red car", "blue sky", "green hat", "old man"])
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    ext = tmp_path / "external20"
    (ext / "images").mkdir(parents=True)
    (ext / "images/01.png").write_bytes(png("red"))
    (ext / "images/02.png").write_bytes(png("blue"))
    (ext / "samples.csv").write_text("sample_id,image_path,evaluation_prompt_en\n01,images/01.png,a red square\n02,images/02.png,a blue square\n")
    (ext / "ratings_corrected.csv").write_text("sample_id,reviewer,score_1_to_5\n01,a,5\n01,b,4\n02,a,1\n02,b,2\n")
    output = tmp_path / "out"
    env = {"PYTHONPATH": str(stubs), "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", "")}
    proc = subprocess.run([sys.executable, str(SCRIPT), "--manifest", manifest, "--data-root", data,
                           "--output", output, "--device", "cpu", "--batch-size", "4", "--external20", ext],
                          capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr
    report = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert report["status"] == "PASS_EXECUTION_NOT_MODEL_QUALITY"
    assert set(report["genai_bench"]["splits"]) == {"train", "val"}
    assert report["count"] == sum(r["split"] in ("train", "val") for r in records)
    assert set(report["genai_bench"]["calibration"]) == {"linear", "isotonic"}
    assert report["external20"]["count"] == 2
    assert "calibrated_linear" in report["external20"]
    with (output / "scores.csv").open(encoding="utf-8", newline="") as f:
        rows = list(__import__("csv").DictReader(f))
    assert len(rows) == report["count"] and {r["split"] for r in rows} == {"train", "val"}
    assert {r["pair_id"] for r in rows} == {r["pair_id"] for r in records if r["split"] != "test"}
    assert len((output / "external20_scores.csv").read_text().splitlines()) == 3
