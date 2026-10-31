"""End-to-end CLI test for scripts/score_external_itm_gate.py with synthetic PNGs.

Real torch and the real regression head are used; ``transformers`` is replaced through
PYTHONPATH with a stub that provides CLIP and BLIP objects with deterministic outputs.
"""

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest
import torch

from scripts.score_external_itm_gate import load_prompts, load_ratings, resolve_image
from src.data.qa import sha256_file
from src.evaluation.itm_gate import RULES
from src.models.regression_head import RegressionHead, save_head

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "score_external_itm_gate.py"
DIM = 4  # stub embedding width; head dimension is 2 * DIM

FAKE_TRANSFORMERS = """
import torch

DIM = %d


class _Batch(dict):
    def to(self, device):
        return self


class _Output:
    def __init__(self, **values):
        self.__dict__.update(values)


def _signature(images):
    # Mean red channel in [0,1]; deterministic per image colour.
    return torch.tensor([float(sum(p[0] for p in im.getdata()) / (255 * im.width * im.height)) for im in images])


class _Config:
    class text_config:
        max_position_embeddings = 77


class CLIPProcessor:
    @classmethod
    def from_pretrained(cls, path, **kwargs):
        return cls()

    def __call__(self, text, images, **kwargs):
        return _Batch(text=list(text), images=list(images))


class CLIPModel(torch.nn.Module):
    config = _Config()

    def __init__(self):
        super().__init__()
        self.register_buffer("weight", torch.arange(DIM, dtype=torch.float32))

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        return cls()

    def forward(self, text, images):
        sig = _signature(images)
        image = torch.stack([sig, 1 - sig, sig * sig, torch.ones_like(sig)], dim=1)
        length = torch.tensor([float(len(t)) / 100 for t in text])
        text_embeds = torch.stack([length, 1 - length, length * 0.5, torch.ones_like(length)], dim=1)
        return _Output(text_embeds=text_embeds, image_embeds=image)


class BlipProcessor:
    @classmethod
    def from_pretrained(cls, path, **kwargs):
        return cls()

    def __call__(self, images, text, **kwargs):
        return _Batch(images=list(images), text=list(text))


class BlipForImageTextRetrieval(torch.nn.Module):
    @classmethod
    def from_pretrained(cls, path, **kwargs):
        return cls()

    def forward(self, images, text, use_itm_head=True):
        sig = _signature(images)
        logit = (sig - 0.5) * 8  # red images match, others do not
        return _Output(itm_score=torch.stack([-logit, logit], dim=1))
""" % DIM


def png(path, color):
    Image.new("RGB", (4, 4), color).save(path, format="PNG")


def clip_fingerprint():
    digest = hashlib.sha256()
    state = {"weight": torch.arange(DIM, dtype=torch.float32)}
    for name, tensor in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def write_training(tmp_path):
    training = tmp_path / "training"
    training.mkdir()
    head = RegressionHead(2 * DIM, "mlp", hidden=3)
    save_head(head, training / "mlp.pt")
    (training / "result.json").write_text(json.dumps({
        "encoder_state_sha256": clip_fingerprint(),
        "checkpoints": {"mlp": {"file": "mlp.pt", "sha256": sha256_file(training / "mlp.pt")}}}), encoding="utf-8")
    return training


def write_dataset(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    for name, color in [("01.png", "red"), ("02.png", "blue"), ("03.png", "green")]:
        png(images / name, color)
    prompts = tmp_path / "prompts.csv"
    prompts.write_text("image_path,prompt\nimages/01.png,a red square\nimages/02.png,a blue square\n"
                       "images/03.png,a green square\n", encoding="utf-8")
    ratings = tmp_path / "ratings.csv"
    ratings.write_text("﻿image_id,rater,score\n01.png,a,5\n01.png,b,4\n02.png,a,1\n02.png,b,2\n03.png,a,3\n03.png,b,3\n",
                       encoding="utf-8")
    return images, prompts, ratings


def run_cli(tmp_path, *extra):
    stubs = tmp_path / "stubs"
    stubs.mkdir(exist_ok=True)
    (stubs / "transformers.py").write_text(FAKE_TRANSFORMERS, encoding="utf-8")
    clip_dir = tmp_path / "clip"
    clip_dir.mkdir(exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k in ("SYSTEMROOT", "PATH", "TEMP", "TMP", "USERPROFILE", "HOME")}
    env["PYTHONPATH"] = str(stubs)
    return subprocess.run([sys.executable, str(SCRIPT), "--clip-model", str(clip_dir), "--device", "cpu", *extra],
                          capture_output=True, text=True, env=env)


def test_cli_end_to_end_with_ratings(tmp_path):
    training = write_training(tmp_path)
    images, prompts, ratings = write_dataset(tmp_path)
    output = tmp_path / "out"
    proc = run_cli(tmp_path, "--prompts", str(prompts), "--images-dir", str(images), "--ratings", str(ratings),
                   "--training", str(training), "--output", str(output), "--batch-size", "2")
    assert proc.returncode == 0, proc.stderr
    report = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert report["status"] == "PASS_EXECUTION_NOT_MODEL_QUALITY"
    assert report["official_rule"] == "gate_sqrt" and report["count"] == 3
    assert set(report["results"]) == set(RULES)
    for metrics in report["results"].values():
        assert set(metrics) >= {"mae", "rmse", "spearman"} and metrics["count"] == 3
    assert [i["image_id"] for i in report["images"]] == ["01.png", "02.png", "03.png"]
    assert report["images"][0]["sha256"] == sha256_file(images / "01.png")
    assert report["mlp_checkpoint_sha256"] == sha256_file(training / "mlp.pt")
    with (output / "scores.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["image_id"] for r in rows] == ["01.png", "02.png", "03.png"]
    assert set(rows[0]) == {"image_id", "prompt", "mlp", "itm", "target"} | set(RULES)
    assert [float(r["target"]) for r in rows] == pytest.approx([4.5, 1.5, 3])
    itm = [float(r["itm"]) for r in rows]
    assert itm[0] > 0.9 and itm[1] < 0.1  # red matches, blue does not
    for r in rows:
        s, m = float(r["mlp"]), float(r["itm"])
        assert 1 <= s <= 5 and 0 <= m <= 1
        assert float(r["gate_sqrt"]) == pytest.approx(1 + (s - 1) * m ** 0.5, abs=1e-4)
        assert float(r["match_only"]) == pytest.approx(1 + 4 * m, abs=1e-4)
        assert 1 <= float(r["gate_multiply"]) <= s + 1e-6


def test_cli_without_ratings_and_rejects_existing_output(tmp_path):
    training = write_training(tmp_path)
    images, prompts, _ = write_dataset(tmp_path)
    output = tmp_path / "out"
    proc = run_cli(tmp_path, "--prompts", str(prompts), "--images-dir", str(images),
                   "--training", str(training), "--output", str(output))
    assert proc.returncode == 0, proc.stderr
    report = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert report["status"] == "PASS_EXECUTION_NOT_MODEL_QUALITY" and "results" not in report
    with (output / "scores.csv").open(encoding="utf-8", newline="") as f:
        assert "target" not in csv.DictReader(f).fieldnames
    proc = run_cli(tmp_path, "--prompts", str(prompts), "--images-dir", str(images),
                   "--training", str(training), "--output", str(output))
    assert proc.returncode != 0


def test_cli_fails_on_checkpoint_mismatch(tmp_path):
    training = write_training(tmp_path)
    images, prompts, _ = write_dataset(tmp_path)
    meta = json.loads((training / "result.json").read_text(encoding="utf-8"))
    meta["checkpoints"]["mlp"]["sha256"] = "0" * 64
    (training / "result.json").write_text(json.dumps(meta), encoding="utf-8")
    output = tmp_path / "out"
    proc = run_cli(tmp_path, "--prompts", str(prompts), "--images-dir", str(images),
                   "--training", str(training), "--output", str(output))
    assert proc.returncode != 0
    report = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert report["status"] == "FAIL" and "checksum mismatch" in report["error"]


def test_loaders_accept_external20_and_v2_column_names(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    png(images / "01.png", "red")
    png(images / "02.png", "blue")
    prompts = tmp_path / "samples.csv"
    prompts.write_text("﻿sample_id,image_path,evaluation_prompt_en\n01,images/01.png,a red square\n"
                       "02,02.png,a blue square\n", encoding="utf-8")
    records = load_prompts(prompts, images)
    assert [(r["image_id"], r["path"].name, r["prompt"]) for r in records] == [
        ("01", "01.png", "a red square"), ("02", "02.png", "a blue square")]
    ratings = tmp_path / "ratings_corrected.csv"
    ratings.write_text("sample_id,reviewer,score_1_to_5,comment\n01,x,5,ok\n01,y,3,ok\n02,x,2,ok\n", encoding="utf-8")
    assert load_ratings(ratings, ["01", "02"]) == pytest.approx({"01": 4, "02": 2})
    # Stem matching works both ways: ratings keyed by file name, prompts keyed by bare id.
    ratings.write_text("image_id,rater,score\n01.png,x,5\n02.png,x,1\n", encoding="utf-8")
    assert load_ratings(ratings, ["01", "02"]) == pytest.approx({"01": 5, "02": 1})
    with pytest.raises(ValueError):
        load_ratings(ratings, ["03"])
    with pytest.raises(ValueError):
        resolve_image(images, {"image_path": "images/99.png"})
    with pytest.raises(ValueError):
        resolve_image(images, {"prompt": "no id"})
    assert resolve_image(images, {"image_id": "02"}) == ("02", images / "02.png")
