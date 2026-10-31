"""GatedPredictor logic with stubbed encoder/head/BLIP; no model weights are loaded."""

import json
from threading import Lock

import numpy as np
from PIL import Image
import pytest

from src.inference import gated_predictor as gp
from src.inference.gated_predictor import GatedPredictor, NON_ENGLISH_WARNING


class StubEncoder:
    def encode(self, prompts, images):
        n = len(images)
        return {"text": np.ones((n, 4), dtype=np.float32), "image": np.ones((n, 4), dtype=np.float32),
                "cosine": np.full(n, 0.25)}


class StubHead:
    def __init__(self, values):
        self.values = values

    def predict(self, x):
        return np.asarray(self.values[: len(x)], dtype=float)


def make_predictor(mlp_values, itm_values):
    predictor = GatedPredictor.__new__(GatedPredictor)
    predictor.encoder = StubEncoder()
    predictor.head = StubHead(mlp_values)
    predictor.lock = Lock()
    predictor.match_probability = lambda prompt, images: np.asarray(itm_values[: len(images)], dtype=float)
    predictor.version, predictor.scope, predictor.warnings = "stub-gate", "test_fixture", []
    predictor.metrics = {"scope": "test_fixture", "models": {}, "split": "val"}
    return predictor


def images(n):
    return [Image.new("RGB", (8, 8), "red") for _ in range(n)]


def test_english_prompt_is_gated_with_sqrt_rule():
    predictor = make_predictor([5.0, 3.0], [0.25, 1.0])
    items = predictor.predict("a red square", images(2))
    assert items[0] == {"score": pytest.approx(3.0), "cosine": 0.25, "mlp": 5.0, "itm": 0.25, "gated": True}
    assert items[1]["score"] == pytest.approx(3.0) and items[1]["itm"] == 1.0


def test_head_output_is_clipped_before_fusion():
    predictor = make_predictor([5.7, 0.2], [1.0, 1.0])
    scores = [i["score"] for i in predictor.predict("a red square", images(2))]
    assert scores == [5.0, 1.0]


def test_vietnamese_prompt_skips_gate_and_warns():
    calls = []
    predictor = make_predictor([4.0], [0.0])
    predictor.match_probability = lambda prompt, images: calls.append(prompt) or np.zeros(1)
    item = predictor.predict("một con mèo đỏ", images(1))[0]
    assert item == {"score": 4.0, "cosine": 0.25, "mlp": 4.0, "itm": None, "gated": False,
                    "warning": NON_ENGLISH_WARNING}
    assert calls == []


def test_predict_releases_lock_on_error():
    predictor = make_predictor([3.0], [0.5])
    predictor.match_probability = lambda prompt, images: (_ for _ in ()).throw(RuntimeError("blip"))
    with pytest.raises(RuntimeError):
        predictor.predict("a red square", images(1))
    assert predictor.lock.acquire(blocking=False)


def fake_snapshot(tmp_path, name="snap"):
    snapshot = tmp_path / name
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "pytorch_model.bin").write_bytes(b"weights")
    return snapshot


def receipt_for(snapshot, revision):
    return {"rule": gp.RULE, "revision": revision, "files_sha256": gp.snapshot_checksums(snapshot)}


def test_verify_receipt_accepts_identical_snapshot(tmp_path):
    snapshot = fake_snapshot(tmp_path)
    gp.verify_receipt(receipt_for(snapshot, "snap"), snapshot, "snap")


@pytest.mark.parametrize("mutate,message", [
    (lambda r, s: r.update(revision="other"), "revision"),
    (lambda r, s: r.update(rule="gate_multiply"), "rule"),
    (lambda r, s: (s / "pytorch_model.bin").write_bytes(b"tampered"), "differs"),
    (lambda r, s: (s / "extra.bin").write_bytes(b"x"), "file set"),
    (lambda r, s: r.update(files_sha256={}), "no files"),
])
def test_verify_receipt_rejects_mismatch(tmp_path, mutate, message):
    snapshot = fake_snapshot(tmp_path)
    receipt = receipt_for(snapshot, "snap")
    mutate(receipt, snapshot)
    with pytest.raises(ValueError, match=message):
        gp.verify_receipt(receipt, snapshot, "snap")


def test_resolve_snapshot_reads_refs_main_from_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    repo = tmp_path / "models--Org--blip"
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "main").write_text("abc123\n", encoding="utf-8")
    snapshot = fake_snapshot(repo / "snapshots", "abc123")
    assert gp.resolve_snapshot("Org/blip") == (snapshot.resolve(), "abc123")
    assert gp.resolve_snapshot(str(snapshot)) == (snapshot.resolve(), "abc123")
    with pytest.raises(ValueError, match="missing"):
        gp.resolve_snapshot("Org/blip", "deadbeef")
    with pytest.raises(ValueError):
        gp.resolve_snapshot("Org/missing")
    with pytest.raises(ValueError):
        gp.resolve_snapshot("../escape")


def test_shipped_receipt_matches_module_rule():
    receipt = json.loads((gp.Path(__file__).resolve().parents[2] / "configs/blip_itm_base_coco_receipt.json")
                         .read_text(encoding="utf-8"))
    assert receipt["rule"] == gp.RULE == "gate_sqrt"
    assert receipt["model"] == "Salesforce/blip-itm-base-coco"
    assert len(receipt["revision"]) == 40 and len(receipt["files_sha256"]) >= 3
    assert all(len(v) == 64 for v in receipt["files_sha256"].values())
    assert receipt["source_report"].endswith("G7_itm_gate_full_20261030.md")

