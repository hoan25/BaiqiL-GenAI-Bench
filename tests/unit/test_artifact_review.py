import json
from types import SimpleNamespace
import zipfile

import pytest

from scripts.review_training import compare, review
from src.data.acceptance import sha256
from src.models.regression_head import RegressionHead, save_head
from src.inference.predictor import Predictor


def test_unsafe_archive_does_not_write_outside_run(tmp_path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("../outside.txt", "no")
    with pytest.raises(ValueError, match="Unsafe"):
        review(archive, tmp_path / "output")
    assert not (tmp_path / "outside.txt").exists()


def test_metric_replay_catches_changed_values():
    compare({"mae": 0.5, "spearman": None}, {"mae": 0.5, "spearman": None})
    with pytest.raises(ValueError):
        compare({"mae": 0.6}, {"mae": 0.5})


@pytest.fixture
def reviewed(tmp_path, monkeypatch):
    model = tmp_path / "clip"
    model.mkdir()
    (model / "config.json").write_text("{}")
    training = tmp_path / "training"
    training.mkdir()
    save_head(RegressionHead(4, "mean"), training / "mean.pt")
    report = {"status": "PASS", "scope": "reviewed_full_data_validation_only",
              "selected": "mean", "encoder_state_sha256": "fixture",
              "manifest_sha256": "fixture", "metrics": {},
              "checkpoints": {"mean": {"file": "mean.pt", "sha256": sha256(training / "mean.pt")}}}
    (training / "result.json").write_text(json.dumps(report))
    receipt = {"status": "VALIDATED_FOR_LOCAL_DEMO",
               "training_report_sha256": sha256(training / "result.json"),
               "model_files_sha256": {"config.json": sha256(model / "config.json")}}
    (training / "artifact_review.json").write_text(json.dumps(receipt))
    monkeypatch.setattr("src.inference.predictor.ClipEncoder", lambda path: SimpleNamespace(
        fingerprint=lambda: "fixture", model=SimpleNamespace(config=SimpleNamespace(projection_dim=2))))
    return model, training


def test_reviewed_head_loads_without_development_override(reviewed):
    predictor = Predictor(*reviewed)
    assert predictor.scope == "reviewed_full_data_validation_only"
    assert "holdout" in predictor.warnings[0]


@pytest.mark.parametrize("name", ["result.json", "mean.pt", "model"])
def test_modified_reviewed_artifact_is_rejected(reviewed, name):
    model, training = reviewed
    file = model / "config.json" if name == "model" else training / name
    with file.open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError):
        Predictor(model, training)
