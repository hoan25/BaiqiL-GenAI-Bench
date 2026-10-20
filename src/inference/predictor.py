import json
from pathlib import Path
from threading import Lock

from src.data.qa import sha256_file
from src.models.clip_encoder import ClipEncoder
from src.models.regression_head import load_head, features


class Predictor:
    def __init__(self, model_path, training_path, allow_development=False):
        training_path = Path(training_path)
        report = json.loads((training_path / "result.json").read_text(encoding="utf-8"))
        if report["status"] != "PASS":
            raise ValueError("training run did not pass")
        # A production release gate will be implemented after full-data acceptance.
        if not allow_development:
            raise ValueError("no accepted production artifact yet")
        self.scope = "development_only_not_dataset_benchmark"
        selected = report["selected"]
        checkpoint = report["checkpoints"][selected]
        path = (training_path / checkpoint["file"]).resolve()
        if not path.is_relative_to(training_path.resolve()) or sha256_file(path) != checkpoint["sha256"]:
            raise ValueError("checkpoint checksum/path mismatch")
        self.head = load_head(path)
        self.encoder = ClipEncoder(model_path)
        if self.encoder.fingerprint() != report["encoder_state_sha256"]:
            raise ValueError("encoder differs from training")
        if self.head.config["dimension"] != 2 * self.encoder.model.config.projection_dim:
            raise ValueError("head/encoder dimensions differ")
        self.version = f"{selected}-{checkpoint['sha256'][:12]}"
        self.metrics = {"scope": self.scope, "split": "val", "models": report["metrics"],
                        "manifest_sha256": report["manifest_sha256"], "selected": selected}
        self.lock = Lock()

    def predict(self, prompt, images):
        with self.lock:
            encoded = self.encoder.encode([prompt] * len(images), images)
            scores = self.head.predict(features(encoded["text"], encoded["image"]))
        return [{"score": float(score), "cosine": float(cosine)}
                for score, cosine in zip(scores, encoded["cosine"])]
