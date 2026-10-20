"""Exercise real CLIP/head through HTTP on a synthetic image; development only."""

import argparse
from io import BytesIO
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from PIL import Image
from backend.app.main import create_app
from src.inference.predictor import Predictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--training", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    predictor = Predictor(args.model, args.training, allow_development=True)
    client = TestClient(create_app(predictor))
    stream = BytesIO()
    Image.new("RGB", (32, 32), "red").save(stream, format="PNG")
    payload = stream.getvalue()
    single = client.post("/api/evaluate", data={"prompt": "a red square"}, files={"image": ("a.png", payload)})
    batch = client.post("/api/compare", data={"prompt": "a red square"}, files=[("images", ("a.png", payload))] * 2)
    assert single.status_code == batch.status_code == 200
    assert 1 <= single.json()["data"]["items"][0]["score"] <= 5
    assert len(batch.json()["data"]["items"]) == 2
    result = {"status": "PASS", "scope": "real_encoder_synthetic_head_and_image_not_benchmark",
              "single": single.json(), "compare": batch.json()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print("PASS: real CLIP + local fixture head through HTTP")


if __name__ == "__main__":
    main()
