from io import BytesIO
import hashlib

import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.data.ingest import iter_rows
from src.data.manifest import build_manifest
from src.data.validate import MODELS
from src.training.extract_embeddings import extract_embeddings


class FakeEncoder:
    def encode(self, prompts, images):
        # Distinguishable image vectors make ordering mistakes observable.
        values = np.asarray([list(i.getpixel((0, 0))) for i in images], dtype=float)
        return {"text": values, "image": values, "cosine": np.zeros(len(images))}


def fixture_manifest(tmp_path):
    row = {"Prompt": "colors", "HumanRatings": {m: [1, 2, 3] for m in MODELS}}
    for index, model in enumerate(MODELS):
        stream = BytesIO()
        Image.new("RGB", (4, 4), (index, 0, 0)).save(stream, format="PNG")
        row[model] = {"bytes": stream.getvalue(), "path": None}
    pq.write_table(pa.Table.from_pylist([row]), tmp_path / "a.parquet")
    return build_manifest(iter_rows(tmp_path))[0]


def test_extraction_preserves_manifest_order_and_partial_batch(tmp_path):
    records = fixture_manifest(tmp_path)[::-1]
    output = extract_embeddings(records, tmp_path, FakeEncoder(), batch_size=4)
    assert output["pair_ids"].tolist() == [r["pair_id"] for r in records]
    assert output["image"][:, 0].tolist() == [MODELS.index(r["image_model"]) for r in records]


@pytest.mark.parametrize("field,value,error", [("image_sha256", "bad", "image differs"),
    ("prompt", "changed", "prompt differs"), ("row", 99, "missing source"),
    ("shard", "../outside.parquet", "outside")])
def test_extraction_rejects_stale_manifest(tmp_path, field, value, error):
    records = fixture_manifest(tmp_path)
    records[0][field] = value
    with pytest.raises(ValueError, match=error):
        extract_embeddings(records, tmp_path, FakeEncoder())
