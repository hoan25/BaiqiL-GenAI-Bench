"""Extract one embedding pair per manifest record, preserving pair IDs."""

from io import BytesIO
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.ingest import iter_rows


def extract_embeddings(records, data_root, encoder, batch_size=16):
    if not records or batch_size < 1:
        raise ValueError("nonempty manifest and positive batch size required")
    ids = [r["pair_id"] for r in records]
    keys = [(r["shard"], r["row"], r["image_model"]) for r in records]
    if len(set(ids)) != len(ids) or len(set(keys)) != len(keys):
        raise ValueError("duplicate pair ID or source coordinate")
    root = Path(data_root).resolve()
    paths = []
    for shard in sorted({r["shard"] for r in records}):
        path = (root / shard).resolve()
        if not path.is_relative_to(root):
            raise ValueError("shard path outside data root")
        paths.append(path)
    wanted = {key: i for i, key in enumerate(keys)}
    pending = []
    results = {}

    def flush():
        if not pending:
            return
        images = []
        try:
            for _, payload in pending:
                with Image.open(BytesIO(payload)) as image:
                    images.append(image.convert("RGB"))
            output = encoder.encode([records[i]["prompt"] for i, _ in pending], images)
            text, image, cosine = (np.asarray(output[name]) for name in ("text", "image", "cosine"))
            if (text.ndim != 2 or image.shape != text.shape or text.shape[0] != len(pending)
                    or cosine.shape != (len(pending),) or text.shape[1] == 0):
                raise ValueError("encoder output shape mismatch")
            if not all(np.isfinite(a).all() for a in (text, image, cosine)):
                raise ValueError("nonfinite encoder output")
            for position, (index, _) in enumerate(pending):
                results[index] = (text[position], image[position], cosine[position])
        finally:
            for image in images:
                image.close()
        pending.clear()

    for row in iter_rows(root, paths):
        for model, payload in row["images"].items():
            index = wanted.get((row["shard"], row["row"], model))
            if index is None:
                continue
            record = records[index]
            if row["prompt"] != record["prompt"]:
                raise ValueError("source prompt differs from manifest")
            if not isinstance(payload, bytes) or hashlib.sha256(payload).hexdigest() != record["image_sha256"]:
                raise ValueError("source image differs from manifest")
            pending.append((index, payload))
            if len(pending) >= batch_size:
                flush()
    flush()
    if len(results) != len(records):
        raise ValueError("manifest contains missing source rows or models")
    return {"pair_ids": np.asarray(ids),
            "text": np.stack([results[i][0] for i in range(len(records))]),
            "image": np.stack([results[i][1] for i in range(len(records))]),
            "cosine": np.asarray([results[i][2] for i in range(len(records))])}
