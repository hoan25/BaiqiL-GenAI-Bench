"""Read local GenAI-Bench Parquet shards without Hugging Face downloads.

Column contract: https://huggingface.co/datasets/BaiqiL/GenAI-Bench
Images must have embedded bytes. External paths are deliberately not resolved.
"""

from pathlib import Path

import pyarrow.parquet as pq

from .validate import MODELS

REQUIRED_COLUMNS = ("Prompt", "HumanRatings", *MODELS)


def discover_shards(root):
    root = Path(root).resolve()
    paths = sorted(root.rglob("*.parquet"))
    if not paths:
        raise ValueError(f"no Parquet files under {root}")
    return paths


def iter_rows(root, paths=None, batch_size=8):
    root = Path(root).resolve()
    paths = discover_shards(root) if paths is None else paths
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    for path in paths:
        path = Path(path).resolve()
        shard = path.relative_to(root).as_posix()
        with pq.ParquetFile(path) as parquet:
            missing = set(REQUIRED_COLUMNS) - set(parquet.schema_arrow.names)
            if missing:
                raise ValueError(f"{shard}: missing columns {sorted(missing)}")
            offset = 0
            for batch in parquet.iter_batches(batch_size=batch_size, columns=list(REQUIRED_COLUMNS)):
                for raw in batch.to_pylist():
                    images = {}
                    for model in MODELS:
                        value = raw[model]
                        images[model] = value.get("bytes") if isinstance(value, dict) else value
                    yield {"shard": shard, "row": offset, "prompt": raw["Prompt"],
                           "ratings": raw["HumanRatings"], "images": images}
                    offset += 1
