"""Canonical input: shard, row, prompt, images[model] bytes, ratings[model].

This contract is independent of the future Parquet adapter. Image bytes stay
outside the manifest; source coordinates identify the original encoded image.
"""

import hashlib
import json

from .validate import MODELS, normalize_prompt, validate_image, validate_ratings


def build_manifest(rows):
    records, rejected, seen = [], [], set()
    row_count = 0
    for row in rows:
        shard, index = row.get("shard"), row.get("row")
        if not isinstance(shard, str) or not shard.strip():
            raise ValueError("shard must be a nonempty source identifier")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("row must be a nonnegative integer")
        if (shard, index) in seen:
            raise ValueError(f"duplicate source row: {(shard, index)}")
        seen.add((shard, index))
        row_count += 1
        for model in MODELS:
            source = {"shard": shard, "row": index, "image_model": model}
            try:
                prompt = normalize_prompt(row.get("prompt"))
                ratings, target, std = validate_ratings(row.get("ratings", {}).get(model))
                payload = row.get("images", {}).get(model)
                dimensions = validate_image(payload)
            except (ValueError, AttributeError) as exc:
                rejected.append({**source, "reason": str(exc)})
                continue
            pair_id = json.dumps([shard, index, model], ensure_ascii=False, separators=(",", ":"))
            records.append({
                **source, "pair_id": pair_id, "prompt": row["prompt"],
                "prompt_group": prompt, "ratings": ratings, "target": target,
                "rating_std": std, "image_sha256": hashlib.sha256(payload).hexdigest(),
                **dimensions,
            })
    records.sort(key=lambda item: item["pair_id"])
    return records, {
        "source_rows": row_count, "expected_pairs": row_count * len(MODELS),
        "valid_pairs": len(records), "rejected_pairs": len(rejected),
        "rejections": rejected,
    }


def inference_input(record, image):
    """Explicit allowlist: labels and generator metadata never become features."""
    return {"prompt": record["prompt"], "image": image}
