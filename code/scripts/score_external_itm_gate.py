"""Score a loose external image set (not GenAI-Bench parquet) with the CLIP MLP head and BLIP-ITM.

Each row of --prompts (columns ``image_path`` or ``image_id``, and ``prompt``) is encoded with
the locked CLIP, scored by the MLP head, matched by BLIP-ITM, and fused with every rule in
src/evaluation/itm_gate.py (``gate_sqrt`` is the official one). With --ratings (long format
``image_id,rater,score``) the per-image mean rating becomes the target and MAE/RMSE/Spearman
are reported per rule. Status PASS_EXECUTION_NOT_MODEL_QUALITY only means the run finished.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image

from src.data.qa import sha256_file
from src.evaluation.itm_gate import RULES, fuse
from src.evaluation.metrics import regression_metrics
from src.models.clip_encoder import ClipEncoder
from src.models.regression_head import features, load_head

PROMPT_COLUMNS = ("prompt", "evaluation_prompt_en")
ID_COLUMNS = ("image_id", "sample_id")
RATING_ID_COLUMNS = ("image_id", "sample_id")
RATING_SCORE_COLUMNS = ("score", "score_1_to_5")


def _column(row, names, what):
    for name in names:
        if name in row and str(row[name]).strip():
            return str(row[name]).strip()
    raise ValueError(f"missing {what} column (one of {', '.join(names)})")


def resolve_image(images_dir, row):
    """Return (image_id, path) for a prompts.csv row; path must be an existing file."""
    images_dir = Path(images_dir)
    rel = str(row.get("image_path") or "").strip()
    explicit = next((str(row[c]).strip() for c in ID_COLUMNS if str(row.get(c) or "").strip()), None)
    if not rel and explicit is None:
        raise ValueError("each prompts row needs image_path or image_id")
    image_id = explicit or Path(rel).name
    name = Path(rel).name if rel else image_id
    for path in ([images_dir / rel] if rel else []) + [images_dir / name]:
        if path.is_file():
            return image_id, path
    found = sorted(p for p in images_dir.rglob(name) if p.is_file()) if images_dir.is_dir() else []
    if not found and not Path(name).suffix and images_dir.is_dir():
        found = sorted(p for p in images_dir.rglob(f"{name}.*") if p.is_file())
    if len(found) != 1:
        raise ValueError(f"cannot resolve image for {image_id}")
    return image_id, found[0]


def load_prompts(prompts_csv, images_dir):
    with Path(prompts_csv).open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("prompts file is empty")
    records, seen = [], set()
    for row in rows:
        image_id, path = resolve_image(images_dir, row)
        if image_id in seen:
            raise ValueError(f"duplicate image id {image_id}")
        seen.add(image_id)
        records.append({"image_id": image_id, "path": path, "prompt": _column(row, PROMPT_COLUMNS, "prompt")})
    return records


def load_ratings(ratings_csv, image_ids):
    """Mean rater score per image; ids match exactly or by stem (``01`` vs ``01.png``)."""
    scores = {}
    with Path(ratings_csv).open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            key = _column(row, RATING_ID_COLUMNS, "rating id")
            value = float(_column(row, RATING_SCORE_COLUMNS, "score"))
            if not 1 <= value <= 5:
                raise ValueError(f"rating out of [1,5] for {key}")
            scores.setdefault(key, []).append(value)
    by_stem = {}
    for key in scores:
        by_stem.setdefault(Path(key).stem, []).append(key)
    targets = {}
    for image_id in image_ids:
        keys = [image_id] if image_id in scores else by_stem.get(Path(image_id).stem, [])
        if len(keys) != 1:
            raise ValueError(f"no unique rating entry for {image_id}")
        targets[image_id] = statistics.fmean(scores[keys[0]])
    return targets


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompts", type=Path, required=True, help="CSV with image_path or image_id, and prompt")
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--ratings", type=Path, help="long CSV image_id,rater,score; target = mean per image")
    parser.add_argument("--clip-model", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True, help="train_regression output dir (result.json + mlp.pt)")
    parser.add_argument("--itm-model", default="Salesforce/blip-itm-base-coco")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {"status": "FAIL", "scope": "external_images_itm_gate", "official_rule": "gate_sqrt",
              "itm_model": args.itm_model, "prompts_sha256": sha256_file(args.prompts),
              "ratings_sha256": sha256_file(args.ratings) if args.ratings else None}
    try:
        training = json.loads((args.training / "result.json").read_text(encoding="utf-8"))
        checkpoint = training["checkpoints"]["mlp"]
        head_path = args.training / checkpoint["file"]
        if sha256_file(head_path) != checkpoint["sha256"]:
            raise ValueError("mlp checkpoint checksum mismatch")
        report["mlp_checkpoint_sha256"] = checkpoint["sha256"]
        head = load_head(head_path)
        clip = ClipEncoder(args.clip_model, args.device)
        if clip.fingerprint() != training["encoder_state_sha256"]:
            raise ValueError("CLIP differs from the one used for training")
        from transformers import BlipForImageTextRetrieval, BlipProcessor
        processor = BlipProcessor.from_pretrained(args.itm_model)
        blip = BlipForImageTextRetrieval.from_pretrained(args.itm_model).to(args.device).eval()

        records = load_prompts(args.prompts, args.images_dir)
        targets = load_ratings(args.ratings, [r["image_id"] for r in records]) if args.ratings else None
        report["images"] = [{"image_id": r["image_id"], "file": r["path"].name, "sha256": sha256_file(r["path"])}
                            for r in records]

        mlp, itm = [], []
        for i in range(0, len(records), args.batch_size):
            chunk = records[i:i + args.batch_size]
            pil = [Image.open(r["path"]).convert("RGB") for r in chunk]
            text = [r["prompt"] for r in chunk]
            encoded = clip.encode(text, pil)
            mlp.extend(head.predict(features(encoded["text"], encoded["image"])).tolist())
            inputs = processor(images=pil, text=text, return_tensors="pt", padding=True, truncation=True).to(args.device)
            with torch.inference_mode():
                logits = blip(**inputs, use_itm_head=True).itm_score.float()
            itm.extend(torch.softmax(logits, -1)[:, 1].cpu().tolist())
            for p in pil:
                p.close()
            print(f"{i + len(chunk)}/{len(records)} ({time.perf_counter() - started:.0f}s)", flush=True)
        mlp, itm = np.asarray(mlp), np.asarray(itm)
        fused = {rule: fuse(rule, mlp, itm) for rule in RULES}

        with (args.output / "scores.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_id", "prompt", "mlp", "itm"] + (["target"] if targets else []) + list(RULES))
            for k, r in enumerate(records):
                row = [r["image_id"], r["prompt"], f"{mlp[k]:.6f}", f"{itm[k]:.6f}"]
                if targets:
                    row.append(f"{targets[r['image_id']]:.6f}")
                writer.writerow(row + [f"{fused[rule][k]:.6f}" for rule in RULES])
        report["count"] = len(records)
        report["itm_quantiles"] = np.quantile(itm, [0, .1, .5, .9, 1]).round(4).tolist()
        if targets:
            target = np.asarray([targets[r["image_id"]] for r in records])
            report["results"] = {rule: regression_metrics(target, fused[rule]) for rule in RULES}
        report["status"] = "PASS_EXECUTION_NOT_MODEL_QUALITY"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["seconds"] = round(time.perf_counter() - started, 1)
        (args.output / "result.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        for rule, m in report.get("results", {}).items():
            spearman = "n/a" if m["spearman"] is None else f"{m['spearman']:.4f}"
            print(f"{rule:14s} MAE {m['mae']:.4f} RMSE {m['rmse']:.4f} spearman {spearman}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
