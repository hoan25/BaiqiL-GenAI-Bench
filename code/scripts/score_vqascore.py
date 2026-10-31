"""Zero-shot VQAScore over the reviewed GenAI-Bench manifest (train/val by default).

Usage on Kaggle (GPU, after `pip install t2v-metrics`):
  python scripts/score_vqascore.py --manifest <run>/data/manifest.jsonl \
      --data-root <parquet dir> --output /kaggle/working/vqascore-xl \
      --model clip-flant5-xl --device cuda --batch-size 8

Writes scores.csv (pair_id, split, target, score) and result.json with raw
Spearman/pairwise per split and train-fit linear/isotonic calibration MAE.
Test split is skipped unless --include-test is given explicitly.
Optional --external20 <dir> scores data/external20 against the 3-rater mean.
"""

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.data.qa import sha256_file
from src.evaluation.metrics import regression_metrics
from src.evaluation.vqascore import (apply_calibration, evaluate_scores, manifest_pairs,
                                     score_pairs)


def load_scorer(model, device, scorer_kind="vqascore"):
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; refusing CPU fallback")
    if scorer_kind == "vqascore":
        import t2v_metrics  # pip install t2v-metrics
        metric = t2v_metrics.VQAScore(model=model, device=device)

        def scorer(paths, prompts):
            dataset = [{"images": [p], "texts": [t]} for p, t in zip(paths, prompts)]
            with torch.inference_mode():
                out = metric.batch_forward(dataset=dataset, batch_size=len(dataset))
            return out.reshape(-1).float().cpu().tolist()
        return scorer
    if scorer_kind == "blip-itm":
        # Cross-attention image-text matching head; P(match). Small enough for a 4GB GPU.
        from PIL import Image
        from transformers import BlipForImageTextRetrieval, BlipProcessor
        processor = BlipProcessor.from_pretrained(model)
        net = BlipForImageTextRetrieval.from_pretrained(model).to(device).eval()

        def scorer(paths, prompts):
            images = [Image.open(p).convert("RGB") for p in paths]
            inputs = processor(images=images, text=list(prompts), return_tensors="pt",
                               padding=True, truncation=True).to(device)
            with torch.inference_mode():
                logits = net(**inputs, use_itm_head=True).itm_score
            return torch.softmax(logits.float(), dim=-1)[:, 1].cpu().tolist()
        return scorer
    raise ValueError("unknown scorer")


def external20_pairs(root):
    root = Path(root)
    with (root / "samples.csv").open(encoding="utf-8-sig", newline="") as f:
        samples = list(csv.DictReader(f))
    ratings = {}
    with (root / "ratings_corrected.csv").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            ratings.setdefault(row["sample_id"], []).append(float(row["score_1_to_5"]))
    pairs, targets = [], {}
    for s in samples:
        path = root / s["image_path"]
        if not path.is_file():
            candidates = list((root / "images").rglob(Path(s["image_path"]).name))
            if len(candidates) != 1:
                raise ValueError(f"cannot resolve image for sample {s['sample_id']}")
            path = candidates[0]
        pairs.append((s["sample_id"], s["evaluation_prompt_en"], path.read_bytes()))
        targets[s["sample_id"]] = statistics.fmean(ratings[s["sample_id"]])
    return pairs, targets


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="clip-flant5-xl",
                        help="t2v_metrics model name, or HF id such as Salesforce/blip-itm-base-coco for --scorer blip-itm")
    parser.add_argument("--scorer", choices=["vqascore", "blip-itm"], default="vqascore")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--include-test", action="store_true", help="Also score the holdout split (explicit)")
    parser.add_argument("--limit", type=int, help="Smoke test: only first N manifest records per split")
    parser.add_argument("--external20", type=Path, help="data/external20 directory")
    args = parser.parse_args()
    if bool(args.manifest) != bool(args.data_root):
        parser.error("--manifest and --data-root go together")
    if not args.manifest and not args.external20:
        parser.error("nothing to score")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "FAIL", "scope": "zero_shot_vqascore_development_no_release",
              "model": args.model, "scorer": args.scorer, "device": args.device, "batch_size": args.batch_size}
    started = time.perf_counter()
    try:
        scorer = load_scorer(args.model, args.device, args.scorer)
        if args.manifest:
            splits = ("train", "val", "test") if args.include_test else ("train", "val")
            records = [json.loads(l) for l in args.manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
            records = [r for r in records if r["split"] in splits]
            if args.limit:
                kept, counts = [], {}
                for r in records:
                    if counts.get(r["split"], 0) < args.limit:
                        kept.append(r); counts[r["split"]] = counts.get(r["split"], 0) + 1
                records = kept
                report["limit"] = args.limit
            report.update(manifest_sha256=sha256_file(args.manifest), splits=list(splits), count=len(records))

            def progress(done):
                if done % (args.batch_size * 25) == 0 or done == len(records):
                    print(f"scored {done}/{len(records)} ({time.perf_counter() - started:.0f}s)", flush=True)
            scores = score_pairs(manifest_pairs(records, args.data_root, splits), scorer,
                                 args.batch_size, progress=progress)
            with (args.output / "scores.csv").open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["pair_id", "split", "target", "score"])
                for r in records:
                    writer.writerow([r["pair_id"], r["split"], r["target"], f"{scores[r['pair_id']]:.8f}"])
            report["genai_bench"] = evaluate_scores(records, scores, splits)
        if args.external20:
            pairs, targets = external20_pairs(args.external20)
            ext = score_pairs(pairs, scorer, args.batch_size)
            ids = [k for k, _, _ in pairs]
            raw = np.asarray([ext[i] for i in ids]); target = np.asarray([targets[i] for i in ids])
            block = {"count": len(ids), "raw": regression_metrics(target, raw) | {"note": "MAE/RMSE of raw score are meaningless; see spearman"},
                     "reference_mlp42": {"mae": 1.456520, "rmse": 1.623083, "spearman": -0.307534}}
            cal = report.get("genai_bench", {}).get("calibration", {})
            for kind, item in cal.items():
                block[f"calibrated_{kind}"] = regression_metrics(target, apply_calibration(item["model"], raw))
            with (args.output / "external20_scores.csv").open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["sample_id", "target", "score"] + [f"calibrated_{k}" for k in cal])
                for i, t, s in zip(ids, target, raw):
                    writer.writerow([i, t, f"{s:.8f}"] + [f"{apply_calibration(cal[k]['model'], [s])[0]:.4f}" for k in cal])
            report["external20"] = block
        report["status"] = "PASS_EXECUTION_NOT_MODEL_QUALITY"
    except Exception as exc:  # keep partial evidence
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["seconds"] = round(time.perf_counter() - started, 1)
        (args.output / "result.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({k: report.get(k) for k in ("status", "model", "count", "seconds", "error")}, ensure_ascii=False))
        gb = report.get("genai_bench")
        if gb:
            for split, m in gb["splits"].items():
                print(f"{split}: raw spearman={m['raw_spearman']} pairwise={m['raw_pairwise'].get('accuracy')}")
            for kind, item in gb["calibration"].items():
                print(f"{kind} calibration val MAE={item['metrics'].get('val', {}).get('mae')}")
            print(f"reference MLP42 val: {gb['reference_mlp42_validation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
