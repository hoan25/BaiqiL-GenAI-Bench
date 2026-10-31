"""ITM gate experiment: locked CLIP MLP head x BLIP-ITM P(match), with prompt-swap sensitivity.

For every manifest record in --splits (default val only): encode with CLIP (head input),
score with BLIP-ITM, do the same with a prompt borrowed from another prompt group
(--swap-seed), then report metrics for each fusion rule in src/evaluation/itm_gate.py.
Holdout is only touched with --splits test given explicitly.
"""

import argparse
import csv
import hashlib
import json
from io import BytesIO
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image

from src.data.ingest import iter_rows
from src.data.qa import sha256_file
from src.data.validate import normalize_prompt
from src.evaluation.itm_gate import evaluate_rules
from src.models.clip_encoder import ClipEncoder
from src.models.regression_head import features, load_head


def load_images(records, data_root):
    wanted = {(r["shard"], r["row"], r["image_model"]): r for r in records}
    root = Path(data_root).resolve()
    paths = [root / s for s in sorted({r["shard"] for r in records})]
    out = {}
    for row in iter_rows(root, paths):
        for model, payload in row["images"].items():
            record = wanted.get((row["shard"], row["row"], model))
            if record is None:
                continue
            if row["prompt"] != record["prompt"] or hashlib.sha256(payload).hexdigest() != record["image_sha256"]:
                raise ValueError("source differs from manifest")
            out[record["pair_id"]] = payload
    if len(out) != len(records):
        raise ValueError("missing source rows")
    return out


def swapped_prompts(records, seed):
    rng = np.random.default_rng(seed)
    by_group = {}
    for r in records:
        by_group.setdefault(r["prompt_group"], r["prompt"])
    groups = sorted(by_group)
    if len(groups) < 2:
        raise ValueError("need at least two prompt groups to swap")
    result = []
    for r in records:
        choices = [g for g in groups if g != r["prompt_group"]]
        result.append(by_group[choices[rng.integers(len(choices))]])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--clip-model", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True, help="train_regression output dir (result.json + mlp.pt)")
    parser.add_argument("--itm-model", default="Salesforce/blip-itm-base-coco")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["val"])
    parser.add_argument("--swap-seed", type=int, default=1701)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {"status": "FAIL", "scope": "development_itm_gate_no_release", "splits": args.splits,
              "swap_seed": args.swap_seed, "itm_model": args.itm_model,
              "manifest_sha256": sha256_file(args.manifest)}
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

        records = [json.loads(l) for l in args.manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
        records = [r for r in records if r["split"] in args.splits]
        if not records:
            raise ValueError("no records in requested splits")
        images = load_images(records, args.data_root)
        swaps = swapped_prompts(records, args.swap_seed)

        def run(prompts):
            mlp, itm = [], []
            for i in range(0, len(records), args.batch_size):
                chunk = records[i:i + args.batch_size]
                pil = [Image.open(BytesIO(images[r["pair_id"]])).convert("RGB") for r in chunk]
                text = list(prompts[i:i + args.batch_size])
                encoded = clip.encode(text, pil)
                mlp.extend(head.predict(features(encoded["text"], encoded["image"])).tolist())
                inputs = processor(images=pil, text=text, return_tensors="pt", padding=True, truncation=True).to(args.device)
                with torch.inference_mode():
                    logits = blip(**inputs, use_itm_head=True).itm_score.float()
                itm.extend(torch.softmax(logits, -1)[:, 1].cpu().tolist())
                for p in pil:
                    p.close()
                if (i // args.batch_size) % 20 == 0:
                    print(f"{i + len(chunk)}/{len(records)} ({time.perf_counter() - started:.0f}s)", flush=True)
            return np.asarray(mlp), np.asarray(itm)

        mlp, itm = run([r["prompt"] for r in records])
        mlp_swap, itm_swap = run(swaps)
        with (args.output / "scores.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["pair_id", "split", "target", "mlp", "itm", "swap_prompt", "mlp_swap", "itm_swap"])
            for r, a, b, s, c, d in zip(records, mlp, itm, swaps, mlp_swap, itm_swap):
                writer.writerow([r["pair_id"], r["split"], r["target"], f"{a:.6f}", f"{b:.6f}", s, f"{c:.6f}", f"{d:.6f}"])
        report["results"] = {}
        for split in args.splits:
            idx = [i for i, r in enumerate(records) if r["split"] == split]
            target = [records[i]["target"] for i in idx]
            groups = [normalize_prompt(records[i]["prompt"]) for i in idx]
            report["results"][split] = {"count": len(idx), **evaluate_rules(
                target, mlp[idx], itm[idx], groups, mlp_swap[idx], itm_swap[idx])}
            report["results"][split]["itm_quantiles"] = {
                "original": np.quantile(itm[idx], [0, .1, .5, .9, 1]).round(4).tolist(),
                "swapped": np.quantile(itm_swap[idx], [0, .1, .5, .9, 1]).round(4).tolist()}
        report["status"] = "PASS_EXECUTION_NOT_MODEL_QUALITY"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["seconds"] = round(time.perf_counter() - started, 1)
        (args.output / "result.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        for split, res in report.get("results", {}).items():
            print(f"== {split} (n={res['count']})")
            for rule in ("mlp", "gate_multiply", "gate_min", "gate_sqrt", "gate_soft25", "match_only"):
                m = res[rule]
                print(f"{rule:14s} MAE {m['mae']:.4f} spearman {m['spearman']:.4f} pairwise {m['pairwise']['accuracy']:.4f} "
                      f"| swap {m['swap']['original_wins']}/{m['swap']['count']} drop {m['swap']['mean_drop']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
