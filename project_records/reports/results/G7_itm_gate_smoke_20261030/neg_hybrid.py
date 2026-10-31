"""Candidate: concat + ITM + cosine head trained with prompt-swap negatives (label 1).
Reports val metrics (real labels only) and prompt-swap sensitivity on val."""
import json, sys
from io import BytesIO
import numpy as np, torch
from PIL import Image
sys.path.insert(0, r"E:\BaiqiL-GenAI-Bench\code")
from src.data.ingest import iter_rows
from src.data.validate import normalize_prompt
from src.models.clip_encoder import ClipEncoder
from src.models.regression_head import features
from src.training.train_regression import fit_heads
from src.evaluation.metrics import pairwise_accuracy
from transformers import BlipForImageTextRetrieval, BlipProcessor

root = r"E:\BaiqiL-GenAI-Bench\.cache"
records = [r for r in (json.loads(l) for l in open(root + r"\genai-bench-smoke\manifest_dev.jsonl", encoding="utf-8") if l.strip()) if r["split"] != "test"]
want = {(r["shard"], r["row"], r["image_model"]): r for r in records}
imgs = {}
for row in iter_rows(root + r"\genai-bench-smoke\data"):
    for m, b in row["images"].items():
        r = want.get((row["shard"], row["row"], m))
        if r: imgs[r["pair_id"]] = b
dev = "cuda"
proc = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
blip = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco").to(dev).eval()
clip = ClipEncoder(root + r"\models\clip-vit-base-patch32\3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268", dev)

def feats(pairs):  # pairs: list of (image_bytes, prompt) -> [N, 1026]
    out = []
    for i in range(0, len(pairs), 16):
        chunk = pairs[i:i + 16]
        images = [Image.open(BytesIO(b)).convert("RGB") for b, _ in chunk]
        prompts = [p for _, p in chunk]
        inp = proc(images=images, text=prompts, return_tensors="pt", padding=True, truncation=True).to(dev)
        with torch.inference_mode():
            itm = torch.softmax(blip(**inp, use_itm_head=True).itm_score.float(), -1)[:, 1].cpu().numpy()
        e = clip.encode(prompts, images)
        out.append(np.concatenate([features(e["text"], e["image"]), itm[:, None], e["cosine"][:, None]], 1))
    return np.concatenate(out)

rng = np.random.default_rng(1701)
split = {s: [r for r in records if r["split"] == s] for s in ("train", "val")}
def swapped(items):
    groups = sorted({r["prompt_group"] for r in items})
    res = []
    for r in items:
        g = rng.choice([x for x in groups if x != r["prompt_group"]])
        res.append(next(x["prompt"] for x in items if x["prompt_group"] == g))
    return res

import os
cache = "feats_cache.npz"
if os.path.exists(cache):
    z = np.load(cache); X = {"train": z["xtr"], "val": z["xva"]}; val_swap, train_neg = z["vs"], z["tn"]
else:
    X = {s: feats([(imgs[r["pair_id"]], r["prompt"]) for r in items]) for s, items in split.items()}
    val_swap = feats([(imgs[r["pair_id"]], p) for r, p in zip(split["val"], swapped(split["val"]))])
    train_neg = feats([(imgs[r["pair_id"]], p) for r, p in zip(split["train"], swapped(split["train"]))])
    np.savez(cache, xtr=X["train"], xva=X["val"], vs=val_swap, tn=train_neg)
Y = {s: np.array([r["target"] for r in items]) for s, items in split.items()}
groups_val = [normalize_prompt(r["prompt"]) for r in split["val"]]
NEG_RATIO = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25
k = int(len(train_neg) * NEG_RATIO)
sel = rng.choice(len(train_neg), k, replace=False)

configs = {
    "concat only, no negatives": (slice(0, 1024), False),
    "concat only + negatives": (slice(0, 1024), True),
    "concat+itm+cos, no negatives": (slice(0, 1026), False),
    "concat+itm+cos + negatives": (slice(0, 1026), True),
    "itm+cos only + negatives": (slice(1024, 1026), True),
}
for label, (cols, use_neg) in configs.items():
    xt, yt = X["train"][:, cols], Y["train"]
    if use_neg:
        xt = np.concatenate([xt, train_neg[sel][:, cols]]); yt = np.concatenate([yt, np.ones(k)])
    rows = []
    for seed in (42, 43, 44):
        models, rep = fit_heads(xt, yt, X["val"][:, cols], Y["val"], seed=seed)
        m = models["mlp"]
        pv, ps = m.predict(X["val"][:, cols]), m.predict(val_swap[:, cols])
        rows.append((rep["metrics"]["mlp"]["mae"], rep["metrics"]["mlp"]["spearman"],
                     pairwise_accuracy(Y["val"], pv, groups_val)["accuracy"], (pv > ps).mean(), (pv - ps).mean()))
    r = np.array(rows).mean(0)
    print(f"{label:32s} MAE {r[0]:.4f} spearman {r[1]:.4f} pairwise {r[2]:.4f} | swap-win {r[3]*100:.0f}% drop {r[4]:.3f}")

# Post-hoc fusion rules on top of the plain concat head (no negatives), seed 42
models, _ = fit_heads(X["train"][:, :1024], Y["train"], X["val"][:, :1024], Y["val"], seed=42)
m = models["mlp"]
pv, ps = m.predict(X["val"][:, :1024]), m.predict(val_swap[:, :1024])
iv, isw = X["val"][:, 1024], val_swap[:, 1024]
from src.evaluation.metrics import regression_metrics
rules = {
    "mlp (reference)": (pv, ps),
    "1+(mlp-1)*itm": (1 + (pv - 1) * iv, 1 + (ps - 1) * isw),
    "min(mlp, 1+4*itm)": (np.minimum(pv, 1 + 4 * iv), np.minimum(ps, 1 + 4 * isw)),
    "mlp if itm>0.5 else 1+2*itm": (np.where(iv > 0.5, pv, 1 + 2 * iv), np.where(isw > 0.5, ps, 1 + 2 * isw)),
    "0.5*mlp+0.5*(1+4*itm)": (0.5 * pv + 0.5 * (1 + 4 * iv), 0.5 * ps + 0.5 * (1 + 4 * isw)),
}
print("--- fusion rules (val, seed42) ---")
for label, (a, b) in rules.items():
    mm = regression_metrics(Y["val"], a)
    print(f"{label:32s} MAE {mm['mae']:.4f} spearman {mm['spearman']:.4f} pairwise {pairwise_accuracy(Y['val'], a, groups_val)['accuracy']:.4f} | swap-win {(a > b).mean()*100:.0f}% drop {(a - b).mean():.3f}")
print("val itm quantiles", np.quantile(iv, [0, .1, .25, .5, .75, .9, 1]).round(3), "swap itm quantiles", np.quantile(isw, [0, .5, .9, 1]).round(3))
