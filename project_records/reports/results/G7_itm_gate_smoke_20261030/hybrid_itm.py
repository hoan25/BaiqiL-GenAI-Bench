"""Dev check: does adding BLIP-ITM P(match) as a feature help the concat head on the shard?"""
import csv, json, sys
import numpy as np
sys.path.insert(0, r"E:\BaiqiL-GenAI-Bench\code")
from src.training.train_regression import fit_heads
from src.evaluation.metrics import pairwise_accuracy
from src.data.validate import normalize_prompt

root = r"E:\BaiqiL-GenAI-Bench\.cache"
records = [r for r in (json.loads(l) for l in open(root + r"\genai-bench-smoke\manifest_dev.jsonl", encoding="utf-8") if l.strip()) if r["split"] != "test"]
arr = np.load(root + r"\clip-smoke-extraction\embeddings.npz")
keep = [i for i, p in enumerate(arr["pair_ids"].tolist()) if p in {r["pair_id"] for r in records}]; arr = {k: arr[k][keep] for k in ("pair_ids", "text", "image", "cosine")}; assert arr["pair_ids"].tolist() == [r["pair_id"] for r in records]
itm = {}
for name in sys.argv[1:]:
    with open(root + f"\\{name}\\scores.csv", encoding="utf-8", newline="") as f:
        itm[name] = {r["pair_id"]: float(r["score"]) for r in csv.DictReader(f)}

def run(label, feats):
    out = {}
    for split in ("train", "val"):
        idx = [i for i, r in enumerate(records) if r["split"] == split]
        out[split] = (feats[idx], np.array([records[i]["target"] for i in idx]))
    groups = [normalize_prompt(r["prompt"]) for r in records if r["split"] == "val"]
    res = []
    for seed in (42, 43, 44):
        models, rep = fit_heads(*out["train"], *out["val"], seed=seed)
        m = rep["metrics"]["mlp"]
        pw = pairwise_accuracy(out["val"][1], models["mlp"].predict(out["val"][0]), groups)["accuracy"]
        res.append((m["mae"], m["spearman"], pw))
    res = np.array(res)
    print(f"{label:40s} val MAE {res[:,0].mean():.4f}  spearman {res[:,1].mean():.4f}  pairwise {res[:,2].mean():.4f}  (3 seeds)")

text, image, cos = arr["text"], arr["image"], arr["cosine"][:, None]
run("concat text+image (baseline)", np.concatenate([text, image], 1))
for name, s in itm.items():
    col = np.array([[s[r["pair_id"]]] for r in records])
    run(f"itm only [{name}]", np.concatenate([col, col**2], 1))
    run(f"concat + itm [{name}]", np.concatenate([text, image, col], 1))
    run(f"concat + itm + cosine [{name}]", np.concatenate([text, image, col, cos], 1))
    run(f"image only + itm [{name}]", np.concatenate([image, col], 1))
