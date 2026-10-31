"""Prompt-swap sensitivity on shard val pairs: original prompt vs prompt from another prompt group.
Compares BLIP-ITM (zero-shot), CLIP cosine, and the shard-trained concat MLP."""
import json, sys
from io import BytesIO
import numpy as np, torch
from PIL import Image
sys.path.insert(0, r"E:\BaiqiL-GenAI-Bench\code")
from src.data.ingest import iter_rows
from src.models.clip_encoder import ClipEncoder
from src.models.regression_head import load_head, features
from transformers import BlipForImageTextRetrieval, BlipProcessor

root = r"E:\BaiqiL-GenAI-Bench\.cache"
records = [json.loads(l) for l in open(root + r"\genai-bench-smoke\manifest_dev.jsonl", encoding="utf-8") if l.strip()]
val = [r for r in records if r["split"] == "val"]
want = {(r["shard"], r["row"], r["image_model"]): r for r in val}
imgs = {}
for row in iter_rows(root + r"\genai-bench-smoke\data"):
    for m, b in row["images"].items():
        r = want.get((row["shard"], row["row"], m))
        if r: imgs[r["pair_id"]] = b
groups = sorted({r["prompt_group"] for r in val})
rng = np.random.default_rng(0)
# swapped prompt = prompt from a different group
other = {}
for r in val:
    choices = [g for g in groups if g != r["prompt_group"]]
    other[r["pair_id"]] = next(x["prompt"] for x in val if x["prompt_group"] == rng.choice(choices))

dev = "cuda"
proc = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
blip = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco").to(dev).eval()
clip = ClipEncoder(root + r"\models\clip-vit-base-patch32\3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268", dev)
rep = json.load(open(root + r"\clip-smoke-training\result.json", encoding="utf-8"))
head = load_head(root + r"\clip-smoke-training\\" + rep["checkpoints"]["mlp"]["file"])

def itm(images, prompts):
    inp = proc(images=images, text=prompts, return_tensors="pt", padding=True, truncation=True).to(dev)
    with torch.inference_mode():
        return torch.softmax(blip(**inp, use_itm_head=True).itm_score.float(), -1)[:, 1].cpu().numpy()

def mlp_and_cos(images, prompts):
    e = clip.encode(prompts, images)
    return head.predict(features(e["text"], e["image"])), e["cosine"]

wins = {"blip_itm": 0, "clip_cosine": 0, "shard_mlp": 0}
drops = {k: [] for k in wins}
B = 8
for i in range(0, len(val), B):
    chunk = val[i:i + B]
    images = [Image.open(BytesIO(imgs[r["pair_id"]])).convert("RGB") for r in chunk]
    orig = [r["prompt"] for r in chunk]; swap = [other[r["pair_id"]] for r in chunk]
    a, b = itm(images, orig), itm(images, swap)
    ma, ca = mlp_and_cos(images, orig); mb, cb = mlp_and_cos(images, swap)
    for name, (x, y) in {"blip_itm": (a, b), "clip_cosine": (ca, cb), "shard_mlp": (ma, mb)}.items():
        wins[name] += int((x > y).sum()); drops[name] += (x - y).tolist()
n = len(val)
for k in wins:
    print(f"{k:12s} original beats swapped prompt: {wins[k]}/{n}   mean drop {np.mean(drops[k]):.4f}")
