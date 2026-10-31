"""Time GatedPredictor on CPU with a synthetic 512x512 image: gated (English prompt)
versus gate skipped (Vietnamese prompt, BLIP not run). Not a quality benchmark."""

import argparse
import json
from pathlib import Path
import platform
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from PIL import Image

from src.inference.gated_predictor import GatedPredictor
from src.inference.language import gate_supported

DEFAULT_RECEIPT = Path(__file__).resolve().parents[1] / "configs/blip_itm_base_coco_receipt.json"


def timed(predictor, prompt, image, runs, warmup=2):
    for _ in range(warmup):
        predictor.predict(prompt, [image])
    samples, last = [], None
    for _ in range(runs):
        started = time.perf_counter()
        last = predictor.predict(prompt, [image])[0]
        samples.append((time.perf_counter() - started) * 1000)
    samples.sort()
    return {"runs": runs, "warmup": warmup, "mean_ms": round(statistics.mean(samples), 1),
            "median_ms": round(statistics.median(samples), 1), "min_ms": round(samples[0], 1),
            "max_ms": round(samples[-1], 1), "p90_ms": round(samples[int(0.9 * (runs - 1))], 1),
            "sample_output": last}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--training", required=True)
    parser.add_argument("--itm-model", default="Salesforce/blip-itm-base-coco")
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--threads", type=int, default=None, help="torch CPU threads; default leaves torch's choice")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--note", default="", help="free text about the machine state during the run")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.threads:
        torch.set_num_threads(args.threads)
    load_started = time.perf_counter()
    predictor = GatedPredictor(args.model, args.training, args.itm_model, args.receipt,
                               allow_development=True, device="cpu")
    load_ms = (time.perf_counter() - load_started) * 1000
    image = Image.new("RGB", (512, 512), (180, 40, 40))
    english, vietnamese = "a red square on a plain background", "một hình vuông màu đỏ trên nền trơn"
    assert gate_supported(english) and not gate_supported(vietnamese)
    gated = timed(predictor, english, image, args.runs)
    skipped = timed(predictor, vietnamese, image, args.runs)
    assert gated["sample_output"]["gated"] is True and skipped["sample_output"]["gated"] is False
    report = {"status": "PASS", "scope": "cpu_latency_synthetic_image_not_quality_benchmark",
              "predictor": "GatedPredictor", "rule": predictor.rule, "itm_model": args.itm_model,
              "itm_revision": predictor.itm_revision, "version": predictor.version,
              "image": "512x512 synthetic RGB", "device": "cpu", "torch_threads": torch.get_num_threads(),
              "torch": torch.__version__, "python": platform.python_version(), "machine": platform.machine(),
              "load_ms": round(load_ms, 1), "gated": gated, "gate_skipped": skipped,
              "gate_overhead_ms_median": round(gated["median_ms"] - skipped["median_ms"], 1),
              "note": args.note}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({k: report[k] for k in ("load_ms", "gate_overhead_ms_median")}),
          "gated median", gated["median_ms"], "ms; skipped median", skipped["median_ms"], "ms")


if __name__ == "__main__":
    main()
