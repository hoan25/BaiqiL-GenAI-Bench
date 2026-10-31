"""Development-only manifest from a partial local shard set (not the reviewed lock)."""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2])
sys.path.insert(0, r"E:\BaiqiL-GenAI-Bench\code")
from src.data.ingest import iter_rows
from src.data.manifest import build_manifest
from src.data.split import assign_splits

records, summary = build_manifest(iter_rows(root))
records = assign_splits(records, seed=42)
out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8")
from collections import Counter
print(summary["valid_pairs"], summary["rejected_pairs"], Counter(r["split"] for r in records))
