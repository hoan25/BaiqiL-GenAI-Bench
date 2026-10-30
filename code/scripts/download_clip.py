"""Download the authorized CLIP snapshot and record its revision and checksums."""

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268")
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--output", type=Path, default=Path("../project_records/reports/results/clip_download.json"))
    args = parser.parse_args()
    repo_id = "openai/clip-vit-base-patch32"
    files = [
        "config.json", "preprocessor_config.json", "tokenizer_config.json",
        "special_tokens_map.json", "vocab.json", "merges.txt", "tokenizer.json",
        "pytorch_model.bin",
    ]
    destination = args.destination or Path("../.cache/models/clip-vit-base-patch32") / args.revision
    snapshot_download(
        repo_id, revision=args.revision, local_dir=destination,
        allow_patterns=files, token=False, max_workers=2,
    )
    records = []
    for path in sorted(destination.iterdir()):
        if path.is_file():
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            records.append({"file": path.name, "bytes": path.stat().st_size, "sha256": digest})
    evidence = {
        "repo_id": repo_id, "revision": args.revision,
        "local_path": destination.as_posix(), "files": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
