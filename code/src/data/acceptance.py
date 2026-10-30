"""Bind an explicitly reviewed QA archive to an immutable experiment manifest."""

import hashlib
import json
from pathlib import Path
import zipfile

SPEC = Path(__file__).resolve().parents[2] / "configs/genai_bench_review.json"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def spec_sha256(path):
    # Git may change LF/CRLF across Windows and Linux; hash the specification's
    # content canonically, while retaining exact-byte hashes for dataset artifacts.
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    payload = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_bundle(archive, spec_path=SPEC):
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    archive = Path(archive)
    names = ("qa/manifest.jsonl", "qa/qa.json", "result.json", "dataset.json")
    if archive.is_dir():
        contents = {name: (archive / name).read_bytes() for name in names}
    else:
        if sha256(archive) != spec["archive_sha256"]:
            raise ValueError("QA archive differs from reviewed archive")
        with zipfile.ZipFile(archive) as z:
            # Read only known members; never extract archive paths.
            contents = {name: z.read(name) for name in names}
    if any(hashlib.sha256(payload).hexdigest() != spec["member_sha256"][name]
           for name, payload in contents.items()):
        raise ValueError("QA member checksum differs from reviewed archive")
    payload = contents["qa/manifest.jsonl"]
    qa = json.loads(contents["qa/qa.json"])
    result = json.loads(contents["result.json"])
    dataset = json.loads(contents["dataset.json"])
    if (hashlib.sha256(payload).hexdigest() != spec["manifest_sha256"]
            or qa["manifest_sha256"] != spec["manifest_sha256"]
            or dataset["revision"] != spec["dataset_revision"]
            or qa["sources"] != spec["sources"]
            or qa["split_pairs"] != spec["split_pairs"]
            or result["status"] != "REVIEW_REQUIRED"
            or qa["valid_pairs"] != spec["valid_pairs"]):
        raise ValueError("QA provenance differs from reviewed specification")
    return spec, payload


def accept_bundle(archive, output, reviewer, spec_path=SPEC):
    if not reviewer or not reviewer.strip():
        raise ValueError("A named reviewer must explicitly accept the documented data decisions")
    spec, payload = read_bundle(archive, spec_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "manifest.jsonl").write_bytes(payload)
    lock = {
        "status": "ACCEPTED_FOR_EXPERIMENT",
        "scope": "data_decisions_only_not_milestone_or_release_acceptance",
        "reviewer": reviewer.strip(),
        "review_spec_sha256": spec_sha256(spec_path),
        "manifest_sha256": spec["manifest_sha256"],
        "dataset_revision": spec["dataset_revision"],
        "sources": spec["sources"], "decisions": spec["decisions"],
    }
    (output / "lock.json").write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return output / "lock.json"


def validate_lock(lock_path, manifest, data_root=None, spec_path=SPEC):
    lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    if (lock.get("status") != "ACCEPTED_FOR_EXPERIMENT"
            or not isinstance(lock.get("reviewer"), str) or not lock["reviewer"].strip()
            or lock.get("review_spec_sha256") != spec_sha256(spec_path)
            or lock.get("manifest_sha256") != spec["manifest_sha256"]
            or sha256(manifest) != spec["manifest_sha256"]
            or lock.get("dataset_revision") != spec["dataset_revision"]
            or lock.get("sources") != spec["sources"]
            or lock.get("decisions") != spec["decisions"]):
        raise ValueError("missing approval or stale/tampered data lock")
    if data_root is not None:
        root = Path(data_root).resolve()
        for item in spec["sources"]:
            path = (root / item["shard"]).resolve()
            if (not path.is_relative_to(root) or not path.is_file()
                    or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]):
                raise ValueError(f"source checksum mismatch: {item['shard']}")
    return lock
