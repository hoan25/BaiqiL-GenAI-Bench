import hashlib
import json
import zipfile

import pytest

from src.data.acceptance import accept_bundle, read_bundle, sha256, validate_lock


@pytest.fixture
def bundle(tmp_path):
    data = tmp_path / "raw"
    data.mkdir()
    (data / "fixture.parquet").write_bytes(b"synthetic shard for checksum tests")
    payload = b'{"pair_id":"fixture"}\n'
    spec = {"manifest_sha256": hashlib.sha256(payload).hexdigest(),
            "dataset_revision": "fixture-revision",
            "sources": [{"shard": "fixture.parquet", "bytes": (data / "fixture.parquet").stat().st_size,
                         "sha256": sha256(data / "fixture.parquet")}],
            "split_pairs": {"train": 1}, "valid_pairs": 1,
            "decisions": ["fixture approval, not full data"]}
    archive = tmp_path / "qa.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("qa/manifest.jsonl", payload)
        z.writestr("qa/qa.json", json.dumps(spec))
        z.writestr("result.json", json.dumps({"status": "REVIEW_REQUIRED"}))
        z.writestr("dataset.json", json.dumps({"revision": "fixture-revision"}))
        z.writestr("../must-not-extract.txt", "unsafe archive entry")
    spec["archive_sha256"] = sha256(archive)
    with zipfile.ZipFile(archive) as z:
        spec["member_sha256"] = {name: hashlib.sha256(z.read(name)).hexdigest()
                                for name in ("qa/manifest.jsonl", "qa/qa.json", "result.json", "dataset.json")}
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    return archive, path, data


def test_acceptance_requires_reviewer_and_preserves_exact_manifest(bundle, tmp_path):
    archive, spec, data = bundle
    with pytest.raises(ValueError, match="reviewer"):
        accept_bundle(archive, tmp_path / "no", " ", spec)
    path = accept_bundle(archive, tmp_path / "accepted", "Fixture reviewer", spec)
    validate_lock(path, path.parent / "manifest.jsonl", data, spec)
    # The same reviewed specification must work after Windows/Linux Git checkout.
    spec.write_bytes(json.dumps(json.loads(spec.read_text()), indent=2).replace("\n", "\r\n").encode())
    validate_lock(path, path.parent / "manifest.jsonl", data, spec)
    assert not (tmp_path / "must-not-extract.txt").exists()
    with pytest.raises(FileExistsError):
        accept_bundle(archive, path.parent, "Fixture reviewer", spec)


def test_unpacked_input_uses_member_hashes(bundle, tmp_path):
    archive, spec, _ = bundle
    directory = tmp_path / "unpacked"
    with zipfile.ZipFile(archive) as z:
        for name in ("qa/manifest.jsonl", "qa/qa.json", "result.json", "dataset.json"):
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(z.read(name))
    assert read_bundle(directory, spec)[1] == read_bundle(archive, spec)[1]
    (directory / "dataset.json").write_text("{}")
    with pytest.raises(ValueError, match="member checksum"):
        read_bundle(directory, spec)


@pytest.mark.parametrize("change", ["archive", "manifest", "source", "reviewer", "spec", "lock_sources"])
def test_tampering_is_rejected(bundle, tmp_path, change):
    archive, spec, data = bundle
    lock = accept_bundle(archive, tmp_path / "accepted", "Fixture reviewer", spec)
    manifest = lock.parent / "manifest.jsonl"
    if change == "archive":
        with archive.open("ab") as stream:
            stream.write(b"tampered")
        with pytest.raises(ValueError, match="archive"):
            read_bundle(archive, spec)
        return
    if change == "spec":
        value = json.loads(spec.read_text())
        value["decisions"] = ["changed review"]
        spec.write_text(json.dumps(value))
    elif change in {"manifest", "source"}:
        path = {"manifest": manifest, "source": data / "fixture.parquet"}[change]
        path.write_bytes(path.read_bytes() + b" ")
    else:
        value = json.loads(lock.read_text())
        value["reviewer" if change == "reviewer" else "sources"] = "" if change == "reviewer" else []
        lock.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        validate_lock(lock, manifest, data, spec)
