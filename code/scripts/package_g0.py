"""Package an allowlist of G0 source files and the authorized CLIP snapshot."""

import hashlib
import json
from pathlib import Path
import zipfile


def main() -> None:
    manifest_path = Path("../project_records/reports/results/clip_download.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model = Path(manifest["local_path"])
    if not model.is_absolute():
        model = Path("..") / model
    files = [(p, p.as_posix()) for pattern in ("scripts/*.py", "tests/unit/*.py") for p in Path(".").glob(pattern)]
    for name in ("pyproject.toml", "requirements-g0.in", "requirements-g0-win-cpu.lock.txt", "notebooks/00_environment.ipynb"):
        p = Path(name)
        files.append((p, p.as_posix()))
    files.extend([(Path("../project_records/docs/G0_KAGGLE.md"), "docs/G0_KAGGLE.md"),
                  (manifest_path, "reports/results/clip_download.json")])
    for item in manifest["files"]:
        path = model / item["file"]
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != item["sha256"]:
                raise ValueError(f"Checkpoint checksum mismatch: {path.name}")
        files.append((path, "models/clip-vit-base-patch32/" + path.name))
    target = Path("../project_records/artifacts/G0_kaggle_bundle.zip")
    target.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_STORED) as archive:
        for path, name in files:
            archive.write(path, name)
    with zipfile.ZipFile(target) as archive:
        bad_file = archive.testzip()
        if bad_file:
            raise ValueError(f"ZIP integrity failure: {bad_file}")
    with target.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    evidence = {"path": target.as_posix(), "bytes": target.stat().st_size, "sha256": checksum, "files": [name for _, name in files], "zip_integrity": "PASS"}
    Path("../project_records/reports/results/G0_bundle.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
