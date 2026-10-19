from io import BytesIO

from PIL import Image
import pytest

from src.data.manifest import build_manifest, inference_input
from src.data.split import assign_splits
from src.data.validate import MODELS, normalize_prompt, validate_ratings


def row(index=0, prompt="a red square"):
    buffer = BytesIO()
    Image.new("L", (4, 5), 120).save(buffer, format="PNG")
    return {"shard": "fixture", "row": index, "prompt": prompt,
            "images": dict.fromkeys(MODELS, buffer.getvalue()),
            "ratings": {model: [2, 4, 3] for model in MODELS}}


@pytest.mark.parametrize("ratings", [[1, 2], [0, 2, 3], [6, 2, 3],
                                     [True, 2, 3], [1.5, 2, 3], [float("nan"), 2, 3]])
def test_bad_ratings(ratings):
    with pytest.raises(ValueError):
        validate_ratings(ratings)


def test_targets():
    assert validate_ratings([5, 5, 5]) == ([5, 5, 5], 5, 0)
    assert validate_ratings([2, 4, 3])[1:] == pytest.approx((3, (2 / 3) ** 0.5))


def test_mapping_and_features():
    source = row()
    source["ratings"][MODELS[1]] = [5, 5, 5]
    records, report = build_manifest([source])
    assert report["valid_pairs"] == 6
    assert len({r["pair_id"] for r in records}) == 6
    by_model = {r["image_model"]: r for r in records}
    assert by_model[MODELS[0]]["target"] == 3
    assert by_model[MODELS[1]]["target"] == 5
    assert all((r["width"], r["height"], r["mode"]) == (4, 5, "RGB") for r in records)
    assert inference_input(records[0], b"image") == {"prompt": "a red square", "image": b"image"}


def test_corrupt_and_missing_are_accounted_for():
    source = row()
    source["images"][MODELS[0]] = b"not a picture"
    del source["ratings"][MODELS[1]]
    records, report = build_manifest([source])
    assert len(records) == 4
    assert report["expected_pairs"] == report["valid_pairs"] + report["rejected_pairs"] == 6
    assert {r["image_model"] for r in report["rejections"]} == set(MODELS[:2])


def test_duplicate_source_fails():
    with pytest.raises(ValueError, match="duplicate"):
        build_manifest([row(), row()])
    other = row()
    other["shard"] = "another shard"
    assert len(build_manifest([row(), other])[0]) == 12


def test_prompt_normalization_and_split():
    assert normalize_prompt("  Ａ RED\n square ") == "a red square"
    rows = [row(i, f"prompt {i}") for i in range(20)] + [row(20, " PROMPT  0 ")]
    records, _ = build_manifest(rows)
    result = assign_splits(records)
    reverse = assign_splits(list(reversed(records)))
    mapping = lambda values: {r["pair_id"]: r["split"] for r in values}
    assert mapping(result) == mapping(reverse)
    assert mapping(result) != mapping(assign_splits(records, seed=99))
    groups = {name: {r["prompt_group"] for r in result if r["split"] == name}
              for name in ("train", "val", "test")}
    assert all(groups.values())
    assert not groups["train"] & groups["val"]
    assert not groups["train"] & groups["test"]
    assert not groups["val"] & groups["test"]


def test_invalid_split_inputs():
    with pytest.raises(ValueError, match="three distinct"):
        assign_splits([])
    with pytest.raises(ValueError, match="ratios"):
        assign_splits([], ratios=(0.8, 0.2, 0.2))


def test_empty_prompt_excluded():
    records, report = build_manifest([row(prompt="  ")])
    assert records == []
    assert report["rejected_pairs"] == 6
