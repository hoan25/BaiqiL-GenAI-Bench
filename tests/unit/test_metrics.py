import math

import pytest

from src.evaluation.metrics import regression_metrics, spearman, pairwise_accuracy
from src.evaluation.baselines import evaluate_baselines


def test_regression_known_values():
    result = regression_metrics([1, 3, 5], [2, 3, 3])
    assert result["mae"] == 1
    assert result["rmse"] == pytest.approx(math.sqrt(5 / 3))


def test_spearman_ties_and_constants():
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1)
    assert spearman([1, 2, 2], [1, 2, 3]) == pytest.approx(math.sqrt(3) / 2)
    assert spearman([1, 2, 3], [2, 2, 2]) is None
    assert spearman([1], [1]) is None


@pytest.mark.parametrize("a,b", [([], []), ([1], [1, 2]), ([float('nan')], [2]), ([[1]], [[1]])])
def test_invalid_metrics(a, b):
    with pytest.raises(ValueError):
        regression_metrics(a, b)


def test_pairs_only_within_prompt_and_ties():
    result = pairwise_accuracy([1, 3, 3, 5], [0, 1, 1, -1], ["a", "a", "a", "b"])
    assert result["eligible_pairs"] == 2
    assert result["excluded_label_ties"] == 1
    assert result["accuracy"] == 1
    assert pairwise_accuracy([1, 3], [0, 0], ["a", "a"])["accuracy"] == 0
    assert pairwise_accuracy([1, 3], [0, 1], ["a", "b"])["accuracy"] is None


def test_baseline_train_only_no_test_labels():
    records = [dict(pair_id="a", prompt="train", split="train", target=1),
               dict(pair_id="b", prompt="train", split="train", target=3),
               dict(pair_id="c", prompt="val", split="val", target=5),
               dict(pair_id="d", prompt="test", split="test")]
    result = evaluate_baselines(records, {"c": 0.4})
    assert result["M0"]["train_mean"] == 2
    assert result["M0"]["mae"] == 3
    assert result["M0"]["spearman"] is None
    assert "mae" not in result["M1"]
    records[2]["prompt"] = " TRAIN "
    with pytest.raises(ValueError, match="leakage"):
        evaluate_baselines(records, {"c": 0.4})
