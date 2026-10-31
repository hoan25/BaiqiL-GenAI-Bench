import numpy as np
import pytest

from src.evaluation.itm_gate import evaluate_rules, fuse


def test_gate_multiply_keeps_matches_and_pulls_mismatches_to_one():
    score, match = [5, 3, 4], [1, 1, 0]
    assert fuse("gate_multiply", score, match).tolist() == pytest.approx([5, 3, 1])
    assert fuse("gate_min", score, match).tolist() == pytest.approx([5, 3, 1])
    assert fuse("match_only", score, match).tolist() == pytest.approx([5, 5, 1])
    assert fuse("mlp", score, match).tolist() == [5, 3, 4]
    assert fuse("gate_multiply", [5], [0.5]).tolist() == pytest.approx([3])
    assert fuse("gate_sqrt", [5, 5], [0.25, 0]).tolist() == pytest.approx([3, 1])
    assert fuse("gate_soft25", [5, 5], [1, 0]).tolist() == pytest.approx([5, 2])


@pytest.mark.parametrize("score,match", [([6], [1]), ([3], [1.5]), ([3, 4], [1]), ([], []), ([float("nan")], [1])])
def test_fuse_rejects_invalid(score, match):
    with pytest.raises(ValueError):
        fuse("gate_multiply", score, match)
    with pytest.raises(ValueError):
        fuse("nope", [3], [1])


def test_evaluate_rules_reports_swap_sensitivity():
    target = [5, 1, 4, 2]
    score = [4.5, 4.4, 4.0, 3.9]          # regression ignores the prompt
    match = [1, 0, 1, 0]
    groups = ["a", "a", "b", "b"]
    swap_score = [4.5, 4.4, 4.0, 3.9]
    swap_match = [0, 0, 0, 0]
    report = evaluate_rules(target, score, match, groups, swap_score, swap_match)
    assert report["mlp"]["swap"]["original_wins"] == 0
    assert report["gate_multiply"]["swap"]["original_wins"] == 2
    assert report["gate_multiply"]["pairwise"]["accuracy"] == 1
    assert report["gate_multiply"]["mae"] < report["mlp"]["mae"]
    assert "swap" not in evaluate_rules(target, score, match, groups)["mlp"]
