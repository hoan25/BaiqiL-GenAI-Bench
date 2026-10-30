import numpy as np
import pytest
from src.training.synthetic_negatives import negative_indices


def test_negative_pairs_are_cross_group_and_reproducible():
    text = np.repeat(np.eye(4, dtype=np.float32), 3, axis=0)
    groups = np.repeat(np.arange(4), 3)
    a, b = negative_indices(text, groups)
    c, d = negative_indices(text, groups)
    assert len(a) == len(text)
    assert np.all(groups[a] != groups[b])
    assert np.all(np.sum(text[a] * text[b], axis=1) < 0.5)
    np.testing.assert_array_equal(a, c)
    np.testing.assert_array_equal(b, d)
    with pytest.raises(ValueError):
        negative_indices(text, ['same'] * len(text))


def test_similar_prompts_are_not_forced_into_negative_pairs():
    a, b = negative_indices(np.ones((3, 2)), ['a', 'b', 'c'])
    assert len(a) == len(b) == 0
