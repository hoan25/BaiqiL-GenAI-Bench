"""Weak negative pairs, never human labels or an independent benchmark."""
import numpy as np


def negative_indices(text, groups, *, seed=1701, maximum_text_similarity=0.5):
    text = np.asarray(text, dtype=np.float32)
    groups = np.asarray(groups)
    if text.ndim != 2 or groups.shape != (len(text),) or not np.isfinite(text).all():
        raise ValueError('Invalid text embeddings/groups')
    if len(set(groups.tolist())) < 2:
        raise ValueError('Need distinct prompt groups')
    rng = np.random.default_rng(seed)
    source, replacement = [], []
    for i in rng.permutation(len(text)):
        candidates = rng.integers(0, len(text), size=128)
        valid = candidates[(groups[candidates] != groups[i]) &
                           (text[candidates] @ text[i] < maximum_text_similarity)]
        if len(valid):
            source.append(i)
            replacement.append(int(valid[0]))
    return np.asarray(source, dtype=int), np.asarray(replacement, dtype=int)
