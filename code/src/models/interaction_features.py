"""Experimental feature maps; not compatible with the deployed concat head."""
import numpy as np


def interaction_features(text, image, kind):
    text, image = np.asarray(text, dtype=np.float32), np.asarray(image, dtype=np.float32)
    if text.ndim != 2 or text.shape != image.shape or not text.shape[1]:
        raise ValueError('Expected paired [N,D] arrays')
    if not np.isfinite(text).all() or not np.isfinite(image).all():
        raise ValueError('Nonfinite features')
    cosine = np.sum(text * image, axis=1, keepdims=True)
    if kind == 'cosine':
        return cosine
    if kind == 'hybrid':
        return np.concatenate([text, image, cosine, text * image, np.abs(text-image)], axis=1)
    if kind == 'interaction':
        return np.concatenate([cosine, text * image, np.abs(text-image)], axis=1)
    raise ValueError('Unknown feature map')
