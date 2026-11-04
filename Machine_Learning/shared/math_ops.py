"""Stable softmax for logits (not a library classification model)."""
import numpy as np

def softmax(logits):
    z = logits - np.max(logits, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)
