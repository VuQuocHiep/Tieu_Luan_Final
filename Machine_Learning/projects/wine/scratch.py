"""wine: SOFTMAX LOGISTIC REGRESSION - FROM SCRATCH with NumPy only.
No sklearn estimator, Keras or PyTorch models used in this implementation.
"""
import numpy as np
from shared.datasets import EPOCHS, LEARNING_RATE

def train(X, y, n_classes):
    n, d = X.shape
    W = np.zeros((d, n_classes), dtype=np.float64)
    bias = np.zeros(n_classes, dtype=np.float64)
    target = np.eye(n_classes)[y]
    for epoch in range(EPOCHS):
        logits = X @ W + bias
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(logits)
        probability = exp / np.sum(exp, axis=1, keepdims=True)
        error = (probability - target) / n
        dW = X.T @ error
        db = np.sum(error, axis=0)
        W -= LEARNING_RATE * dW
        bias -= LEARNING_RATE * db
    return W, bias
