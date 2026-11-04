"""Load trained checkpoints for the Streamlit pages and test scripts."""
import os
os.environ.setdefault('KERAS_BACKEND', 'torch')  # must precede importing keras
from pathlib import Path
import joblib
import numpy as np
from .math_ops import softmax

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / 'artifacts'
BACKENDS = {'scratch': 'NumPy From Scratch', 'keras': 'Keras 3', 'pytorch': 'PyTorch'}

def load_metadata(name):
    return joblib.load(ARTIFACTS/f'{name}_metadata.joblib')

def load_model(name, backend, n_features, n_classes):
    if backend == 'scratch':
        f = np.load(ARTIFACTS/f'{name}_scratch.npz')
        return f['W'], f['b']
    if backend == 'keras':
        import keras
        return keras.models.load_model(ARTIFACTS/f'{name}_keras.keras', compile=False)
    if backend == 'pytorch':
        import torch
        layer = torch.nn.Linear(n_features, n_classes)
        layer.load_state_dict(torch.load(ARTIFACTS/f'{name}_pytorch.pt',
                                         map_location='cpu', weights_only=True))
        layer.eval()
        return layer
    raise ValueError(backend)

def predict_proba(model, backend, X):
    if backend == 'scratch':
        W, bias = model
        return softmax(X @ W + bias)
    if backend == 'keras':
        return softmax(model.predict(X.astype('float32'), verbose=0))
    import torch
    with torch.no_grad():
        return torch.softmax(model(torch.tensor(X, dtype=torch.float32)), dim=1).numpy()
