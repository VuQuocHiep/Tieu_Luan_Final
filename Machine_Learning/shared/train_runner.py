"""Run 3 comparable implementations for one dataset, save and evaluate artifacts."""
import importlib
import json
import time
import hashlib
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, log_loss
from .datasets import dataset, load_split, SOURCES, EPOCHS, LEARNING_RATE, SEED
from .inference import ARTIFACTS, predict_proba

BACKENDS = ['scratch', 'keras', 'pytorch']

def evaluate(y, p):
    predicted = p.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, predicted, average='macro', zero_division=0)
    return {
        'accuracy': round(float(accuracy_score(y, predicted)), 6),
        'precision_macro': round(float(precision), 6),
        'recall_macro': round(float(recall), 6),
        'f1_macro': round(float(f1), 6),
        'log_loss': round(float(log_loss(y, p, labels=np.arange(p.shape[1]))), 6),
        'confusion_matrix': confusion_matrix(y, predicted).tolist(),
    }

def run_dataset(name):
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    raw, scaler, (train, val, test), (train_ids,val_ids,test_ids) = load_split(name)
    Xtr, ytr = train
    Xval, yval = val
    Xtest, ytest = test
    n_classes = len(raw.target_names)
    joblib.dump({'scaler': scaler, 'features': list(map(str, raw.feature_names)),
                 'classes': list(map(str, raw.target_names)), 'dataset': name,
                 'kaggle_url': raw.source_url, 'csv_sha256': hashlib.sha256(Path(raw.source_csv).read_bytes()).hexdigest()},
                ARTIFACTS/f'{name}_metadata.joblib')
    preview = pd.DataFrame(raw.data[:5], columns=raw.feature_names)
    preview['class'] = [str(raw.target_names[n]) for n in raw.target[:5]]
    preview.to_csv(ARTIFACTS/f'{name}_sample.csv',index=False)
    results = []
    for backend in BACKENDS:
        mod = importlib.import_module(f'projects.{name}.{backend}')
        start = time.perf_counter()
        model = mod.train(Xtr, ytr, n_classes)
        elapsed = time.perf_counter()-start
        if backend == 'scratch':
            W, b = model
            np.savez(ARTIFACTS/f'{name}_scratch.npz',W=W,b=b)
        elif backend == 'keras':
            model.save(ARTIFACTS/f'{name}_keras.keras')
        else:
            torch.save(model.state_dict(),ARTIFACTS/f'{name}_pytorch.pt')
        val_score = evaluate(yval, predict_proba(model, backend, Xval))
        test_score = evaluate(ytest, predict_proba(model, backend, Xtest))
        results.append({'dataset':name,'backend':backend,
                        'kaggle_url':raw.source_url,
                        'train_size':len(ytr),'val_size':len(yval),'test_size':len(ytest),
                        'train_seconds':round(elapsed,4),
                        'validation_accuracy':val_score['accuracy'],
                        'validation_f1_macro':val_score['f1_macro'],
                        **test_score})
        print(f'{name:14} | {backend:8} | Accuracy {test_score["accuracy"]:.4f} | F1 {test_score["f1_macro"]:.4f}')
    (ARTIFACTS/f'{name}_metrics.json').write_text(json.dumps(results, indent=2),encoding='utf-8')
    (ARTIFACTS/f'{name}_test_indices.json').write_text(json.dumps(list(map(int,test_ids))),encoding='utf-8')
    return results

def run_all():
    results = []
    for key in SOURCES:
        results.extend(run_dataset(key))
    pd.DataFrame(results).to_csv(ARTIFACTS/'metrics.csv',index=False)
    print('Saved metrics to',ARTIFACTS/'metrics.csv')
    return results
