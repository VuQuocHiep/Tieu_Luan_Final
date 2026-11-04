import json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from .config import CONFIG, ARTIFACTS, DEFAULTS, SEED
from .datasets import load_sequences
from .models import (
    train_scratch, train_keras, train_pytorch,
    load_scratch, load_keras, load_pytorch,
    keras_predict_proba, pytorch_predict_proba,
)


def artifact_paths(key, method):
    base = ARTIFACTS / key
    base.mkdir(parents=True, exist_ok=True)
    ext = {'scratch': '.npz', 'keras': '.keras', 'pytorch': '.pt'}[method]
    return {
        'model': base / f'{method}{ext}',
        'vocab': base / 'vocab.json',
        'meta': base / 'meta.json',
        'metrics': base / f'{method}_metrics.json',
        'history': base / f'{method}_history.json',
    }


def _opts(overrides=None):
    o = dict(DEFAULTS)
    o['seed'] = SEED
    for k, v in (overrides or {}).items():
        if v is not None and k in o:
            o[k] = v
    return o


def _metrics(y_true, proba, classes):
    pred = np.asarray(proba).argmax(axis=1)
    p, r, f, _ = precision_recall_fscore_support(y_true, pred, average='macro', zero_division=0)
    return {
        'accuracy': float(accuracy_score(y_true, pred)),
        'macro_precision': float(p),
        'macro_recall': float(r),
        'macro_f1': float(f),
        'confusion_matrix': confusion_matrix(y_true, pred, labels=list(range(len(classes)))).astype(int).tolist(),
        'n_test': int(len(y_true)),
        'classes': list(classes),
    }


def run_one(key, method, overrides=None):
    if key not in CONFIG:
        raise ValueError(f'Dataset không hỗ trợ: {key}')
    if method not in ('scratch', 'keras', 'pytorch'):
        raise ValueError(f'Method không hỗ trợ: {method}')
    opts = _opts(overrides)
    bundle = load_sequences(key, max_train=opts['max_train'], max_val=opts['max_val'], max_test=opts['max_test'],
                            max_len=opts['max_len'], vocab_size=opts['vocab_size'], seed=opts['seed'])
    paths = artifact_paths(key, method)
    with paths['vocab'].open('w', encoding='utf-8') as f:
        json.dump(bundle.vocab, f, ensure_ascii=False)
    meta = {
        'key': key, 'title': CONFIG[key]['title'], 'classes': bundle.classes,
        'original_counts': bundle.original_counts, 'distribution': bundle.distribution,
        'options': opts, 'vocab_size_actual': len(bundle.vocab), 'samples': bundle.samples,
    }
    with paths['meta'].open('w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f'\n=== {key.upper()} / {method.upper()} ===')
    print(f'train={len(bundle.y_train)} val={len(bundle.y_val)} test={len(bundle.y_test)} vocab={len(bundle.vocab)} max_len={opts["max_len"]}')
    if method == 'scratch':
        model, history = train_scratch(bundle, opts, paths['model'])
        proba = model.predict_proba(bundle.x_test)
    elif method == 'keras':
        model, history = train_keras(bundle, opts, paths['model'])
        proba = keras_predict_proba(model, bundle.x_test)
    else:
        model, history = train_pytorch(bundle, opts, paths['model'])
        proba = pytorch_predict_proba(model, bundle.x_test)

    m = _metrics(bundle.y_test, proba, bundle.classes)
    m.update({'key': key, 'method': method, 'options': opts})
    with paths['metrics'].open('w', encoding='utf-8') as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    with paths['history'].open('w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f'{key:10s} | {method:8s} | Accuracy {m["accuracy"]:.4f} | Macro-F1 {m["macro_f1"]:.4f}')
    return m


def run_all(dataset='all', method='all', overrides=None):
    keys = list(CONFIG) if dataset == 'all' else [dataset]
    methods = ['scratch', 'keras', 'pytorch'] if method == 'all' else [method]
    results = []
    for key in keys:
        for meth in methods:
            results.append(run_one(key, meth, overrides))
    return results


def load_for_inference(key, method):
    paths = artifact_paths(key, method)
    missing = [str(paths[n]) for n in ('model', 'vocab', 'meta') if not paths[n].exists()]
    if missing:
        raise FileNotFoundError('Chưa train mô hình. Thiếu: ' + ', '.join(missing))
    vocab = json.loads(paths['vocab'].read_text(encoding='utf-8'))
    meta = json.loads(paths['meta'].read_text(encoding='utf-8'))
    if method == 'scratch': model = load_scratch(paths['model'])
    elif method == 'keras': model = load_keras(paths['model'])
    else: model = load_pytorch(paths['model'])
    return model, vocab, meta


def predict_text(key, method, text):
    from .text_utils import encode_texts
    model, vocab, meta = load_for_inference(key, method)
    x = encode_texts([text], vocab, max_len=meta['options']['max_len'])
    if method == 'scratch': proba = model.predict_proba(x)[0]
    elif method == 'keras': proba = keras_predict_proba(model, x)[0]
    else: proba = pytorch_predict_proba(model, x)[0]
    idx = int(np.argmax(proba))
    return {'class_index': idx, 'label': meta['classes'][idx], 'probabilities': proba.tolist(), 'classes': meta['classes']}
