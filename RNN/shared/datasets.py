from dataclasses import dataclass
from pathlib import Path
import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import CONFIG, RAW, SEED
from .text_utils import build_vocab, encode_texts


@dataclass
class RawDataset:
    key: str
    train_texts: list
    train_labels: np.ndarray
    val_texts: list
    val_labels: np.ndarray
    test_texts: list
    test_labels: np.ndarray
    classes: list
    original_counts: dict
    distribution: dict
    samples: list


@dataclass
class SequenceBundle:
    key: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    classes: list
    vocab: dict
    original_counts: dict
    distribution: dict
    samples: list


def _find_file(key, filename):
    base = RAW / key
    if not base.exists():
        raise FileNotFoundError(f'Chưa có thư mục dữ liệu {base}. Chạy: python download_kaggle.py')
    direct = base / filename
    if direct.exists():
        return direct
    matches = [p for p in base.rglob('*') if p.is_file() and p.name.lower() == filename.lower()]
    if matches:
        return matches[0]
    raise FileNotFoundError(f'{key}: không tìm thấy {filename} trong {base}. Chạy lại python download_kaggle.py')


def _read_csv_flexible(path):
    path = Path(path)
    last = None
    for enc in ('utf-8-sig', 'utf-8', 'latin1'):
        try:
            df = pd.read_csv(path, encoding=enc)
            df.columns = [str(c).strip() for c in df.columns]
            return df
        except Exception as e:
            last = e
    raise ValueError(f'Không đọc được CSV {path.name}: {last}')


def _normalize_binary(values, mapping, key):
    out = []
    for v in values:
        s = str(v).strip().lower()
        if s not in mapping:
            raise ValueError(f'{key}: nhãn không hợp lệ: {v!r}. Nhãn hỗ trợ: {sorted(mapping)}')
        out.append(mapping[s])
    return np.asarray(out, dtype=np.int64)


def _split_text(texts, labels, seed=SEED):
    idx = np.arange(len(labels))
    tr, temp = train_test_split(idx, test_size=0.30, stratify=labels, random_state=seed)
    va, te = train_test_split(temp, test_size=0.50, stratify=labels[temp], random_state=seed + 1)
    arr = np.asarray(texts, dtype=object)
    return (arr[tr].tolist(), labels[tr], arr[va].tolist(), labels[va], arr[te].tolist(), labels[te])


def _load_sms():
    path = _find_file('sms', 'spam.csv')
    df = _read_csv_flexible(path)
    # Kaggle common schema: v1=label, v2=text, plus unnamed empty columns.
    cols = {str(c).strip().lower(): c for c in df.columns}
    label_col = cols.get('v1') or cols.get('label') or cols.get('category') or cols.get('class')
    text_col = cols.get('v2') or cols.get('message') or cols.get('text') or cols.get('sms')
    if label_col is None or text_col is None:
        usable = [c for c in df.columns if not str(c).lower().startswith('unnamed')]
        if len(usable) >= 2:
            # Detect column that looks like ham/spam.
            for c in usable:
                vals = set(df[c].dropna().astype(str).str.strip().str.lower().unique())
                if vals and vals.issubset({'ham', 'spam'}):
                    label_col = c
                    break
            remaining = [c for c in usable if c != label_col]
            if remaining:
                text_col = remaining[0]
    if label_col is None or text_col is None:
        raise ValueError(f'sms: không nhận diện được schema. Cột hiện có: {list(df.columns)}')
    df = df[[label_col, text_col]].dropna().copy()
    texts = df[text_col].astype(str).tolist()
    labels = _normalize_binary(df[label_col], {'ham': 0, 'spam': 1}, 'sms')
    tr_t, tr_y, va_t, va_y, te_t, te_y = _split_text(texts, labels)
    classes = CONFIG['sms']['classes']
    dist = {classes[i]: int(np.count_nonzero(labels == i)) for i in range(2)}
    samples = [{'text': texts[i], 'label': classes[int(labels[i])]} for i in range(min(5, len(texts)))]
    return RawDataset('sms', tr_t, tr_y, va_t, va_y, te_t, te_y, classes,
                      {'all': len(labels), 'train': len(tr_y), 'val': len(va_y), 'test': len(te_y)}, dist, samples)


def _load_imdb():
    path = _find_file('imdb', 'IMDB Dataset.csv')
    df = _read_csv_flexible(path)
    cols = {str(c).strip().lower(): c for c in df.columns}
    text_col = cols.get('review') or cols.get('text') or cols.get('sentence')
    label_col = cols.get('sentiment') or cols.get('label') or cols.get('class')
    if text_col is None or label_col is None:
        raise ValueError(f'imdb: mong đợi cột review và sentiment. Cột hiện có: {list(df.columns)}')
    df = df[[text_col, label_col]].dropna().copy()
    texts = df[text_col].astype(str).tolist()
    labels = _normalize_binary(df[label_col], {'negative': 0, 'positive': 1, 'neg': 0, 'pos': 1, '0': 0, '1': 1}, 'imdb')
    tr_t, tr_y, va_t, va_y, te_t, te_y = _split_text(texts, labels)
    classes = CONFIG['imdb']['classes']
    dist = {classes[i]: int(np.count_nonzero(labels == i)) for i in range(2)}
    samples = [{'text': texts[i][:240], 'label': classes[int(labels[i])]} for i in range(min(5, len(texts)))]
    return RawDataset('imdb', tr_t, tr_y, va_t, va_y, te_t, te_y, classes,
                      {'all': len(labels), 'train': len(tr_y), 'val': len(va_y), 'test': len(te_y)}, dist, samples)


def _read_emotion_txt(path):
    texts, labels = [], []
    with Path(path).open('r', encoding='utf-8-sig', errors='replace') as f:
        for line_no, line in enumerate(f, 1):
            line = line.rstrip('\n\r')
            if not line.strip():
                continue
            if ';' not in line:
                raise ValueError(f'{Path(path).name}:{line_no}: thiếu dấu ; phân cách text và label')
            text, label = line.rsplit(';', 1)
            text, label = text.strip(), label.strip().lower()
            if not text or not label:
                raise ValueError(f'{Path(path).name}:{line_no}: dòng rỗng text/label')
            texts.append(text)
            labels.append(label)
    return texts, labels


def _load_emotion():
    train_p = _find_file('emotion', 'train.txt')
    val_p = _find_file('emotion', 'val.txt')
    test_p = _find_file('emotion', 'test.txt')
    tr_t, tr_raw = _read_emotion_txt(train_p)
    va_t, va_raw = _read_emotion_txt(val_p)
    te_t, te_raw = _read_emotion_txt(test_p)
    classes = CONFIG['emotion']['classes']
    enc = {name: i for i, name in enumerate(classes)}
    def enc_labels(raw, split):
        bad = sorted(set(raw) - set(enc))
        if bad:
            raise ValueError(f'emotion/{split}: nhãn lạ: {bad}')
        return np.asarray([enc[v] for v in raw], dtype=np.int64)
    tr_y, va_y, te_y = enc_labels(tr_raw, 'train'), enc_labels(va_raw, 'val'), enc_labels(te_raw, 'test')
    all_y = np.concatenate([tr_y, va_y, te_y])
    dist = {classes[i]: int(np.count_nonzero(all_y == i)) for i in range(len(classes))}
    samples = [{'text': tr_t[i], 'label': classes[int(tr_y[i])]} for i in range(min(5, len(tr_t)))]
    return RawDataset('emotion', tr_t, tr_y, va_t, va_y, te_t, te_y, classes,
                      {'all': len(all_y), 'train': len(tr_y), 'val': len(va_y), 'test': len(te_y)}, dist, samples)


def load_raw(key):
    if key == 'sms':
        return _load_sms()
    if key == 'imdb':
        return _load_imdb()
    if key == 'emotion':
        return _load_emotion()
    raise ValueError(f'Ứng dụng không hỗ trợ: {key}')


def _subset(texts, labels, limit, seed):
    if limit is None or limit <= 0 or limit >= len(labels):
        return list(texts), np.asarray(labels, dtype=np.int64)
    idx, _ = train_test_split(np.arange(len(labels)), train_size=int(limit), stratify=labels, random_state=seed)
    arr = np.asarray(texts, dtype=object)
    return arr[idx].tolist(), np.asarray(labels, dtype=np.int64)[idx]


def load_sequences(key, max_train=4000, max_val=800, max_test=1200,
                   max_len=60, vocab_size=4000, seed=SEED):
    raw = load_raw(key)
    tr_t, tr_y = _subset(raw.train_texts, raw.train_labels, max_train, seed)
    va_t, va_y = _subset(raw.val_texts, raw.val_labels, max_val, seed + 1)
    te_t, te_y = _subset(raw.test_texts, raw.test_labels, max_test, seed + 2)
    vocab = build_vocab(tr_t, max_vocab=vocab_size)
    xtr = encode_texts(tr_t, vocab, max_len=max_len)
    xva = encode_texts(va_t, vocab, max_len=max_len)
    xte = encode_texts(te_t, vocab, max_len=max_len)
    return SequenceBundle(key, xtr, tr_y, xva, va_y, xte, te_y, raw.classes, vocab,
                          raw.original_counts, raw.distribution, raw.samples)


def dataset_summary(key):
    raw = load_raw(key)
    return {
        'key': key,
        'title': CONFIG[key]['title'],
        'url': CONFIG[key]['url'],
        'counts': raw.original_counts,
        'distribution': raw.distribution,
        'samples': raw.samples,
        'classes': raw.classes,
    }
