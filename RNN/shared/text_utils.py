import html
import re
from collections import Counter
import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)


def clean_text(text):
    text = html.unescape(str(text))
    text = re.sub(r'<[^>]+>', ' ', text)
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text


def tokenize(text):
    return TOKEN_RE.findall(clean_text(text))


def build_vocab(texts, max_vocab=4000):
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for token, _ in counter.most_common(max(0, int(max_vocab) - 2)):
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def encode_texts(texts, vocab, max_len=60):
    n = len(texts)
    out = np.zeros((n, int(max_len)), dtype=np.int64)
    unk = vocab.get('<UNK>', 1)
    for i, text in enumerate(texts):
        ids = [vocab.get(tok, unk) for tok in tokenize(text)]
        ids = ids[:max_len]
        if ids:
            out[i, :len(ids)] = ids
    return out


def lengths_from_ids(x):
    return np.count_nonzero(x, axis=1).astype(np.int64)
