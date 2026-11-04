import json
from pathlib import Path
import numpy as np


class ScratchRNN:
    """Simple word-level Elman RNN with masking and Adam, implemented with NumPy."""
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, seed=42):
        rng = np.random.default_rng(seed)
        self.config = dict(vocab_size=int(vocab_size), embed_dim=int(embed_dim), hidden_dim=int(hidden_dim), n_classes=int(n_classes))
        self.p = {
            'E': rng.normal(0, 0.05, (vocab_size, embed_dim)).astype(np.float32),
            'Wx': (rng.normal(0, 1, (embed_dim, hidden_dim)) / np.sqrt(embed_dim)).astype(np.float32),
            'Wh': (rng.normal(0, 1, (hidden_dim, hidden_dim)) / np.sqrt(hidden_dim)).astype(np.float32),
            'bh': np.zeros(hidden_dim, dtype=np.float32),
            'Wy': (rng.normal(0, 1, (hidden_dim, n_classes)) / np.sqrt(hidden_dim)).astype(np.float32),
            'by': np.zeros(n_classes, dtype=np.float32),
        }
        self.p['E'][0] = 0.0
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.step = 0

    @staticmethod
    def _softmax(logits):
        z = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)

    def forward(self, x, cache=False):
        x = np.asarray(x, dtype=np.int64)
        bsz, steps = x.shape
        h = np.zeros((bsz, self.config['hidden_dim']), dtype=np.float32)
        caches = [] if cache else None
        E, Wx, Wh, bh = self.p['E'], self.p['Wx'], self.p['Wh'], self.p['bh']
        for t in range(steps):
            ids = x[:, t]
            emb = E[ids]
            h_prev = h
            cand = np.tanh(emb @ Wx + h_prev @ Wh + bh)
            mask = (ids != 0).astype(np.float32)[:, None]
            h = mask * cand + (1.0 - mask) * h_prev
            if cache:
                caches.append((ids, emb, h_prev, cand, mask))
        logits = h @ self.p['Wy'] + self.p['by']
        return (logits, h, caches) if cache else logits

    def loss_and_grads(self, x, y):
        logits, h_final, caches = self.forward(x, cache=True)
        probs = self._softmax(logits)
        n = len(y)
        loss = -np.log(np.maximum(probs[np.arange(n), y], 1e-12)).mean()
        dlog = probs
        dlog[np.arange(n), y] -= 1.0
        dlog /= n
        grads = {k: np.zeros_like(v) for k, v in self.p.items()}
        grads['Wy'] = h_final.T @ dlog
        grads['by'] = dlog.sum(axis=0)
        dh = dlog @ self.p['Wy'].T
        for ids, emb, h_prev, cand, mask in reversed(caches):
            dcand = dh * mask
            dpre = dcand * (1.0 - cand * cand)
            grads['Wx'] += emb.T @ dpre
            grads['Wh'] += h_prev.T @ dpre
            grads['bh'] += dpre.sum(axis=0)
            demb = dpre @ self.p['Wx'].T
            np.add.at(grads['E'], ids, demb)
            dh = dpre @ self.p['Wh'].T + dh * (1.0 - mask)
        grads['E'][0] = 0.0
        return float(loss), grads

    @staticmethod
    def _clip(grads, max_norm=5.0):
        total = np.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
        if total > max_norm:
            scale = max_norm / (total + 1e-12)
            for k in grads:
                grads[k] *= scale
        return total

    def adam_step(self, grads, lr=0.002, beta1=0.9, beta2=0.999, eps=1e-8):
        self._clip(grads)
        self.step += 1
        for k, g in grads.items():
            self.m[k] = beta1 * self.m[k] + (1.0 - beta1) * g
            self.v[k] = beta2 * self.v[k] + (1.0 - beta2) * (g * g)
            mhat = self.m[k] / (1.0 - beta1 ** self.step)
            vhat = self.v[k] / (1.0 - beta2 ** self.step)
            self.p[k] -= lr * mhat / (np.sqrt(vhat) + eps)
        self.p['E'][0] = 0.0

    def fit(self, x, y, epochs=8, batch_size=64, lr=0.002, seed=42, verbose=True):
        rng = np.random.default_rng(seed)
        history = []
        for epoch in range(1, int(epochs) + 1):
            order = rng.permutation(len(y))
            losses = []
            for start in range(0, len(y), int(batch_size)):
                idx = order[start:start + int(batch_size)]
                loss, grads = self.loss_and_grads(x[idx], y[idx])
                self.adam_step(grads, lr=lr)
                losses.append(loss)
            train_acc = float((self.predict(x) == y).mean())
            row = {'epoch': epoch, 'loss': float(np.mean(losses)), 'train_accuracy': train_acc}
            history.append(row)
            if verbose:
                print(f'epoch {epoch:02d}/{epochs} loss={row["loss"]:.4f} train_acc={train_acc:.4f}')
        return history

    def predict_proba(self, x, batch_size=512):
        out = []
        for start in range(0, len(x), batch_size):
            out.append(self._softmax(self.forward(x[start:start + batch_size])))
        return np.vstack(out) if out else np.empty((0, self.config['n_classes']), dtype=np.float32)

    def predict(self, x):
        return self.predict_proba(x).argmax(axis=1)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {**self.p, '__config__': np.array(json.dumps(self.config))}
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        config = json.loads(str(z['__config__'].item()))
        model = cls(**config)
        for k in model.p:
            model.p[k] = z[k].astype(np.float32)
        return model
