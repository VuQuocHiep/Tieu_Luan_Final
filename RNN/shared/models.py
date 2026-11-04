import os
os.environ.setdefault('KERAS_BACKEND', 'torch')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
from pathlib import Path
import json
import numpy as np

from .rnn_scratch import ScratchRNN


def train_scratch(bundle, opts, out_path):
    model = ScratchRNN(len(bundle.vocab), opts['embed_dim'], opts['hidden_dim'], len(bundle.classes), seed=opts['seed'])
    history = model.fit(bundle.x_train, bundle.y_train, epochs=opts['epochs'], batch_size=opts['batch_size'], lr=opts['lr'], seed=opts['seed'])
    model.save(out_path)
    return model, history


def load_scratch(path):
    return ScratchRNN.load(path)


def train_keras(bundle, opts, out_path):
    import keras
    keras.utils.set_random_seed(opts['seed'])
    model = keras.Sequential([
        keras.layers.Input(shape=(bundle.x_train.shape[1],), dtype='int32'),
        keras.layers.Embedding(len(bundle.vocab), opts['embed_dim'], mask_zero=True),
        keras.layers.SimpleRNN(opts['hidden_dim'], activation='tanh'),
        keras.layers.Dense(len(bundle.classes)),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=opts['lr']),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    h = model.fit(bundle.x_train, bundle.y_train, validation_data=(bundle.x_val, bundle.y_val),
                  epochs=opts['epochs'], batch_size=opts['batch_size'], shuffle=True, verbose=1)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(out_path)
    history = []
    for i in range(len(h.history['loss'])):
        history.append({'epoch': i + 1, 'loss': float(h.history['loss'][i]),
                        'train_accuracy': float(h.history.get('accuracy', [0]*len(h.history['loss']))[i]),
                        'val_accuracy': float(h.history.get('val_accuracy', [0]*len(h.history['loss']))[i])})
    return model, history


def keras_predict_proba(model, x, batch_size=512):
    import keras
    logits = model.predict(x, batch_size=batch_size, verbose=0)
    return keras.ops.convert_to_numpy(keras.ops.softmax(logits, axis=-1))


def load_keras(path):
    import keras
    return keras.models.load_model(path)


def _torch_classes():
    import torch
    import torch.nn as nn

    class TorchRNN(nn.Module):
        def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.cell = nn.RNNCell(embed_dim, hidden_dim, nonlinearity='tanh')
            self.fc = nn.Linear(hidden_dim, n_classes)
            self.cfg = dict(vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim, n_classes=n_classes)

        def forward(self, x):
            emb = self.embedding(x)
            h = torch.zeros((x.size(0), self.cell.hidden_size), dtype=emb.dtype, device=x.device)
            for t in range(x.size(1)):
                cand = self.cell(emb[:, t], h)
                m = (x[:, t] != 0).to(emb.dtype).unsqueeze(1)
                h = m * cand + (1.0 - m) * h
            return self.fc(h)
    return torch, nn, TorchRNN


def train_pytorch(bundle, opts, out_path):
    torch, nn, TorchRNN = _torch_classes()
    torch.manual_seed(opts['seed'])
    model = TorchRNN(len(bundle.vocab), opts['embed_dim'], opts['hidden_dim'], len(bundle.classes))
    opt = torch.optim.Adam(model.parameters(), lr=opts['lr'])
    criterion = nn.CrossEntropyLoss()
    rng = np.random.default_rng(opts['seed'])
    history = []
    for epoch in range(1, opts['epochs'] + 1):
        model.train()
        order = rng.permutation(len(bundle.y_train))
        losses = []
        correct = total = 0
        for start in range(0, len(order), opts['batch_size']):
            idx = order[start:start + opts['batch_size']]
            xb = torch.as_tensor(bundle.x_train[idx], dtype=torch.long)
            yb = torch.as_tensor(bundle.y_train[idx], dtype=torch.long)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.item()))
            correct += int((logits.argmax(1) == yb).sum().item())
            total += len(idx)
        row = {'epoch': epoch, 'loss': float(np.mean(losses)), 'train_accuracy': correct / max(total, 1)}
        history.append(row)
        print(f'epoch {epoch:02d}/{opts["epochs"]} loss={row["loss"]:.4f} train_acc={row["train_accuracy"]:.4f}')
    out = Path(out_path); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'config': model.cfg, 'state_dict': model.state_dict()}, out)
    return model, history


def pytorch_predict_proba(model, x, batch_size=512):
    torch, _, _ = _torch_classes()
    model.eval(); out = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            logits = model(torch.as_tensor(x[start:start + batch_size], dtype=torch.long))
            out.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.vstack(out)


def load_pytorch(path):
    torch, _, TorchRNN = _torch_classes()
    obj = torch.load(path, map_location='cpu', weights_only=False)
    model = TorchRNN(**obj['config'])
    model.load_state_dict(obj['state_dict'])
    model.eval()
    return model
