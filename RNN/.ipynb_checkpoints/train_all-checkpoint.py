import argparse
from shared.config import DEFAULTS
from shared.train_runner import run_all

p = argparse.ArgumentParser(description='Train 3 RNN apps x Scratch/Keras/PyTorch')
p.add_argument('--dataset', choices=['all','sms','imdb','emotion'], default='all')
p.add_argument('--method', choices=['all','scratch','keras','pytorch'], default='all')
for name, default, typ in [
    ('max-train', DEFAULTS['max_train'], int), ('max-val', DEFAULTS['max_val'], int), ('max-test', DEFAULTS['max_test'], int),
    ('max-len', DEFAULTS['max_len'], int), ('vocab-size', DEFAULTS['vocab_size'], int),
    ('embed-dim', DEFAULTS['embed_dim'], int), ('hidden-dim', DEFAULTS['hidden_dim'], int),
    ('epochs', DEFAULTS['epochs'], int), ('batch-size', DEFAULTS['batch_size'], int), ('lr', DEFAULTS['lr'], float)]:
    p.add_argument('--'+name, type=typ, default=default)
a = p.parse_args()
opts = {k.replace('-','_'): v for k,v in vars(a).items() if k not in ('dataset','method')}
run_all(a.dataset, a.method, opts)
