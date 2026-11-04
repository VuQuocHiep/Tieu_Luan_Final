import sys
import os
os.environ.setdefault('KERAS_BACKEND','torch')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

mods=['numpy','pandas','sklearn','torch','keras','streamlit','plotly','kagglehub']
failed=[]
for m in mods:
    try:
        __import__(m); print(f'[OK] import {m}')
    except Exception as e:
        failed.append((m,str(e))); print(f'[FAIL] import {m}: {e}')
if failed:
    raise SystemExit('Thiếu thư viện. Chạy: pip install -r requirements.txt')

from shared.config import CONFIG
from shared.datasets import load_raw
for key in CONFIG:
    try:
        d=load_raw(key); print(f'[OK] data {key}: {d.original_counts}')
    except FileNotFoundError as e:
        print(f'[WAIT] data {key}: {e}')
    except Exception as e:
        print(f'[FAIL] data {key}: {e}'); raise
print('Doctor finished.')
