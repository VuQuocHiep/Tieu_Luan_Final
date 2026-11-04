"""Tải 3 bộ CSV GỐC từ 3 trang Kaggle, lưu vào data/raw/.

Usage: python download_kaggle.py
Nếu Kaggle yêu cầu token, đăng nhập Kaggle theo hướng dẫn của kagglehub.
Không thay thế CSV bằng bộ load_*() của scikit-learn.
"""
from pathlib import Path
import shutil
import argparse
import kagglehub
from shared.datasets import SOURCES, DATA


def fetch_one(key):
    spec=SOURCES[key]
    target=DATA/key/spec['filename']
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        print(f'[OK] Đã tồn tại: {target}')
        return target
    folder=Path(kagglehub.dataset_download(spec['slug']))
    matches = list(folder.rglob(spec['filename']))
    if len(matches)!=1:
        raise FileNotFoundError(f'Không tìm được chính xác {spec["filename"]} trong {folder}: {matches}')
    shutil.copy2(matches[0],target)
    print(f'[OK] Kaggle {spec["slug"]} => {target} ({target.stat().st_size:,} bytes)')
    return target

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',choices=['all',*SOURCES],default='all')
    args=ap.parse_args()
    for key in (list(SOURCES) if args.dataset=='all' else [args.dataset]):
        fetch_one(key)
    print('Đã chuẩn bị dữ liệu. Chạy: python train_all.py')
