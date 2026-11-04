"""Download official CSV files from Kaggle via kagglehub (Internet required)."""
from pathlib import Path
import shutil
from shared.config import CONFIG,RAW

def main():
    try:import kagglehub
    except ImportError as exc:raise SystemExit('Cài thư viện: pip install kagglehub') from exc
    for key,spec in CONFIG.items():
        dst=RAW/key;dst.mkdir(parents=True,exist_ok=True)
        if all((dst/spec[n]).exists() for n in ('train','test')):
            print(f'[SKIP] {key}: đã có 2 file CSV');continue
        print(f'Đang tải {spec["kaggle"]} ...',flush=True)
        folder=Path(kagglehub.dataset_download(spec['kaggle']))
        for tag in ('train','test'):
            filename=spec[tag]
            matches=[p for p in folder.rglob('*.csv') if p.name.lower()==filename.lower()]
            if not matches:
                raise FileNotFoundError(f'Kaggle {key}: thiếu {filename}. Các CSV hiện có: {[p.name for p in folder.rglob("*.csv")]}')
            shutil.copy2(matches[0],dst/filename)
            print(f'[OK] {dst/filename} ({(dst/filename).stat().st_size:,} bytes)',flush=True)
    print('Đã chuẩn bị CSV Kaggle. Tiếp tục: python train_all.py')

if __name__=='__main__':main()
