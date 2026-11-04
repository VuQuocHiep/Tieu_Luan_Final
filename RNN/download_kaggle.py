from pathlib import Path
import shutil
from shared.config import CONFIG, RAW

try:
    import kagglehub
except ImportError:
    raise SystemExit('Thiếu kagglehub. Chạy: pip install kagglehub')

RAW.mkdir(parents=True, exist_ok=True)
for key, spec in CONFIG.items():
    print(f'\nDownloading {spec["dataset"]} ...')
    cache = Path(kagglehub.dataset_download(spec['dataset']))
    dest = RAW / key
    dest.mkdir(parents=True, exist_ok=True)
    for wanted in spec['files']:
        matches = [p for p in cache.rglob('*') if p.is_file() and p.name.lower() == wanted.lower()]
        if not matches:
            names = [p.name for p in cache.rglob('*') if p.is_file()][:30]
            raise FileNotFoundError(f'{key}: không tìm thấy {wanted}. File nhận được: {names}')
        target = dest / wanted
        shutil.copy2(matches[0], target)
        print(f'[OK] {key}: {target} ({target.stat().st_size:,} bytes)')
print('\nĐã chuẩn bị dữ liệu Kaggle. Tiếp tục: python tests/test_kaggle_data.py')
