import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.config import CONFIG
from shared.datasets import load_raw


def main():
    for key, spec in CONFIG.items():
        d = load_raw(key)
        total = d.original_counts['all']
        assert total > 0
        assert len(d.classes) == len(set(d.train_labels) | set(d.val_labels) | set(d.test_labels))
        assert all(str(x).strip() for x in d.train_texts[:20])
        print(f"PASS {key:8s} total={total:6d} train={d.original_counts['train']:6d} val={d.original_counts['val']:5d} test={d.original_counts['test']:5d} classes={len(d.classes)}")
        print('  samples:', [(s['label'], s['text'][:70].replace('\n',' ')) for s in d.samples[:2]])
    print('KAGGLE DATA VALIDATION PASSED: có thể chạy python train_all.py')

if __name__ == '__main__':
    main()
