"""Validate exact Kaggle CSV paths, schema and label distribution before training."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from shared.datasets import _read_csv,_to_arrays,file_paths,class_distribution
from shared.config import CONFIG


def main():
    for key,spec in CONFIG.items():
        train_path,test_path=file_paths(key)
        train_label=_read_csv(train_path,labels_only=True)
        test_label=_read_csv(test_path,labels_only=True)
        first=_read_csv(train_path,nrows=5)
        x,y=_to_arrays(first,key)
        assert len(train_label)==spec['n_train'],f'{key}: train size {len(train_label)}'
        assert len(test_label)==spec['n_test'],f'{key}: test size {len(test_label)}'
        assert x.shape==(5,1,28,28)
        assert (0<=x).all() and (x<=1).all()
        distribution=class_distribution(key)
        assert len(distribution)==len(spec['classes'])
        print(f'PASS {key:<14} train={len(train_label):>6} test={len(test_label):>5} classes={len(distribution)} first_labels={y.tolist()}')
    print('KAGGLE CSV VALIDATION PASSED: bạn có thể chạy python train_all.py')

if __name__=='__main__':main()
