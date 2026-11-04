"""Run all 9 training jobs and write comparison table.
Usage: python train_all.py
"""
import os
os.environ.setdefault('KERAS_BACKEND','torch')
from shared.train_runner import run_all
if __name__ == '__main__':
    run_all()
