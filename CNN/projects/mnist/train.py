"""Train all three CNN implementations for dataset mnist."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.train_runner import run_all

if __name__ == '__main__':
    run_all('mnist')
