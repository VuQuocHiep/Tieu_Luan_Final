from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
ARTIFACTS = ROOT / 'artifacts'
REPORTS = ROOT / 'reports'
SEED = 42

CONFIG = {
    'sms': {
        'title': 'SMS Spam Detection',
        'short': 'SMS Spam',
        'dataset': 'uciml/sms-spam-collection-dataset',
        'url': 'https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset',
        'files': ['spam.csv'],
        'classes': ['Ham', 'Spam'],
        'description': 'Phân loại tin nhắn SMS hợp lệ hoặc spam.',
        'expected_total': 5574,
    },
    'imdb': {
        'title': 'IMDB Movie Review Sentiment',
        'short': 'IMDB Sentiment',
        'dataset': 'lakshmi25npathi/imdb-dataset-of-50k-movie-reviews',
        'url': 'https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews',
        'files': ['IMDB Dataset.csv'],
        'classes': ['Negative', 'Positive'],
        'description': 'Phân loại cảm xúc tích cực hoặc tiêu cực của bài đánh giá phim.',
        'expected_total': 50000,
    },
    'emotion': {
        'title': 'Emotion Classification',
        'short': 'Emotion',
        'dataset': 'praveengovi/emotions-dataset-for-nlp',
        'url': 'https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp',
        'files': ['train.txt', 'val.txt', 'test.txt'],
        'classes': ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'],
        'description': 'Phân loại câu tiếng Anh vào 6 trạng thái cảm xúc.',
        'expected_total': 20000,
    },
}

DEFAULTS = {
    'max_train': 4000,
    'max_val': 800,
    'max_test': 1200,
    'max_len': 60,
    'vocab_size': 4000,
    'embed_dim': 24,
    'hidden_dim': 32,
    'epochs': 8,
    'batch_size': 64,
    'lr': 0.002,
}
