"""Đọc CSV tải từ Kaggle (không dùng sklearn.datasets / dữ liệu giả)."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.utils import Bunch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'raw'
SEED, EPOCHS, LEARNING_RATE = 42, 350, 0.15
SOURCES = {
    'iris': dict(name='Iris Species', vi_name='Phân loại hoa Iris', icon='🌼', domain='Sinh học',
        slug='uciml/iris', filename='Iris.csv', source='https://www.kaggle.com/datasets/uciml/iris',
        description='150 mẫu hoa, 4 đặc trưng đo đài và cánh hoa; phân loại thành 3 loài.',
        insight='Phân bố lớp cân bằng 50/50/50. Cột Id chỉ là định danh và được loại bỏ.',
        accent='#6CC8FF'),
    'wine': dict(name='Red Wine Quality',vi_name='Đánh giá chất lượng vang đỏ',icon='🍇',domain='Hóa học thực phẩm',
        slug='uciml/red-wine-quality-cortez-et-al-2009',filename='winequality-red.csv',
        source='https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009',
        description='1599 mẫu, 11 thuộc tính hóa lý; phân loại chất lượng từ 6 điểm trở lên và dưới 6.',
        insight='Phân bố hai nhãn khoảng 744/855; đây là bộ Red Wine Quality, KHÁC Wine 178 mẫu.',
        accent='#C9A7FF'),
    'breast_cancer': dict(name='Breast Cancer Wisconsin (Diagnostic)',vi_name='Phân loại đặc trưng tế bào',icon='🔬',domain='Dữ liệu nghiên cứu y sinh',
        slug='uciml/breast-cancer-wisconsin-data',filename='data.csv',
        source='https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data',
        description='569 mẫu, 30 đặc trưng nhân tế bào; hai nhãn Malignant và Benign.',
        insight='212 Malignant, 357 Benign: chưa cân bằng hoàn toàn; cần chú ý Recall lớp Malignant.',
        accent='#FFBC83'),
}


def dataset(name):
    if name not in SOURCES:
        raise ValueError(f'Bộ dữ liệu không hợp lệ: {name}')
    spec = SOURCES[name]
    csv = DATA/name/spec['filename']
    if not csv.is_file():
        raise FileNotFoundError(
            f'Chưa tìm thấy dữ liệu Kaggle: {csv}. Chạy `python download_kaggle.py` '
            'hoặc tải CSV theo link trong README.md và đặt đúng thư mục.')
    df = pd.read_csv(csv, sep=None, engine='python')
    if name == 'iris':
        features = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
        labels = df['Species'].astype(str).str.replace('Iris-', '', regex=False).str.lower()
        classes = ['setosa','versicolor','virginica']
    elif name == 'wine':
        features = ['fixed acidity','volatile acidity','citric acid','residual sugar',
                    'chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']
        quality = pd.to_numeric(df['quality'], errors='raise')
        labels = np.where(quality >= 6, 'Từ 6 điểm', 'Dưới 6 điểm')
        classes = ['Dưới 6 điểm','Từ 6 điểm']
    else:
        features = [x for x in df.columns if x not in ['id','diagnosis','Unnamed: 32'] and not x.startswith('Unnamed:')]
        if len(features) != 30:
            raise ValueError(f'Kaggle Breast Cancer cần 30 feature, CSV hiện có {len(features)}: {features}')
        labels = df['diagnosis'].map({'M':'Malignant (M)','B':'Benign (B)'})
        classes = ['Malignant (M)','Benign (B)']
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f'CSV {csv} thiếu các cột: {missing}')
    data = df[features].apply(pd.to_numeric, errors='raise').to_numpy(dtype=np.float64)
    if not np.isfinite(data).all():
        raise ValueError(f'Dữ liệu {csv} chứa feature thiếu hoặc không hữu hạn, cần làm sạch trước.')
    labelmap = {c:i for i,c in enumerate(classes)}
    if pd.isna(labels).any() or not set(np.asarray(labels)).issubset(labelmap):
        raise ValueError(f'CSV {csv} có nhãn không hợp lệ. Nhãn hợp lệ: {classes}')
    y = np.array([labelmap[x] for x in labels], dtype=np.int64)
    if len(np.unique(y)) != len(classes):
        raise ValueError('Tập dữ liệu chưa đủ lớp để phân loại.')
    return Bunch(data=data, target=y, target_names=np.array(classes),feature_names=np.array(features),
                 source_csv=str(csv), source_url=spec['source'], raw_shape=df.shape)


def load_split(name):
    raw = dataset(name)
    X, y = raw.data, raw.target
    idx = np.arange(len(y))
    trainval, test = train_test_split(idx, test_size=.15, stratify=y, random_state=SEED)
    train, valid = train_test_split(trainval, test_size=.15/.85, stratify=y[trainval], random_state=SEED)
    scaler = StandardScaler().fit(X[train])
    sets = tuple((scaler.transform(X[ids]), y[ids]) for ids in (train,valid,test))
    return raw, scaler, sets, (train,valid,test)
