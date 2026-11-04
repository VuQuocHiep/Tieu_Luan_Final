"""Read original Kaggle CSV, detect schema, and preserve official test holdout."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import CONFIG, RAW, SEED


@dataclass
class DataBundle:
    key: str

    x_train: np.ndarray
    y_train: np.ndarray

    x_val: np.ndarray
    y_val: np.ndarray

    x_test: np.ndarray
    y_test: np.ndarray

    classes: list
    label_codes: list

    n_train_original: int
    n_test_original: int

    train_distribution: dict

    example_images: np.ndarray
    example_labels: np.ndarray


# ============================================================
# FILE PATH
# ============================================================

def file_paths(key):
    if key not in CONFIG:
        raise ValueError(f"Dataset không hỗ trợ: {key}")

    s = CONFIG[key]

    train_file = RAW / key / s["train"]
    test_file = RAW / key / s["test"]

    return train_file, test_file


# ============================================================
# READ CSV
# ============================================================

def _detect_separator(path):
    """
    Tự xác định CSV dùng ',' hay ';'.
    """

    path = Path(path)

    with path.open("r", encoding="utf-8-sig") as f:
        header = f.readline()

    if header.count(";") > header.count(","):
        return ";"

    return ","


def _read_csv(path, nrows=None, labels_only=False):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Chưa có file {path.name}.\n"
            f"Hãy chạy:\n"
            f"python download_kaggle.py"
        )

    sep = _detect_separator(path)

    # --------------------------------------------------------
    # Chỉ đọc label
    # --------------------------------------------------------
    if labels_only:
        header_df = pd.read_csv(
            path,
            sep=sep,
            nrows=0,
            encoding="utf-8-sig"
        )

        header_df.columns = [
            str(c).strip().replace("\ufeff", "")
            for c in header_df.columns
        ]

        label_col = next(
            (
                c
                for c in header_df.columns
                if str(c).strip().lower() == "label"
            ),
            None
        )

        if label_col is None:
            raise ValueError(
                f"{path.name}: không tìm thấy cột label.\n"
                f"Các cột đầu: {list(header_df.columns)[:10]}"
            )

        df = pd.read_csv(
            path,
            sep=sep,
            usecols=[label_col],
            encoding="utf-8-sig"
        )

        df.columns = [
            str(c).strip().replace("\ufeff", "")
            for c in df.columns
        ]

        return df

    # --------------------------------------------------------
    # Đọc toàn bộ dữ liệu
    # --------------------------------------------------------
    df = pd.read_csv(
        path,
        sep=sep,
        nrows=nrows,
        encoding="utf-8-sig"
    )

    df.columns = [
        str(c).strip().replace("\ufeff", "")
        for c in df.columns
    ]

    # Loại các cột thừa dạng:
    # Unnamed: 0
    # Unnamed: 785...
    unnamed = [
        c
        for c in df.columns
        if str(c).strip().lower().startswith("unnamed:")
    ]

    if unnamed:
        df = df.drop(columns=unnamed)

    return df


# ============================================================
# FIND LABEL COLUMN
# ============================================================

def _label_column(df):
    label_col = next(
        (
            c
            for c in df.columns
            if str(c).strip().lower() == "label"
        ),
        None
    )

    if label_col is None:
        raise ValueError(
            "CSV không có cột 'label'. "
            f"Các cột đầu tiên: {list(df.columns)[:10]}"
        )

    return label_col


# ============================================================
# FIND PIXEL COLUMNS
# ============================================================

def _pixel_columns(df, label_col):
    """
    Dataset ảnh 28x28 cần đúng 784 giá trị pixel.

    Không phụ thuộc tên cột.

    Hỗ trợ:
        MNIST:
            1x1, 1x2, ..., 28x28

        Fashion-MNIST:
            pixel1, pixel2, ..., pixel784

        Sign Language:
            pixel1, pixel2, ..., pixel784
    """

    pixels = []

    for c in df.columns:

        if c == label_col:
            continue

        name = str(c).strip()

        if name.lower().startswith("unnamed:"):
            continue

        pixels.append(c)

    if len(pixels) != 784:
        raise ValueError(
            "\nSai cấu trúc dataset ảnh.\n"
            f"Cần: 784 cột pixel\n"
            f"Thực tế: {len(pixels)} cột\n"
            f"Label column: {label_col}\n"
            f"10 cột đầu CSV: {list(df.columns)[:10]}\n"
            f"10 cột pixel đầu: {pixels[:10]}"
        )

    return pixels


# ============================================================
# DATAFRAME -> NUMPY
# ============================================================

def _to_arrays(df, key):
    """
    Chuyển DataFrame Kaggle:

        label + 784 pixels

    thành:

        X: (N, 1, 28, 28)
        y: (N,)
    """

    label_col = _label_column(df)

    pixels = _pixel_columns(
        df,
        label_col
    )

    # --------------------------------------------------------
    # Pixel -> float32
    # --------------------------------------------------------
    try:
        pixel_data = (
            df[pixels]
            .apply(pd.to_numeric, errors="raise")
            .to_numpy(dtype=np.float32)
        )

    except Exception as e:
        raise ValueError(
            f"{key}: có pixel không phải dữ liệu số."
        ) from e

    # --------------------------------------------------------
    # Kiểm tra giá trị pixel
    # --------------------------------------------------------
    if not np.isfinite(pixel_data).all():
        raise ValueError(
            f"{key}: dữ liệu pixel chứa NaN hoặc infinity."
        )

    min_pixel = pixel_data.min()
    max_pixel = pixel_data.max()

    if min_pixel < 0 or max_pixel > 255:
        raise ValueError(
            f"{key}: pixel phải nằm trong [0,255], "
            f"nhưng nhận min={min_pixel}, max={max_pixel}"
        )

    # --------------------------------------------------------
    # reshape:
    # N x 784
    #
    # ->
    #
    # N x 1 x 28 x 28
    # --------------------------------------------------------
    x = pixel_data.reshape(
        -1,
        1,
        28,
        28
    )

    # Normalize 0 -> 1
    x = x / 255.0

    # --------------------------------------------------------
    # Label
    # --------------------------------------------------------
    try:
        y_raw = pd.to_numeric(
            df[label_col],
            errors="raise"
        ).to_numpy(dtype=np.int64)

    except Exception as e:
        raise ValueError(
            f"{key}: cột label chứa dữ liệu không hợp lệ."
        ) from e

    if len(x) != len(y_raw):
        raise ValueError(
            f"{key}: số ảnh và số label không bằng nhau."
        )

    return x, y_raw


# ============================================================
# SELECT STRATIFIED SUBSET
# ============================================================

def select_subset(x, y, limit, seed):
    """
    Giảm số lượng mẫu để train nhanh,
    nhưng vẫn giữ phân bố lớp.
    """

    if limit is None:
        return x, y

    if limit <= 0:
        return x, y

    if limit >= len(y):
        return x, y

    classes = np.unique(y)

    if limit < len(classes) * 2:
        raise ValueError(
            f"Giới hạn {limit} quá nhỏ so với "
            f"{len(classes)} lớp.\n"
            f"Nên dùng ít nhất {len(classes) * 2} mẫu."
        )

    indices = np.arange(len(y))

    selected, _ = train_test_split(
        indices,
        train_size=limit,
        stratify=y,
        random_state=seed
    )

    return x[selected], y[selected]


# ============================================================
# ENCODE LABEL
# ============================================================

def _encode_labels(y_train_raw, y_test_raw, key):
    """
    Chuyển label gốc thành:
        0,1,2,...,C-1

    Đặc biệt Sign Language:
        label gốc:
        0..8, 10..24

    được chuyển thành:
        0..23
    """

    label_codes = sorted(
        set(map(int, y_train_raw))
        |
        set(map(int, y_test_raw))
    )

    # --------------------------------------------------------
    # Sign Language MNIST không có J và Z
    #
    # J = 9
    # Z = 25
    # --------------------------------------------------------
    if key == "sign_language":

        if 9 in label_codes:
            raise ValueError(
                "Sign Language MNIST không được chứa "
                "label 9 (chữ J)."
            )

        if 25 in label_codes:
            raise ValueError(
                "Sign Language MNIST không được chứa "
                "label 25 (chữ Z)."
            )

    expected_classes = len(
        CONFIG[key]["classes"]
    )

    if len(label_codes) != expected_classes:
        raise ValueError(
            f"{key}: mong đợi {expected_classes} lớp "
            f"nhưng CSV có {len(label_codes)} lớp.\n"
            f"Label tìm thấy: {label_codes}"
        )

    encode = {
        original: index
        for index, original in enumerate(label_codes)
    }

    y_train = np.array(
        [
            encode[int(v)]
            for v in y_train_raw
        ],
        dtype=np.int64
    )

    y_test = np.array(
        [
            encode[int(v)]
            for v in y_test_raw
        ],
        dtype=np.int64
    )

    return y_train, y_test, label_codes


# ============================================================
# LOAD DATA
# ============================================================

def load_data(
    key,
    max_train=1600,
    max_val=450,
    max_test=900,
    seed=SEED
):
    """
    Quy trình:

    Kaggle TRAIN
        |
        +--> Train
        |
        +--> Validation

    Kaggle TEST
        |
        +--> Test

    Official Kaggle test KHÔNG dùng để train.
    """

    if key not in CONFIG:
        raise ValueError(
            f"Ứng dụng không hỗ trợ: {key}"
        )

    train_file, test_file = file_paths(key)

    # ========================================================
    # READ CSV
    # ========================================================

    train_df = _read_csv(
        train_file
    )

    test_df = _read_csv(
        test_file
    )

    # ========================================================
    # CONVERT IMAGE
    # ========================================================

    x, y_train_raw = _to_arrays(
        train_df,
        key
    )

    tx, y_test_raw = _to_arrays(
        test_df,
        key
    )

    n_train_original = len(
        y_train_raw
    )

    n_test_original = len(
        y_test_raw
    )

    # ========================================================
    # ENCODE LABEL
    # ========================================================

    y, ty, label_codes = _encode_labels(
        y_train_raw,
        y_test_raw,
        key
    )

    classes = CONFIG[key]["classes"]

    # ========================================================
    # CLASS DISTRIBUTION
    # ========================================================

    distribution = {}

    for i, class_name in enumerate(classes):
        distribution[class_name] = int(
            np.count_nonzero(
                y == i
            )
        )

    # ========================================================
    # SAVE EXAMPLES
    # ========================================================

    number_examples = min(
        8,
        len(x)
    )

    example_images = (
        x[:number_examples]
        .copy()
    )

    example_labels = (
        y[:number_examples]
        .copy()
    )

    # ========================================================
    # TRAIN / VALIDATION
    # ========================================================

    indices = np.arange(
        len(y)
    )

    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.15,
        stratify=y,
        random_state=seed
    )

    x_train = x[train_idx]
    y_train = y[train_idx]

    x_val = x[val_idx]
    y_val = y[val_idx]

    # ========================================================
    # SUBSET FOR QUICK DEMO
    # ========================================================

    x_train, y_train = select_subset(
        x_train,
        y_train,
        max_train,
        seed
    )

    x_val, y_val = select_subset(
        x_val,
        y_val,
        max_val,
        seed + 1
    )

    x_test, y_test = select_subset(
        tx,
        ty,
        max_test,
        seed + 2
    )

    return DataBundle(
        key=key,

        x_train=x_train,
        y_train=y_train,

        x_val=x_val,
        y_val=y_val,

        x_test=x_test,
        y_test=y_test,

        classes=classes,
        label_codes=label_codes,

        n_train_original=n_train_original,
        n_test_original=n_test_original,

        train_distribution=distribution,

        example_images=example_images,
        example_labels=example_labels
    )


# ============================================================
# PREVIEW DATA
# ============================================================

def preview_data(key, count=8):
    """
    Dùng cho giao diện Streamlit:
        - đọc vài dòng đầu
        - hiển thị ảnh
        - hiển thị label
        - hiển thị một vài pixel
    """

    train_file, _ = file_paths(key)

    df = _read_csv(
        train_file,
        nrows=count
    )

    x, y_raw = _to_arrays(
        df,
        key
    )

    label_col = _label_column(
        df
    )

    pixels = _pixel_columns(
        df,
        label_col
    )

    # --------------------------------------------------------
    # Convert raw label -> class name
    # --------------------------------------------------------

    if key == "sign_language":

        valid_codes = [
            i
            for i in range(26)
            if i not in (9, 25)
        ]

        code_to_name = {
            code: CONFIG[key]["classes"][i]
            for i, code in enumerate(valid_codes)
        }

        labels = [
            code_to_name.get(
                int(v),
                str(v)
            )
            for v in y_raw
        ]

    else:

        labels = [
            CONFIG[key]["classes"][int(v)]
            for v in y_raw
        ]

    # --------------------------------------------------------
    # Chỉ lấy vài cột pixel để bảng không quá rộng
    # --------------------------------------------------------

    sample_cols = (
        pixels[:8]
        +
        pixels[-4:]
    )

    preview_table = df[
        [label_col]
        +
        sample_cols
    ].copy()

    return (
        x,
        labels,
        preview_table
    )


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

def class_distribution(key):
    """
    Đọc riêng cột label để thống kê.
    Không cần load toàn bộ 784 pixel vào RAM.
    """

    train_file, _ = file_paths(key)

    df = _read_csv(
        train_file,
        labels_only=True
    )

    label_col = _label_column(
        df
    )

    raw_distribution = (
        df[label_col]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    raw_codes = sorted(
        int(k)
        for k in raw_distribution.keys()
    )

    classes = CONFIG[key]["classes"]

    if len(raw_codes) != len(classes):
        raise ValueError(
            f"{key}: số lớp trong CSV "
            f"({len(raw_codes)}) "
            f"khác CONFIG "
            f"({len(classes)})."
        )

    result = {}

    for i, code in enumerate(raw_codes):

        result[
            classes[i]
        ] = int(
            raw_distribution[code]
        )

    return result