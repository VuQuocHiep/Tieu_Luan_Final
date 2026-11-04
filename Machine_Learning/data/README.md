# Dữ liệu gốc từ Kaggle

Chạy ở thư mục gốc: `python download_kaggle.py`

| App | Kaggle | CSV chính xác | Điểm cần chú ý |
|---|---|---|---|
| Iris | https://www.kaggle.com/datasets/uciml/iris | `data/raw/iris/Iris.csv` | bỏ cột `Id`, nhãn `Species` |
| Wine | https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009 | `data/raw/wine/winequality-red.csv` | phân tách dấu `;`, label `quality >= 6` |
| Breast Cancer | https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data | `data/raw/breast_cancer/data.csv` | bỏ `id` và `Unnamed: 32`, label `diagnosis` |

Thư mục dữ liệu sẽ trống trước khi tải để tránh nhầm bộ dữ liệu giả với CSV gốc Kaggle.
Khi cần chạy offline, tải ba CSV bằng trình duyệt theo các URL trên và đặt đúng vị trí.
Không sử dụng dữ liệu số liệu từ `sklearn.datasets` làm nguồn huấn luyện.
