# MACHINE LEARNING LAB — 3 ứng dụng Kaggle × 3 cách cài đặt

Đồ án minh họa Chương 2: **cùng một bài toán Logistic Regression (Softmax)**, cài đặt bằng NumPy từ đầu, Keras 3 và PyTorch. Mỗi ứng dụng có một trang Streamlit riêng và thư mục code riêng. Dữ liệu tải **trực tiếp từ Kaggle** bằng `download_kaggle.py`, KHÔNG gọi `sklearn.datasets.load_*()`.

## Dữ liệu đầu vào

| App | Dataset Kaggle | CSV lưu trong project | Nhiệm vụ |
|---|---|---|---|
| Iris | https://www.kaggle.com/datasets/uciml/iris | `data/raw/iris/Iris.csv` | 3 loài hoa, 150 mẫu, 4 feature |
| Wine | https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009 | `data/raw/wine/winequality-red.csv` | Rượu vang đỏ, 1599 mẫu, 11 feature; chất lượng `>=6` |
| Breast Cancer | https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data | `data/raw/breast_cancer/data.csv` | 569 mẫu, 30 feature; nhãn M/B |

**QUAN TRỌNG:** Wine ở đây là *Red Wine Quality* 1599 mẫu (2 nhóm đánh giá), không phải bộ *Wine Recognition* 178 mẫu (3 nhóm). Số liệu đánh giá của project trước không được dùng lại. Dữ liệu bệnh học chỉ phục vụ nghiên cứu, không dùng chẩn đoán.

## Bắt đầu (VS Code, IntelliJ/PyCharm)

Khuyến nghị Python 3.11/3.12. Mở Terminal tại thư mục có `app.py`:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python download_kaggle.py
python train_all.py
python tests/test_project.py
python -m scripts.export_results
streamlit run app.py
```

Mở trang `http://localhost:8501`. Có thể chạy `python -m projects.iris.train`, `python -m projects.wine.train`, `python -m projects.breast_cancer.train` để huấn luyện riêng từng app.

Nếu không có kết nối hoặc Kaggle yêu cầu tài khoản, vào 3 link bên trên → Download → giải nén CSV → đặt vào đúng đường dẫn. Sau đó chạy `python train_all.py`. Script KHÔNG tự thay thế bằng dữ liệu có tên tương tự từ thư viện khác.

## Cấu trúc project

```text
ML_Kaggle_3_Apps/
├── app.py                          # Dashboard: chọn 1 trong 3 app
├── pages/
│   ├── 01_Iris.py                  # App 1: phân loại hoa
│   ├── 02_Wine.py                  # App 2: phân loại chất lượng vang đỏ
│   └── 03_Breast_Cancer.py         # App 3: nhãn nghiên cứu tế bào
├── data/
│   ├── README.md                   # Nguồn Kaggle, cách tải CSV
│   └── raw/
│       ├── iris/Iris.csv            # Tải sau khi chạy download_kaggle.py
│       ├── wine/winequality-red.csv
│       └── breast_cancer/data.csv
├── projects/
│   ├── iris/
│   │   ├── scratch.py              # Softmax Regression tự tính gradient
│   │   ├── keras.py                # Keras Dense & CrossEntropy
│   │   ├── pytorch.py              # torch.nn.Linear & CrossEntropy
│   │   └── train.py                # Huấn luyện riêng Iris
│   ├── wine/
│   │   ├── scratch.py
│   │   ├── keras.py
│   │   ├── pytorch.py
│   │   └── train.py
│   └── breast_cancer/
│       ├── scratch.py
│       ├── keras.py
│       ├── pytorch.py
│       └── train.py
├── shared/
│   ├── datasets.py                 # Đọc CSV Kaggle & mapping nhãn
│   ├── train_runner.py             # Chạy, lưu mô hình & metrics
│   ├── inference.py                # Dự đoán từ checkpoint
│   ├── ui.py                       # Giao diện chung đẹp, Plotly
│   └── math_ops.py                 # Softmax và tiện ích
├── artifacts/                      # Sinh tự động sau train_all.py
│   ├── metrics.csv                 # Bảng 9 phép thử
│   ├── iris_{scratch,keras,pytorch}.*
│   ├── wine_{scratch,keras,pytorch}.*
│   └── breast_cancer_{scratch,keras,pytorch}.*
├── reports/
│   └── Chuong_2_Kaggle_ML.docx     # Báo cáo 10–15 trang
├── scripts/export_results.py        # Xuất số liệu thực nghiệm từ CSV Kaggle
├── download_kaggle.py              # Lấy dữ liệu từ Kaggle
├── train_all.py                    # Huấn luyện tất cả 9 mô hình
├── tests/test_project.py           # Kiểm tra chia tập, score & checkpoint
├── requirements.txt
└── README.md
```

## Đánh giá và minh bạch kết quả

- Chia tập phân tầng ~70% train / 15% validation / 15% test, random seed 42.
- StandardScaler `fit` trên train, `transform` validation/test (tránh rò rỉ dữ liệu).
- Cả 3 cách cài đặt dùng cùng khởi tạo, SGD full-batch, 350 epoch, LR 0.15.
- Chấm Accuracy, Precision/Recall/F1 macro, Log-loss và Confusion Matrix; so sánh cả thời gian huấn luyện.
- File `artifacts/metrics.csv` chỉ được tạo sau khi chạy trên CSV Kaggle thực. Không tái sử dụng chỉ số cũ từ các nguồn khác.
- Keras 3 sử dụng backend PyTorch (`KERAS_BACKEND=torch`); so sánh *API triển khai*, không phải 3 thuật toán ML khác nhau.

## Giao diện và Deploy

Trang chủ dạng dark dashboard. Mỗi app gồm 4 tab: **Khám phá dữ liệu** (nguồn, kích cỡ, 5 mẫu và phân bố), **Dự đoán** (nhập mẫu hoặc tải CSV), **Đánh giá** (bảng, biểu đồ so sánh, confusion matrix), **Source code** (đọc/tải 3 file code).

Để deploy lên Streamlit Community Cloud: chạy `download_kaggle.py` và `train_all.py` **trước**; đưa CSV công khai được phép chia sẻ và các checkpoint nhỏ vào repo hoặc dùng phương án tải dữ liệu và train tại build. Thiết lập `Main file path = app.py`, cài thư viện từ `requirements.txt`. Không đăng khóa Kaggle API hoặc dữ liệu riêng tư lên GitHub.

**Trạng thái kiểm thử:** source code được thiết kế cho dữ liệu gốc Kaggle. Trong môi trường tạo project hiện tại không thể kết nối Kaggle để tải CSV và chạy benchmark cuối trên nguồn đó, vì vậy **không có số Accuracy/F1 Kaggle được xác nhận sẵn**. Chạy các lệnh trên máy có mạng để tạo kết quả thật.
