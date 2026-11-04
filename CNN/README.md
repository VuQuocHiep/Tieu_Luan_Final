# CNN Vision Lab — Chương 3: CNN (3 Kaggle Apps)

Project học tập CNN gồm **3 ứng dụng độc lập**, mỗi ứng dụng có **3 phương pháp code riêng**: NumPy From Scratch, Keras 3 và PyTorch. Cả 9 phiên bản dùng CNN giống nhau để đánh giá công bằng. Giao diện Streamlit nền xanh đậm, mỗi app có 4 tab: Dataset, Dự đoán, Đánh giá, Source Code.

## 1. Dataset gốc (không dùng `torchvision.datasets` hoặc `keras.datasets` để tải)

| App | Kaggle | CSV | Số mẫu (train / test) | Số lớp |
|---|---|---|---|---|
| MNIST chữ số | https://www.kaggle.com/datasets/oddrationale/mnist-in-csv | `mnist_train.csv`, `mnist_test.csv` | 60.000 / 10.000 | 10 |
| Fashion-MNIST | https://www.kaggle.com/datasets/zalando-research/fashionmnist | `fashion-mnist_train.csv`, `fashion-mnist_test.csv` | 60.000 / 10.000 | 10 |
| Sign Language MNIST | https://www.kaggle.com/datasets/datamunge/sign-language-mnist | `sign_mnist_train.csv`, `sign_mnist_test.csv` | 27.455 / 7.172 | 24 |

Mỗi ảnh 28×28 grayscale (cột `label` + 784 cột pixel). Dataset Sign Language không có nhãn 9 (J) và 25 (Z). Dataset được tải bằng `kagglehub` vào `data/raw`; hai CSV train/test được giữ tách biệt.

## 2. Cấu trúc thư mục

```text
CNN_Kaggle_3_Apps/
├── app.py                              # Trang chủ tổng quan
├── pages/
│   ├── 01_MNIST.py                     # App 1 – nhận dạng chữ số
│   ├── 02_Fashion_MNIST.py             # App 2 – phân loại trang phục
│   └── 03_Sign_Language.py             # App 3 – nhận dạng ký hiệu ASL
├── projects/
│   ├── mnist/
│   │   ├── scratch.py                  # App 1 / NumPy
│   │   ├── keras.py                    # App 1 / Keras
│   │   ├── pytorch.py                  # App 1 / PyTorch
│   │   └── train.py                    # Train toàn bộ app 1
│   ├── fashion/
│   │   ├── scratch.py
│   │   ├── keras.py
│   │   ├── pytorch.py
│   │   └── train.py
│   └── sign_language/
│       ├── scratch.py
│       ├── keras.py
│       ├── pytorch.py
│       └── train.py
├── shared/
│   ├── config.py                      # Metadata, URL Kaggle
│   ├── datasets.py                    # Đọc CSV, chuẩn hóa, train/val/test
│   ├── cnn_numpy.py                   # TÍCH CHẬP và BACKPROP thuần NumPy
│   ├── backends.py                    # Kiến trúc CNN Keras / PyTorch
│   ├── train_runner.py                # Huấn luyện, đánh giá, lưu models
│   ├── inference.py                   # Tải model, dự đoán 28×28
│   └── ui.py                          # Dashboard, chart, confusion matrix
├── data/raw/
│   ├── mnist/                          # CSV được tải về
│   ├── fashion/
│   └── sign_language/
├── artifacts/                          # 9 models & JSON metrics sau train
├── reports/
│   ├── Chuong_3_CNN_10_15_trang.docx
│   └── KET_QUA_CAN_DAN_VAO_CHUONG_3.md  # Tạo khi export
├── tests/test_project.py             # Unit & integration test offline
├── tests/test_kaggle_data.py          # Kiểm tra cấu trúc CSV Kaggle thật
├── scripts/export_results.py         # Kết quả thực nghiệm từ CSV thật
├── download_kaggle.py
├── train_all.py
├── requirements.txt
└── .streamlit/config.toml
```

> **Tách source:** 9 file `projects/<app>/<framework>.py` là điểm chạy cho từng ứng dụng. Thuật toán từ đầu nằm trong `shared/cnn_numpy.py`; định nghĩa Keras/PyTorch nằm trong `shared/backends.py` để không copy-paste cùng công thức 9 lần và giúp kiểm soát tính công bằng.

## 3. Cài đặt và chạy trong VS Code / PyCharm (Windows)

Tạo môi trường Python 3.10–3.12 khuyến nghị; mở Terminal ở thư mục project.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python download_kaggle.py
python tests/test_kaggle_data.py
python train_all.py
python tests/test_project.py
python scripts/export_results.py
streamlit run app.py
```

Nếu lỗi quyền kích hoạt PowerShell: mở `cmd` hoặc chạy trực tiếp `.venv\Scripts\python.exe`.

Sau khi chạy thành công, mở **http://localhost:8501**. Dùng sidebar chọn app.

### Muốn chạy nhanh trước khi train 9 model

```powershell
python train_all.py --dataset mnist --max-train 500 --max-val 150 --max-test 200 --epochs 1
streamlit run app.py
```

Chỉ app MNIST có 3 model. Muốn so sánh công bằng trên toàn bộ 3 app, chạy `python train_all.py` không truyền giới hạn đặc biệt.

### Muốn tăng độ chính xác (sẽ chạy lâu hơn)

```powershell
python train_all.py --max-train 8000 --max-val 1500 --max-test 2000 --epochs 8
```

Mặc định **mỗi app chỉ chọn tối đa 1.600 Train, 450 Validation và 900 Test** bằng lấy mẫu phân tầng (để From Scratch chạy được trên CPU). Số mẫu gốc là 60.000/10.000 hoặc 27.455/7.172, nhưng không phải tất cả đều dùng cho huấn luyện mặc định. Bảng số liệu tự ghi số mẫu thực tế được dùng. Test chính thức giữ độc lập, không dùng để điều chỉnh tham số.

## 4. Kiến trúc CNN và đánh giá

Mỗi ảnh `[1,28,28]` → `Conv2D(4 filters, kernel 3x3, valid)` → `ReLU` → `MaxPooling2D(2x2)` → `Flatten(676)` → `Dense(K logits)` → `Softmax (inference)`. Cả ba code dùng cùng khởi tạo trọng số, SGD, Cross-Entropy, seed, thứ tự batch và tập train/val/test.

**From Scratch thực sự tính forward + backward thủ công** bằng NumPy, không dùng torch/keras để tối ưu. Đối với Keras, đặt `KERAS_BACKEND=torch` nên không cần TensorFlow. Test kiểm tra ba mô hình cho kết quả giống nhau ở thời điểm khởi tạo, gradient tích chập đúng qua finite differences, và tất cả 9 luồng train/save/load/predict chạy được với CSV thử tổng hợp.

Các chỉ số trên dashboard: Accuracy, Macro Precision, Macro Recall, Macro F1, Confusion Matrix, biểu đồ so sánh và learning curves. `python scripts/export_results.py` xuất **kết quả đo thật** dưới dạng Markdown/CSV để chèn vào báo cáo.

## 5. Deploy Streamlit Community Cloud

1. Sau khi huấn luyện trên máy, đưa **source code + các file trong `artifacts/`** lên GitHub. CSV Kaggle khá lớn nên không nên đưa lên repo công khai.
2. Khi đưa lên GitHub, chỉnh `.gitignore` để không loại trừ `artifacts/` nếu muốn ứng dụng dự đoán bằng model đã train; hoặc tải artifact qua kho lưu trữ bên ngoài có kiểm soát.
3. Streamlit Community Cloud → **Create app** → repo GitHub → file chính `app.py` → Deploy.
4. Khi không có CSV trong môi trường deploy, trang chủ và phần dự đoán/so sánh vẫn dùng được nếu đã có `artifacts/`; tab Dataset hiển thị thông báo cần tải CSV, ảnh ví dụ lấy từ CSV cũng sẽ không sẵn sàng. Để hỗ trợ trải nghiệm đầy đủ, đưa kèm một vài ảnh demo mẫu không chứa dữ liệu cá nhân.
5. Kiểm tra CPU/RAM và timeout. Thời gian huấn luyện Scratch tăng rõ nếu dùng toàn bộ 60.000 ảnh; **không train mô hình trên cloud app mỗi lần mở trang**.

**Giới hạn:** Bài toán Sign Language MNIST chỉ nhận biết ký hiệu tĩnh trong bộ dữ liệu, không phải hệ thống dịch ngôn ngữ ký hiệu thời gian thực. Kết quả chỉ dành cho học tập và demo.

## 6. Thực nghiệm và tính trung thực số liệu

- Thư mục ZIP **không chứa CSV Kaggle gốc**, vì dữ liệu cần được tải thông qua `download_kaggle.py` trên máy có Internet.
- **Không ghi sẵn các số Accuracy/F1 bịa đặt.** Chúng chỉ xuất hiện sau khi huấn luyện trên CSV thật và được lưu trong JSON.
- `tests/test_project.py` dùng ảnh tổng hợp tạm để kiểm tra phần mềm, **không** đánh giá độ chính xác thực nghiệm của Kaggle.
- Khi chạy 9 model xong, chạy `python scripts/export_results.py` và đưa file `reports/KET_QUA_CAN_DAN_VAO_CHUONG_3.md` vào bảng kết quả cuối báo cáo.
