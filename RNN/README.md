# Chapter 4 - RNN Lab: 3 Kaggle Apps x 3 Implementations

Project minh họa **Recurrent Neural Network (RNN)** trên ba bài toán NLP. Mỗi ứng dụng có ba cách cài đặt độc lập: **From Scratch (NumPy)**, **Keras 3 (PyTorch backend)** và **PyTorch**. Cả ba dùng cùng pipeline tiền xử lý, cùng tập train/validation/test và cùng kiến trúc cơ bản: `Embedding -> Simple RNN (tanh) -> Dense/Linear -> Softmax`.

## 1. Ba dataset Kaggle

1. **SMS Spam Collection** - `uciml/sms-spam-collection-dataset`
   - Link: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
   - File: `spam.csv`
   - Bài toán: Ham / Spam.

2. **IMDB Dataset of 50K Movie Reviews** - `lakshmi25npathi/imdb-dataset-of-50k-movie-reviews`
   - Link: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
   - File: `IMDB Dataset.csv`
   - Bài toán: Negative / Positive.

3. **Emotions dataset for NLP** - `praveengovi/emotions-dataset-for-nlp`
   - Link: https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp
   - Files: `train.txt`, `val.txt`, `test.txt`
   - Bài toán: sadness / joy / love / anger / fear / surprise.

## 2. Cấu trúc project

```text
RNN_Kaggle_3_Apps/
├── app.py
├── pages/
│   ├── 01_SMS_Spam.py
│   ├── 02_IMDB.py
│   └── 03_Emotion.py
├── projects/
│   ├── sms/
│   │   ├── scratch.py
│   │   ├── keras.py
│   │   ├── pytorch.py
│   │   └── train.py
│   ├── imdb/
│   │   ├── scratch.py
│   │   ├── keras.py
│   │   ├── pytorch.py
│   │   └── train.py
│   └── emotion/
│       ├── scratch.py
│       ├── keras.py
│       ├── pytorch.py
│       └── train.py
├── shared/
│   ├── config.py
│   ├── datasets.py
│   ├── text_utils.py
│   ├── rnn_scratch.py
│   ├── models.py
│   ├── train_runner.py
│   └── ui.py
├── tests/
│   ├── test_kaggle_data.py
│   └── test_project.py
├── scripts/
│   ├── doctor.py
│   ├── export_dataset_report.py
│   └── export_results.py
├── data/raw/
├── artifacts/
├── reports/
├── download_kaggle.py
├── train_all.py
└── requirements.txt
```

## 3. Cách chạy chuẩn trên Windows PowerShell

Luôn đứng ở **thư mục gốc project**, ví dụ `C:\RNN` chứ không đứng trong `shared` hoặc `tests`.

```powershell
cd C:\RNN
pip install -r requirements.txt
python scripts/doctor.py
python download_kaggle.py
python tests/test_kaggle_data.py
python tests/test_project.py
python train_all.py
python scripts/export_results.py
python scripts/export_dataset_report.py
streamlit run app.py
```

Mở `http://localhost:8501`.

## 4. Cấu hình train mặc định

- Train tối đa: 4.000 mẫu / dataset
- Validation tối đa: 800
- Test tối đa: 1.200
- Vocabulary tối đa: 4.000 từ
- Độ dài chuỗi: 60 token
- Embedding: 24 chiều
- Hidden state: 32 chiều
- Epoch: 8
- Batch size: 64
- Optimizer: Adam, learning rate 0.002
- Gradient clipping: 5.0 ở NumPy/PyTorch

Cấu hình này được chọn để mô hình thực sự có thời gian học, nhưng vẫn có thể chạy trên CPU. **From Scratch sẽ chậm nhất.**

Có thể train riêng một app hoặc framework:

```powershell
python train_all.py --dataset sms
python train_all.py --dataset imdb --method keras
python train_all.py --dataset emotion --method pytorch --epochs 12
```

## 5. Kiểm thử đã có

`python tests/test_project.py` kiểm tra:
- tokenizer/vocabulary/sequence;
- forward + softmax của NumPy RNN;
- BPTT bằng finite-difference gradient check;
- train/save/reload/infer của Scratch;
- train/save/reload/infer của Keras;
- train/save/reload/infer của PyTorch.

`python tests/test_kaggle_data.py` chỉ chạy sau khi tải dữ liệu và sẽ phát hiện schema sai trước khi train.

## 6. Giao diện Streamlit

Mỗi app có 4 tab:
- **Dataset Kaggle**: kích thước, mẫu dữ liệu, phân bố lớp.
- **Dự đoán RNN**: nhập văn bản và chọn Scratch/Keras/PyTorch.
- **Đánh giá & so sánh**: Accuracy, Macro-F1, Precision, Recall và confusion matrix.
- **Source code**: xem entry point và cài đặt framework.

## 7. Kết quả thực nghiệm

Sau khi train dữ liệu Kaggle thật:

```powershell
python scripts/export_results.py
```

sẽ sinh:
- `reports/rnn_results.csv`
- `reports/rnn_results.md`

Không dùng số liệu synthetic để báo cáo như kết quả Kaggle.
