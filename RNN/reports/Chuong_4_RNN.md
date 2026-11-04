# CHƯƠNG 4: MẠNG NƠ-RON HỒI QUY (RNN)

## 4.1. Tổng quan về RNN

Mạng nơ-ron hồi quy (Recurrent Neural Network - RNN) là một nhóm kiến trúc mạng nơ-ron được thiết kế để xử lý dữ liệu có thứ tự. Khác với dữ liệu bảng thông thường, trong dữ liệu chuỗi, vị trí của mỗi phần tử thường mang ý nghĩa. Trong một câu văn, việc đổi thứ tự các từ có thể làm thay đổi hoàn toàn nội dung. Trong chuỗi thời gian, giá trị ở thời điểm hiện tại thường liên quan đến các thời điểm trước đó. Vì vậy, khi xử lý các dữ liệu như văn bản, tín hiệu, âm thanh hoặc dữ liệu theo thời gian, mô hình cần có cơ chế lưu lại thông tin từ những bước đã đi qua.

Ý tưởng cốt lõi của RNN là sử dụng một trạng thái ẩn (hidden state) được truyền từ bước thời gian trước sang bước hiện tại. Ở mỗi bước t, mô hình nhận đầu vào x_t và trạng thái ẩn h_(t-1), sau đó tính trạng thái mới h_t. Nhờ vậy, thông tin của các phần tử trước không bị bỏ hoàn toàn mà được tổng hợp vào một vector trạng thái. Có thể xem hidden state như một bộ nhớ ngắn hạn của mô hình.

Trong chương này, RNN được khảo sát trên ba bài toán xử lý ngôn ngữ tự nhiên sử dụng dữ liệu từ Kaggle: phát hiện tin nhắn SMS spam, phân tích cảm xúc đánh giá phim IMDB và phân loại cảm xúc của câu văn. Mỗi bài toán được xây dựng theo ba cách: tự cài đặt bằng NumPy, sử dụng Keras và sử dụng PyTorch. Việc dùng cùng pipeline dữ liệu và cùng kiến trúc cơ bản giúp so sánh công bằng giữa ba cách triển khai.

## 4.2. Cấu trúc và nguyên lý hoạt động của RNN

### 4.2.1. Trạng thái ẩn

Với một chuỗi đầu vào x_1, x_2, ..., x_T, Simple RNN tính trạng thái ẩn theo công thức:

h_t = tanh(x_t W_x + h_(t-1) W_h + b_h)

Trong đó W_x là ma trận trọng số từ đầu vào đến hidden state, W_h là ma trận hồi quy liên kết hidden state trước với hidden state hiện tại, b_h là bias và tanh là hàm kích hoạt. Hidden state ban đầu thường được khởi tạo bằng vector 0.

Trong bài toán phân loại chuỗi, sau khi xử lý phần tử cuối cùng, hidden state cuối h_T được xem như biểu diễn tóm tắt cho toàn bộ chuỗi. Vector này được đưa qua lớp tuyến tính để tạo logits:

z = h_T W_y + b_y

Sau đó Softmax chuyển logits thành phân bố xác suất trên các lớp. Lớp có xác suất lớn nhất được chọn làm dự đoán cuối cùng.

### 4.2.2. Tại sao RNN phù hợp với văn bản

Một câu không chỉ là tập hợp các từ độc lập. Ví dụ, hai câu chứa nhiều từ giống nhau nhưng khác thứ tự vẫn có thể mang ý nghĩa khác nhau. RNN đọc chuỗi lần lượt và cập nhật hidden state theo từng token, vì vậy trạng thái cuối có thể phản ánh cả nội dung lẫn thứ tự xuất hiện.

Trong project, văn bản không được đưa trực tiếp vào mạng. Các câu trước hết được làm sạch, tách token, ánh xạ token thành số nguyên và đưa về cùng độ dài. Sau đó lớp Embedding chuyển mỗi token ID thành một vector số thực. RNN xử lý lần lượt các vector embedding này.

### 4.2.3. Embedding

Nếu vocabulary có hàng nghìn từ, biểu diễn one-hot sẽ rất thưa và tốn bộ nhớ. Do đó, project sử dụng một ma trận embedding E. Với token ID i, vector đầu vào của RNN được lấy từ hàng E[i]. Embedding được huấn luyện cùng với RNN, vì vậy các từ có vai trò tương tự có thể dần nhận được các biểu diễn phù hợp với nhiệm vụ phân loại.

Hai token đặc biệt được sử dụng là `<PAD>` và `<UNK>`. `<PAD>` có ID 0 và dùng để lấp đầy các câu ngắn hơn độ dài tối đa. `<UNK>` dùng cho các từ không có trong vocabulary. Vocabulary chỉ được xây dựng từ tập train nhằm tránh rò rỉ thông tin từ validation hoặc test.

## 4.3. Huấn luyện RNN bằng Backpropagation Through Time

### 4.3.1. Hàm mất mát

Đối với các bài toán phân loại trong chương này, hàm Cross-Entropy được sử dụng. Với xác suất dự đoán p và nhãn đúng y, loss của một mẫu là:

L = -log p_y

Loss của batch được lấy trung bình trên tất cả mẫu. Mục tiêu huấn luyện là điều chỉnh các trọng số sao cho loss giảm và xác suất dành cho nhãn đúng tăng lên.

### 4.3.2. Backpropagation Through Time

Do hidden state ở thời điểm t phụ thuộc hidden state t-1, đạo hàm cũng phải truyền ngược qua các bước thời gian. Quá trình này được gọi là Backpropagation Through Time (BPTT).

Trong bản From Scratch của project, forward pass lưu lại token ID, embedding, hidden state trước, candidate hidden state và mask tại từng bước. Khi backward, gradient bắt đầu từ lớp phân loại cuối và đi ngược từ thời điểm T về 1. Gradient được cộng dồn cho W_x, W_h, bias, ma trận embedding và lớp đầu ra. Các token `<PAD>` được mask để chúng không làm thay đổi hidden state và không cập nhật embedding ở vị trí 0.

Để giảm nguy cơ gradient bùng nổ, project dùng gradient clipping với chuẩn tối đa bằng 5.0. Ngoài ra, optimizer Adam được tự cài cho bản NumPy để tăng khả năng hội tụ so với Gradient Descent thuần túy.

### 4.3.3. Vanishing gradient và exploding gradient

RNN truyền thống có thể gặp vanishing gradient khi chuỗi dài. Gradient đi qua nhiều phép nhân với ma trận hồi quy và đạo hàm của tanh, khiến giá trị có thể nhỏ dần. Khi đó mô hình khó học quan hệ giữa những phần tử cách xa nhau. Ngược lại, nếu gradient tăng mạnh qua nhiều bước, exploding gradient có thể xuất hiện.

LSTM và GRU được phát triển để xử lý tốt hơn các phụ thuộc dài hạn thông qua cơ chế cổng. Tuy nhiên, chương này chủ động sử dụng Simple RNN nhằm làm rõ cơ chế hồi quy cơ bản và giúp việc cài đặt From Scratch vẫn đủ trực quan. LSTM và GRU được trình bày như các hướng cải tiến của RNN nhưng không dùng làm kiến trúc chính trong phần so sánh ba framework.

## 4.4. Tiền xử lý dữ liệu chuỗi

Pipeline chung của ba ứng dụng là:

Raw text → Clean text → Tokenization → Vocabulary → Token IDs → Padding/Truncation → Embedding → RNN → Class probabilities

Văn bản được chuyển về chữ thường, loại bỏ HTML tag và khoảng trắng thừa. Tokenizer lấy các từ và số theo biểu thức chính quy. Vocabulary mặc định giới hạn ở 4.000 token thường gặp nhất của tập train. Các câu được đưa về tối đa 60 token. Câu ngắn được thêm `<PAD>`, còn câu dài được cắt bớt.

Việc sử dụng một pipeline chung có hai lợi ích. Thứ nhất, ba cách cài đặt Scratch, Keras và PyTorch nhận đúng cùng dữ liệu đầu vào. Thứ hai, khi triển khai Streamlit, người dùng nhập một câu mới và câu này được biến đổi bằng đúng vocabulary đã lưu khi huấn luyện.

## 4.5. Khảo sát ba bộ dữ liệu Kaggle

### 4.5.1. SMS Spam Collection

Nguồn: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

Kaggle mô tả bộ dữ liệu gồm khoảng 5.574 tin nhắn SMS tiếng Anh được gắn nhãn ham hoặc spam. File phổ biến là `spam.csv`, trong đó `v1` chứa nhãn và `v2` chứa nội dung tin nhắn. Một số phiên bản CSV còn có thêm các cột `Unnamed` rỗng; loader trong project tự loại bỏ các cột này.

Đầu ra của ứng dụng gồm hai lớp: Ham và Spam. Dữ liệu thường mất cân bằng, lớp Ham chiếm đa số rõ rệt. Do vậy, ngoài Accuracy, Macro-F1, Precision và Recall cũng được tính để đánh giá công bằng hơn.

Ví dụ định dạng một bản ghi:

| label | message |
|---|---|
| ham | Can we meet after class this afternoon? |
| spam | You have won a free reward. Call now to claim it. |

Sau khi tải dữ liệu, tab Dataset của giao diện hiển thị trực tiếp năm mẫu đầu và biểu đồ phân bố lớp từ CSV thật. Project không hard-code số lượng dòng; số mẫu sử dụng trong báo cáo thực nghiệm được đọc trực tiếp từ file sau khi download.

### 4.5.2. IMDB Dataset of 50K Movie Reviews

Nguồn: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

Bộ dữ liệu IMDB gồm 50.000 đánh giá phim với hai nhãn positive và negative. File `IMDB Dataset.csv` gồm hai cột chính là `review` và `sentiment`. Bộ dữ liệu này lớn hơn đáng kể so với SMS Spam và mỗi văn bản thường dài hơn, vì vậy nó phù hợp để khảo sát khả năng ghi nhớ chuỗi của RNN.

Trong project, file được chia theo phương pháp stratified thành Train, Validation và Test với tỷ lệ 70% - 15% - 15%. Tuy nhiên để thời gian chạy của bản Scratch phù hợp với máy CPU thông thường, cấu hình mặc định chỉ lấy tối đa 4.000 mẫu train, 800 validation và 1.200 test. Việc lấy subset cũng dùng stratify để giữ tỷ lệ nhãn.

Ví dụ định dạng:

| review | sentiment |
|---|---|
| The story was engaging and the acting was excellent... | positive |
| The film was slow and I did not enjoy the characters... | negative |

Bộ IMDB tương đối cân bằng giữa hai nhãn, vì vậy Accuracy có ý nghĩa rõ hơn so với SMS Spam. Tuy nhiên Macro-F1 vẫn được giữ để thống nhất cách đánh giá giữa ba ứng dụng.

### 4.5.3. Emotions Dataset for NLP

Nguồn: https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp

Dataset cảm xúc gồm ba file `train.txt`, `val.txt`, `test.txt`. Mỗi dòng có dạng `text;label`. Các file phổ biến có 16.000 mẫu train, 2.000 validation và 2.000 test. Sáu nhãn gồm sadness, joy, love, anger, fear và surprise.

Khác với hai dataset trước, tập Emotion đã có sẵn validation và test nên project giữ nguyên các split chính thức. Loader dùng `rsplit(';', 1)` thay vì `split(';')`, giúp an toàn hơn nếu nội dung câu có chứa dấu chấm phẩy.

Ví dụ định dạng:

| text | emotion |
|---|---|
| i feel very happy about today | joy |
| i am worried about what might happen | fear |

Đây là bài toán khó hơn vì có sáu lớp và phân bố giữa các cảm xúc không hoàn toàn cân bằng. Vì vậy confusion matrix đặc biệt hữu ích để quan sát mô hình thường nhầm những cảm xúc nào với nhau.

### 4.5.4. So sánh ba dataset

| Dataset | Quy mô công bố | Số lớp | Dạng dữ liệu | Mục tiêu |
|---|---:|---:|---|---|
| SMS Spam | khoảng 5,5 nghìn | 2 | SMS ngắn | Ham / Spam |
| IMDB | 50.000 | 2 | Review dài | Negative / Positive |
| Emotion | 20.000 | 6 | Câu ngắn | 6 cảm xúc |

Ba dataset có độ khó và đặc trưng khác nhau. SMS ngắn và từ khóa spam thường rõ ràng. IMDB dài hơn và ngữ nghĩa phức tạp hơn. Emotion có nhiều lớp nên yêu cầu mô hình phân biệt nhiều trạng thái gần nhau. Vì thế, dùng cả ba giúp đánh giá RNN trên nhiều tình huống thay vì chỉ một dataset.

## 4.6. Thiết kế kiến trúc thực nghiệm

Ba cách cài đặt đều dùng kiến trúc tương đương:

Embedding(vocabulary_size, 24) → Simple RNN(hidden=32, tanh) → Dense/Linear(number_of_classes) → Softmax

Độ dài chuỗi mặc định là 60 token. Optimizer là Adam với learning rate 0,002; batch size 64; số epoch là 8. Gradient được clipping ở mức 5,0 đối với NumPy và PyTorch.

Cấu hình train mặc định giới hạn tối đa 4.000 mẫu train, 800 validation và 1.200 test. Mục tiêu của giới hạn này là giữ thời gian chạy From Scratch ở mức chấp nhận được. Người dùng có thể tăng các giới hạn bằng tham số dòng lệnh sau khi kiểm tra project hoạt động ổn định.

## 4.7. Cài đặt RNN From Scratch bằng NumPy

Phiên bản Scratch tự cài đặt các thành phần chính, không gọi lớp RNN có sẵn. Tham số gồm ma trận embedding E, W_x, W_h, b_h, W_y và b_y.

Forward pass tại mỗi bước lấy embedding theo token ID, tính candidate hidden bằng tanh, sau đó dùng mask để bỏ qua `<PAD>`. Hidden cuối đi qua lớp tuyến tính và Softmax.

Phần backward thực hiện BPTT. Gradient embedding được cộng dồn bằng `np.add.at`, vì cùng một token có thể xuất hiện nhiều lần trong batch. Sau khi tính toàn bộ gradient, chuẩn gradient được clipping rồi Adam cập nhật tham số.

Một điểm quan trọng của project là có finite-difference gradient check. Test thay đổi một phần tử của W_y một lượng epsilon nhỏ, tính sai khác loss bằng phương pháp số và so sánh với gradient từ BPTT. Nếu sai số tương đối vượt ngưỡng, test thất bại trước khi huấn luyện dữ liệu thật. Điều này giúp phát hiện các lỗi đạo hàm khó thấy bằng mắt.

## 4.8. Cài đặt bằng Keras

Keras sử dụng `Embedding(mask_zero=True)`, `SimpleRNN` và `Dense`. Mask do Embedding tạo ra giúp SimpleRNN bỏ qua padding. Optimizer Adam và Sparse Categorical Cross-Entropy được dùng tương ứng với bản Scratch.

Project đặt `KERAS_BACKEND=torch`, nghĩa là Keras 3 chạy trên PyTorch backend. Cách này giảm một dependency lớn và tránh trường hợp máy đã có PyTorch nhưng chưa có TensorFlow. Mặc dù API Keras ngắn gọn, mô hình vẫn thể hiện đúng kiến trúc RNN cần khảo sát.

Ưu điểm của Keras là mã nguồn ngắn, dễ đọc, hỗ trợ fit/validation và lưu mô hình trực tiếp. Nhược điểm là nhiều chi tiết đạo hàm và vòng lặp huấn luyện được framework ẩn đi, vì vậy người học khó quan sát nội bộ hơn so với bản Scratch.

## 4.9. Cài đặt bằng PyTorch

Phiên bản PyTorch sử dụng `nn.Embedding`, `nn.RNNCell` và `nn.Linear`. Thay vì gọi `nn.RNN` cho toàn bộ chuỗi, project lặp `RNNCell` theo từng thời điểm và dùng mask thủ công. Cách này khiến logic gần bản NumPy hơn và bảo đảm padding không tiếp tục biến đổi hidden state.

Trong vòng lặp huấn luyện, dữ liệu được chuyển thành tensor, tính logits, CrossEntropyLoss, gọi `backward()`, clipping gradient và cập nhật bằng Adam. Mô hình được lưu bằng `state_dict` cùng cấu hình kiến trúc.

PyTorch dài hơn Keras nhưng cho phép kiểm soát rõ hơn từng bước. Đây là lựa chọn phù hợp khi cần nghiên cứu mô hình tùy biến hoặc mở rộng sang LSTM, GRU và Attention sau này.

## 4.10. Đánh giá và so sánh ba cách cài đặt

Các chỉ số được sử dụng gồm Accuracy, Macro-Precision, Macro-Recall, Macro-F1 và Confusion Matrix. Accuracy là tỷ lệ dự đoán đúng tổng thể. Macro-F1 tính F1 riêng cho từng lớp rồi lấy trung bình, vì vậy phù hợp hơn khi lớp mất cân bằng.

Project lưu kết quả của từng mô hình thành JSON trong `artifacts/<dataset>/`. Sau khi đủ ba mô hình, lệnh `python scripts/export_results.py` tạo bảng tổng hợp `reports/rnn_results.csv` và `reports/rnn_results.md`.

Bảng kết quả thực nghiệm được sinh theo cấu trúc:

| Dataset | Method | Accuracy | Macro-F1 | Precision | Recall |
|---|---|---:|---:|---:|---:|
| SMS Spam | Scratch | sinh sau khi train | sinh sau khi train | sinh sau khi train | sinh sau khi train |
| SMS Spam | Keras | sinh sau khi train | sinh sau khi train | sinh sau khi train | sinh sau khi train |
| SMS Spam | PyTorch | sinh sau khi train | sinh sau khi train | sinh sau khi train | sinh sau khi train |
| IMDB | Scratch | sinh sau khi train | sinh sau khi train | sinh sau khi train | sinh sau khi train |
| IMDB | Keras | sinh sau khi train | sinh sau khi train | sinh sau khi train | sinh sau khi train |
| IMDB | PyTorch | sinh sau khi train | sinh sau khi train | sinh sau khi train | sinh sau khi train |
| Emotion | Scratch | sinh sau khi train | sinh sau khi train | sinh sau khi train | sinh sau khi train |
| Emotion | Keras | sinh sau khi train | sinh sau khi train | sinh sau khi train | sinh sau khi train |
| Emotion | PyTorch | sinh sau khi train | sinh sau khi train | sinh sau khi train | sinh sau khi train |

Không đưa Accuracy từ dữ liệu synthetic vào báo cáo như kết quả Kaggle. Synthetic fixture chỉ dùng để kiểm thử code. Số liệu cuối cùng phải được sinh sau khi download và train trên CSV/TXT Kaggle thật.

Về mặt triển khai, ba mô hình có cùng mục tiêu nhưng khác mức độ trừu tượng. Scratch dài nhất và chậm nhất nhưng giải thích rõ cơ chế. Keras ngắn nhất, thích hợp xây dựng nhanh. PyTorch cân bằng giữa khả năng kiểm soát và tiện lợi của framework.

## 4.11. Kiểm thử project

Để hạn chế việc phải sửa sau khi chạy, project được chia hai lớp kiểm thử.

`tests/test_kaggle_data.py` kiểm tra dữ liệu thật sau khi tải. Script xác nhận file tồn tại, schema có thể nhận diện, văn bản không rỗng, nhãn hợp lệ và số lớp đúng. Nếu dataset Kaggle đổi tên cột hoặc file tải không đúng, lỗi xuất hiện trước khi bắt đầu huấn luyện.

`tests/test_project.py` không cần Internet và kiểm tra thuật toán trên synthetic fixture. Nó kiểm tra tokenizer, vocabulary, softmax, khả năng học của NumPy RNN, gradient BPTT bằng finite difference, lưu/tải mô hình Scratch, huấn luyện/lưu/tải Keras và PyTorch. Do đó phần logic chính được xác nhận độc lập với dữ liệu Kaggle.

Ngoài ra, `scripts/doctor.py` kiểm tra các thư viện quan trọng và trạng thái dữ liệu trước khi người dùng chạy toàn bộ project.

## 4.12. Deploy bằng Streamlit

Giao diện sử dụng Streamlit và được tổ chức thành ba trang ứng dụng. Mỗi trang có bốn tab.

Tab Dataset Kaggle hiển thị link nguồn, tổng số mẫu, số lớp, một số bản ghi đầu và biểu đồ phân bố lớp. Dữ liệu được đọc trực tiếp từ file người dùng đã tải.

Tab Dự đoán RNN cho phép nhập văn bản tự do và lựa chọn Scratch, Keras hoặc PyTorch. Ứng dụng tải vocabulary và mô hình tương ứng, mã hóa câu, tính xác suất và hiển thị nhãn dự đoán cùng biểu đồ xác suất các lớp.

Tab Đánh giá & so sánh hiển thị Accuracy tốt nhất, Macro-F1 tốt nhất, số phiên bản đã chạy, bảng chỉ số và confusion matrix. Nếu người dùng chưa train đủ mô hình, giao diện hiển thị lệnh cần chạy thay vì báo lỗi khó hiểu.

Tab Source code cho phép xem entry point riêng của từng app và phần cài đặt dùng chung. Cách tổ chức này giúp project vừa rõ cấu trúc ba ứng dụng, vừa tránh sao chép toàn bộ thuật toán giống nhau vào chín file.

## 4.13. Hướng dẫn chạy project

Sau khi giải nén project, mở PowerShell tại thư mục gốc, ví dụ `C:\RNN`. Không chạy Streamlit từ thư mục `shared` vì khi đó `app.py` không nằm trong thư mục hiện tại.

Các lệnh chuẩn:

```text
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

Nếu chỉ muốn kiểm tra một dataset trước, có thể chạy:

```text
python train_all.py --dataset sms
```

Hoặc chỉ một framework:

```text
python train_all.py --dataset imdb --method keras
```

Sau khi Streamlit khởi động, giao diện mặc định có thể truy cập tại `http://localhost:8501`.

## 4.14. Nhận xét và hạn chế

Simple RNN có ưu điểm là kiến trúc dễ hiểu và phù hợp để minh họa dữ liệu tuần tự. Tuy nhiên, với câu dài, hidden state duy nhất có thể không giữ được đầy đủ thông tin. Việc giới hạn 60 token cũng khiến một phần nội dung của review IMDB dài bị cắt bỏ.

Vocabulary cố định 4.000 từ giúp giảm kích thước mô hình nhưng tạo token `<UNK>` cho các từ hiếm. Mô hình cũng chưa sử dụng embedding tiền huấn luyện. Đây là các đánh đổi để giữ project có thể chạy trên CPU và để bản Scratch vẫn khả thi.

Trong các hướng phát triển tiếp theo, Simple RNN có thể được thay bằng LSTM hoặc GRU, tăng độ dài chuỗi, dùng bidirectional network, thêm Attention hoặc sử dụng embedding tiền huấn luyện. Tuy nhiên, khi thay đổi kiến trúc, cần tiếp tục giữ nguyên cách chia dữ liệu để so sánh công bằng.

## 4.15. Kết luận chương

Chương này đã trình bày nguyên lý của Recurrent Neural Network, trạng thái ẩn, Embedding, Softmax và Backpropagation Through Time. Ba bộ dữ liệu Kaggle được sử dụng để minh họa các bài toán phân loại chuỗi với độ khó khác nhau.

Project triển khai cùng một Simple RNN bằng ba phương pháp: NumPy From Scratch, Keras và PyTorch. Bản Scratch giúp quan sát rõ forward, BPTT, gradient clipping và Adam. Keras cho phép xây dựng mô hình ngắn gọn, trong khi PyTorch cho khả năng kiểm soát vòng lặp huấn luyện linh hoạt.

Toàn bộ quy trình từ download dữ liệu, kiểm tra schema, tiền xử lý văn bản, huấn luyện, lưu mô hình, đánh giá đến deploy Streamlit được tách thành các module rõ ràng. Kết quả Accuracy và Macro-F1 được sinh trực tiếp từ dữ liệu Kaggle thật sau khi chạy `train_all.py`, tránh sử dụng số liệu giả hoặc số liệu từ synthetic test trong báo cáo.

## Tài liệu tham khảo

[1] UCI Machine Learning & Kaggle, SMS Spam Collection Dataset, https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

[2] Lakshmipathi N, IMDB Dataset of 50K Movie Reviews, https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

[3] Kaggle, Emotions dataset for NLP, https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp

[4] Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors.

[5] Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation.

[6] Cho, K. et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation.
