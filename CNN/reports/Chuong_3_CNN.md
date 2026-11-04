# CHƯƠNG 3. MẠNG NƠ-RON TÍCH CHẬP (CONVOLUTIONAL NEURAL NETWORK – CNN)

## 3.1. Giới thiệu chung

### 3.1.1. Khái niệm và vai trò của CNN

Trong các bài toán xử lý ảnh, dữ liệu không chỉ bao gồm các giá trị số riêng lẻ mà còn có cấu trúc không gian. Những điểm ảnh gần nhau thường có liên hệ, chẳng hạn cùng thuộc một đường biên, một góc hoặc một vùng có kết cấu tương đồng. Nếu chuyển toàn bộ ảnh thành một vector rồi đưa thẳng vào mạng kết nối đầy đủ, mô hình rất khó tận dụng trực tiếp đặc điểm không gian này. Mạng nơ-ron tích chập (Convolutional Neural Network – CNN) được phát triển nhằm xử lý hiệu quả dữ liệu dạng lưới, đặc biệt là hình ảnh, bằng cách sử dụng các bộ lọc có trọng số được chia sẻ trên nhiều vị trí.

CNN là một kiến trúc thuộc Deep Learning, sử dụng các lớp tích chập (Convolution), hàm kích hoạt phi tuyến, lớp giảm kích thước (Pooling) và lớp dự đoán để học đặc trưng từ dữ liệu. Khác với phương pháp trích xuất đặc trưng thủ công, CNN có thể học các bộ lọc thích hợp thông qua quá trình tối ưu hàm mất mát. Ban đầu, mô hình thường học những mẫu đơn giản như cạnh, góc hoặc nét; khi nhiều lớp được xếp chồng, biểu diễn có thể tiến tới những cấu trúc phức tạp hơn. Đây là lý do CNN được sử dụng rộng rãi trong nhận dạng ký tự, phân loại ảnh, phát hiện đối tượng và phân tích hình ảnh công nghiệp [1], [2].

Trong báo cáo này, CNN được nghiên cứu theo hai hướng song song: trình bày lý thuyết về kiến trúc và xây dựng project thực nghiệm. Để phần so sánh rõ ràng, project dùng cùng một mô hình CNN nhỏ được cài đặt bằng ba cách: tự lập trình các phép tính với NumPy (From Scratch), sử dụng Keras và sử dụng PyTorch. Ba bộ dữ liệu đều được tải từ Kaggle, gồm nhận dạng chữ số viết tay, phân loại trang phục và nhận diện cử chỉ tay tĩnh theo bảng chữ cái ASL.

### 3.1.2. Vì sao không dùng mạng Fully Connected trực tiếp?

Một ảnh xám 28×28 chứa 784 giá trị pixel. Nếu nối 784 đầu vào với một lớp 128 nơ-ron, số trọng số riêng của lớp này đã là 784×128, chưa kể bias. Hơn nữa, vị trí của một nét vẽ trong ảnh có thể thay đổi nhẹ mà đối tượng vẫn giữ nguyên ý nghĩa. Cách kết nối đầy đủ không tự thể hiện được nguyên lý một đặc trưng có thể xuất hiện tại nhiều vị trí khác nhau. Trong khi đó, một bộ lọc 3×3 của CNN chỉ cần 9 trọng số cho một kênh đầu vào, và cùng bộ lọc có thể quét trên toàn bộ ảnh.

Nhờ kết nối cục bộ và chia sẻ trọng số, CNN giảm đáng kể số tham số của giai đoạn trích xuất đặc trưng. Cấu trúc này phù hợp với hình ảnh và giúp mô hình học các mẫu hình học hữu ích. Dù vậy, CNN không mặc nhiên tốt hơn trong mọi trường hợp: chất lượng còn phụ thuộc dữ liệu, kích thước ảnh, cấu hình mạng, phương pháp huấn luyện và yêu cầu triển khai.

## 3.2. Kiến trúc và nguyên lý hoạt động

### 3.2.1. Phép tích chập và bộ lọc

Phần quan trọng nhất của CNN là lớp Convolution. Một bộ lọc (kernel) thường có kích thước nhỏ như 3×3 hoặc 5×5; bộ lọc lần lượt trượt qua các vị trí của ảnh và tạo ra một bản đồ đặc trưng (feature map). Về mặt tính toán, các thư viện học sâu thường thực hiện phép tương quan chéo (cross-correlation) nhưng vẫn gọi chung là convolution. Với ảnh đầu vào X và kernel K kích thước 3×3, giá trị tại vị trí (i, j) có thể được viết dưới dạng Y(i,j) = b + ∑ K(u,v)·X(i+u,j+v), trong đó tổng chạy trên các vị trí (u,v) của kernel.

Ở trường hợp stride bằng 1 và không padding (valid), ảnh 28×28 đi qua kernel 3×3 sẽ tạo feature map 26×26. Nếu có 4 bộ lọc, kết quả có 4 kênh đặc trưng. Trọng số của các kernel ban đầu được khởi tạo rồi cập nhật trong quá trình huấn luyện. Các kernel khác nhau có thể học cách phản ứng với những nét hoặc cấu trúc khác nhau; ý nghĩa cụ thể không được gán sẵn mà hình thành từ dữ liệu.

### 3.2.2. Stride, Padding và kích thước đầu ra

Stride là số bước dịch chuyển của kernel sau mỗi phép tính. Stride lớn giúp giảm kích thước bản đồ đặc trưng và chi phí tính toán nhưng có thể làm mất chi tiết. Padding là việc thêm các giá trị xung quanh biên ảnh, thường để kiểm soát kích thước đầu ra hoặc giữ lại thông tin gần mép ảnh. Đối với chiều H của đầu vào, kernel K, padding P và stride S, kích thước đầu ra thường được tính bởi H_out = floor((H + 2P - K)/S) + 1. Công thức tương tự áp dụng cho chiều W.

Trong project, lựa chọn kernel 3×3, stride 1 và padding 0 nhằm giữ công thức đơn giản để người học có thể tự viết phép tính tích chập và đạo hàm. Đầu vào có dạng một kênh 28×28 nên đầu ra của lớp tích chập có kích thước 4×26×26.

### 3.2.3. Hàm kích hoạt ReLU

Nếu chỉ liên tiếp áp dụng các phép biến đổi tuyến tính, mô hình tổng thể vẫn tương đương một phép tuyến tính. Do đó, CNN cần hàm kích hoạt phi tuyến. Trong thực nghiệm, hàm Rectified Linear Unit (ReLU) được sử dụng với công thức f(x) = max(0,x). Các giá trị âm được chuyển thành 0, còn giá trị dương được giữ nguyên.

ReLU có phép tính đơn giản, dễ triển khai và được sử dụng phổ biến. Tuy nhiên, nơ-ron có thể rơi vào trạng thái cho đầu ra bằng 0 trong thời gian dài nếu việc cập nhật tham số không phù hợp. Bản chất của ReLU là một bước biến đổi để tăng khả năng biểu diễn phi tuyến, không phải thao tác chuẩn hóa dữ liệu.

### 3.2.4. Max Pooling và Flatten

Pooling là thao tác tổng hợp thông tin trong một vùng lân cận của feature map. Max Pooling lấy giá trị lớn nhất trong một cửa sổ, còn Average Pooling lấy giá trị trung bình. Với cửa sổ 2×2 và stride 2, feature map 26×26 giảm còn 13×13. Lớp pooling không có các trọng số học như convolution; nhiệm vụ chính là giảm kích thước biểu diễn và hỗ trợ ổn định đối với một số thay đổi nhỏ của đặc trưng.

Sau pooling, các bản đồ đặc trưng được sắp xếp thành vector thông qua Flatten. Trong project, 4 bản đồ 13×13 tạo thành 4×13×13 = 676 giá trị. Vector này được đưa vào lớp Dense để tính điểm số cho từng lớp nhãn. Bước cuối cùng dùng Softmax chuyển điểm số thành các xác suất có tổng bằng 1 trong lúc suy luận.

### 3.2.5. Hàm mất mát và thuật toán tối ưu

Với bài toán phân loại nhiều lớp, hàm mất mát Cross-Entropy được dùng để đo sai lệch giữa phân bố xác suất dự đoán và nhãn thật. Với một mẫu có lớp đúng y và xác suất dự đoán p_y, mất mát có dạng L = -log(p_y). Mất mát trên một batch là trung bình các mẫu trong batch. Mô hình điều chỉnh trọng số để giảm hàm mất mát bằng thuật toán Gradient Descent hoặc biến thể của nó.

Trong phiên bản From Scratch, báo cáo triển khai đạo hàm thủ công qua Softmax, lớp Dense, Max Pooling, ReLU và bộ lọc Convolution. Đối với Keras và PyTorch, cơ chế tự động tính đạo hàm hỗ trợ thực hiện cùng chu trình tối ưu. Việc huấn luyện ba cách trên cùng một kiến trúc, cùng thiết lập SGD và cùng dữ liệu giúp việc đối chiếu công bằng hơn so với sử dụng ba kiến trúc hoàn toàn khác nhau.

## 3.3. Thiết kế kiến trúc CNN cho thực nghiệm

### 3.3.1. Cấu trúc TinyCNN được sử dụng

Để ba phương pháp có thể chạy trên laptop không cần GPU, project sử dụng một CNN nhỏ đặt tên TinyCNN. Kiến trúc gồm Conv2D với 4 bộ lọc 3×3, ReLU, Max Pooling 2×2, Flatten và Dense đầu ra. Số nơ-ron đầu ra thay đổi theo số lớp của dataset: 10 cho MNIST, 10 cho Fashion-MNIST và 24 cho Sign Language MNIST. Các thành phần còn lại giữ nguyên.

Bảng 3.1. Kích thước dữ liệu qua các lớp TinyCNN

| Thành phần | Đầu vào / đầu ra | Chức năng |
|---|---|---|
| Input | 1×28×28 | Ảnh xám đã đưa về [0,1] |
| Conv2D | 4×26×26 | Học 4 bản đồ đặc trưng |
| ReLU | 4×26×26 | Phi tuyến |
| MaxPool2D | 4×13×13 | Giảm độ phân giải |
| Flatten | 676 | Chuyển thành vector |
| Dense | 10 hoặc 24 | Tạo logits cho mỗi lớp |
| Softmax | 10 hoặc 24 | Xác suất khi suy luận |

Lựa chọn mô hình nhỏ giúp minh họa cơ chế CNN bằng cách tự tính đạo hàm trong NumPy. Đây không phải cấu hình CNN tối ưu nhất và không nhằm đạt kết quả cạnh tranh với các kiến trúc sâu như VGG hoặc ResNet. Khi cần nâng cao chất lượng, có thể tăng số bộ lọc, bổ sung lớp tích chập, dùng Batch Normalization, Data Augmentation hoặc lựa chọn optimizer khác.

### 3.3.2. Nguyên tắc so sánh giữa ba cách cài đặt

Một yêu cầu quan trọng của phần thực nghiệm là tránh so sánh thiếu công bằng. Nếu phiên bản Keras sử dụng mạng sâu còn From Scratch chỉ dùng một lớp tuyến tính, sự khác biệt về kết quả chủ yếu phản ánh khác biệt kiến trúc. Vì vậy, project cố định cùng TinyCNN, cùng seed, cùng cách khởi tạo trọng số, cùng batch, cùng epoch và cùng thuật toán SGD. Đầu vào cũng được chuẩn hóa giống nhau bằng cách chia toàn bộ mức xám cho 255.

Trong Keras, dữ liệu được bố trí theo thứ tự chiều cao, chiều rộng, kênh (NHWC); trong NumPy và PyTorch, dữ liệu được bố trí theo kênh, chiều cao, chiều rộng (NCHW). Mã nguồn xử lý khác biệt thứ tự này khi đưa trọng số vào Keras để bảo đảm phép nhân ở lớp Dense tương ứng. Các phương pháp cho đầu ra khớp nhau tại thời điểm khởi tạo là một phép kiểm tra quan trọng về tính đúng đắn, trước khi so sánh quá trình huấn luyện.

## 3.4. Khảo sát ba bộ dữ liệu Kaggle

### 3.4.1. Bộ dữ liệu MNIST – chữ số viết tay

MNIST in CSV được công bố trên Kaggle tại địa chỉ https://www.kaggle.com/datasets/oddrationale/mnist-in-csv. Bộ dữ liệu gồm 60.000 ảnh huấn luyện và 10.000 ảnh kiểm tra. Mỗi ảnh là ảnh xám 28×28, được biểu diễn dưới dạng 784 giá trị pixel từ 0 đến 255. Nhãn là số nguyên từ 0 đến 9, cho biết chữ số viết tay xuất hiện trong ảnh. File train có tên mnist_train.csv, file test có tên mnist_test.csv. Mỗi dòng CSV gồm 785 cột, trong đó cột đầu là label [3].

Bảng 3.2. Minh họa cấu trúc bản ghi MNIST (giá trị minh họa về định dạng, không phải bản ghi thực tế trích xuất)

| Trường | Ví dụ dạng dữ liệu | Ý nghĩa |
|---|---|---|
| label | 5 | Nhãn chữ số |
| pixel0, pixel1, pixel2 | 0, 0, 0 | Các pixel đầu tiên |
| pixel783 | 0 | Pixel cuối cùng |
| Kích thước X | 1×28×28 | Ảnh đầu vào sau reshape |
| Output | 10 xác suất | P(0), ..., P(9) |

Về phân bố nhãn, MNIST gần cân bằng nhưng không bằng nhau hoàn toàn. Trên tập train chuẩn, nhãn 1 có 6.742 mẫu, trong khi nhãn 5 có 5.421 mẫu. Các nhãn khác có số lượng khoảng 5.800 đến 6.300 mẫu, nên bài toán không bị mất cân bằng lớp quá nghiêm trọng [4]. Đặc điểm này thuận lợi cho việc xây dựng mô hình ban đầu, nhưng những chữ số có nét tương đồng như 4–9 hoặc 3–5 vẫn có khả năng bị nhầm lẫn.

Bảng 3.3. Phân bố 10 nhãn MNIST trong tập train

| Nhãn | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Số ảnh | 5.923 | 6.742 | 5.958 | 6.131 | 5.842 | 5.421 | 5.918 | 6.265 | 5.851 | 5.949 |

Nhận xét: MNIST phù hợp để kiểm tra thuật toán CNN vì hình ảnh có cấu trúc đơn giản, số lớp rõ ràng và dữ liệu tương đối sạch. Tuy nhiên, tính dễ của bài toán khiến kết quả trên MNIST không đủ để kết luận một phương pháp cài đặt sẽ tốt trên mọi loại hình ảnh. Vì vậy, project tiếp tục đánh giá trên Fashion-MNIST và Sign Language MNIST, là hai bộ dữ liệu có đặc trưng hình ảnh khác biệt.

### 3.4.2. Bộ dữ liệu Fashion-MNIST – phân loại trang phục

Fashion-MNIST được cung cấp bởi Zalando Research trên Kaggle: https://www.kaggle.com/datasets/zalando-research/fashionmnist. Bộ dữ liệu có 60.000 ảnh train, 10.000 ảnh test, kích thước ảnh xám 28×28 và tổng cộng 10 nhóm thời trang. Các nhãn lần lượt là T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag và Ankle boot [5]. Cấu trúc CSV gồm cột label và 784 cột pixel tương tự MNIST, nên có thể sử dụng lại toàn bộ pipeline tiền xử lý và kiến trúc CNN.

Bảng 3.4. Ví dụ diễn giải một số giá trị output của Fashion-MNIST

| Label | Tên lớp | Ý nghĩa đầu ra |
|---|---|---|
| 0 | T-shirt/top | Áo phông hoặc áo kiểu |
| 1 | Trouser | Quần dài |
| 5 | Sandal | Dép/sandal |
| 7 | Sneaker | Giày thể thao |
| 9 | Ankle boot | Bốt cổ thấp |

Phân bố lớp của bộ train chuẩn cân bằng: mỗi lớp có 6.000 ảnh, tương đương 10%. Đặc điểm này phù hợp để so sánh sự khác biệt giữa các mô hình mà không bị một lớp chiếm ưu thế về số lượng. Tuy nhiên, phân loại trang phục phức tạp hơn phân loại chữ số ở một số trường hợp vì nhiều loại áo có đường nét tương tự và ảnh chỉ có một kênh xám, không chứa thông tin màu sắc.

Nhận xét: Fashion-MNIST hữu ích để kiểm tra khả năng trích xuất đặc trưng của CNN thay vì chỉ dựa vào sự khác biệt rõ của chữ số. Khi phân tích confusion matrix, cần chú ý các lớp có hình dáng gần nhau như Shirt, T-shirt/top, Coat hoặc Pullover. Các lỗi này cho thấy giới hạn của CNN nhỏ và tác động của kích thước ảnh thấp tới khả năng phân biệt loại trang phục.

### 3.4.3. Bộ dữ liệu Sign Language MNIST – ký hiệu tay ASL

Sign Language MNIST có trang Kaggle tại https://www.kaggle.com/datasets/datamunge/sign-language-mnist. Bộ dữ liệu gồm 27.455 ảnh huấn luyện và 7.172 ảnh kiểm tra. Mỗi ảnh xám có kích thước 28×28, thể hiện một ký hiệu tay tương ứng với chữ cái thuộc bảng chữ cái ngôn ngữ ký hiệu Mỹ (ASL). Dataset có 24 lớp, loại trừ chữ J và Z vì hai ký hiệu này cần chuyển động nên khó mô tả đầy đủ bằng một ảnh tĩnh [6].

Dữ liệu được lưu trong sign_mnist_train.csv và sign_mnist_test.csv. Khác với MNIST thông thường, nhãn gốc được mã hóa theo các chữ cái từ A đến Z bằng số 0 đến 25 nhưng không có nhãn 9/J và 25/Z. Khi huấn luyện, chương trình cần ánh xạ tập nhãn thực tế sang dãy chỉ số liên tục 0 đến 23 để phù hợp với hàm Cross-Entropy và số nơ-ron đầu ra. Trong phần hiển thị, nhãn dự đoán lại được chuyển thành chữ cái tương ứng.

Bảng 3.5. Ví dụ cách hiểu output của Sign Language MNIST

| Nhãn gốc | Chữ cái | Nhãn huấn luyện (ví dụ) |
|---|---|---|
| 0 | A | 0 |
| 1 | B | 1 |
| 8 | I | 8 |
| 10 | K | 9 |
| 24 | Y | 23 |

Không nên mặc định phân bố 24 lớp hoàn toàn cân bằng. Giao diện sẽ đọc trực tiếp cột label trong CSV Kaggle để tính số mẫu và tỷ lệ từng lớp, đồng thời xuất bảng phân bố qua script của project. Nhận xét ban đầu là bài toán khó hơn vì số lớp lớn và nhiều cử chỉ tay có hình dạng tương tự. Ngoài ra, bộ dữ liệu được tạo từ các ảnh ký hiệu tay đã qua xử lý và tăng cường dữ liệu, vì vậy chưa đủ để khẳng định hệ thống sẽ nhận dạng tốt ảnh camera trong mọi môi trường.

### 3.4.4. So sánh nguồn dữ liệu và cách hiển thị mẫu

Bảng 3.6. Tổng hợp 3 bộ dữ liệu dùng trong project

| Dataset | Train gốc | Test gốc | Số lớp | Ảnh |
|---|---|---|---|---|
| MNIST | 60.000 | 10.000 | 10 | 28×28 xám |
| Fashion-MNIST | 60.000 | 10.000 | 10 | 28×28 xám |
| Sign Language MNIST | 27.455 | 7.172 | 24 | 28×28 xám |

Để đáp ứng yêu cầu khảo sát dữ liệu thực tế, tab Dataset Kaggle của từng app cung cấp đường dẫn gốc, thống kê kích thước, các ảnh mẫu, bảng output năm bản ghi đầu tiên với một số cột pixel và biểu đồ phân bố lớp. Đây là output được đọc trực tiếp từ CSV đã tải. Trong báo cáo, các bảng minh họa cấu trúc chỉ nhằm giải thích cách đọc dữ liệu; bảng mẫu chính xác và số lượng lớp ASL chi tiết có thể xuất bằng lệnh python scripts/export_results.py sau khi tải dữ liệu.

## 3.5. Tiền xử lý và phân chia dữ liệu

### 3.5.1. Đọc CSV và tạo tensor ảnh

Cả ba bộ dữ liệu đều có nhãn và 784 giá trị pixel trên mỗi dòng. Chương trình dùng pandas để đọc file CSV, tìm đúng cột label, phát hiện các cột có dạng pixel0…pixel783 hoặc pixel1…pixel784 và sắp xếp theo số thứ tự. Sau đó dữ liệu được chuyển thành ma trận float32 với kích thước (N,1,28,28), phù hợp với quy ước NCHW của NumPy và PyTorch. Khi đưa vào Keras, trục kênh được đổi sang cuối để có dạng (N,28,28,1).

Giá trị pixel được đưa về [0,1] bằng phép chia X = pixel/255. Việc chuẩn hóa này giúp phép tính ổn định hơn và bảo đảm các framework nhận cùng dữ liệu đầu vào. Với ảnh tải lên từ giao diện, hệ thống chuyển ảnh thành grayscale, thay đổi kích thước về 28×28 và cho phép đảo màu khi nền ảnh không tương thích với định dạng dữ liệu đã huấn luyện.

### 3.5.2. Train – Validation – Test và giới hạn mẫu

Tập test trong CSV Kaggle chính thức được giữ riêng và chỉ dùng cho đánh giá cuối. Từ tập train chính thức, chương trình tách 15% làm Validation bằng phương pháp phân tầng theo nhãn; phần còn lại dành cho Train. Việc phân tầng giúp tỷ lệ các lớp trong tập tách tương đối gần với phân bố gốc. Tuy nhiên, do mô hình From Scratch tự tính gradient nên huấn luyện trên toàn bộ 60.000 ảnh với CPU có thể tốn nhiều thời gian.

Vì vậy, cấu hình chạy mặc định của project sử dụng tối đa 1.600 mẫu Train, 450 mẫu Validation và 900 mẫu Test cho mỗi app, được chọn bằng lấy mẫu phân tầng và seed cố định. Đây là số mẫu thực tế tham gia các phép tính mặc định, **không phải kích thước bộ dữ liệu Kaggle gốc**. Kết quả do chương trình sinh ra luôn ghi rõ số mẫu đã sử dụng, giúp người đọc tránh hiểu nhầm. Nếu cần chất lượng tốt hơn, có thể tăng giới hạn mẫu, số epoch và thời gian huấn luyện; lúc đó phải chạy lại cả ba phiên bản với cấu hình giống nhau.

Để tránh rò rỉ dữ liệu, các bước chọn mô hình và lựa chọn tham số chỉ dựa trên Train/Validation. Tập Test được dùng để báo cáo chất lượng cuối cùng. Trong một đề tài nghiên cứu sâu hơn, nên thực hiện thêm các lần chạy với nhiều seed để đánh giá mức độ dao động của kết quả.

## 3.6. Cài đặt CNN bằng ba phương pháp

### 3.6.1. From Scratch với NumPy

Phiên bản From Scratch được xây dựng trong shared/cnn_numpy.py. Lớp CNNFromScratch nhận ảnh, tạo các cửa sổ 3×3 bằng sliding_window_view và tính tích chập thông qua phép nhân tensor. Sau ReLU và Max Pooling, dữ liệu được làm phẳng để tính điểm lớp bằng một ma trận trọng số Dense. Hàm Softmax đưa các điểm số về xác suất và Cross-Entropy đo sai số với nhãn thật.

Điểm đáng chú ý là quá trình lan truyền ngược không dựa vào thư viện tự động tính đạo hàm. Đạo hàm được tính lần lượt từ Dense về đầu ra pooling, chuyển qua Max Pooling bằng cách trả gradient cho pixel đạt giá trị lớn nhất trong từng vùng 2×2, sau đó đi qua ReLU bằng mặt nạ giá trị dương. Cuối cùng, gradient của bộ lọc Convolution được tích lũy qua các cửa sổ 3×3. Trọng số được cập nhật bằng SGD với learning rate cố định.

Ưu điểm của phương pháp này là thể hiện rõ nguyên lý toán học của CNN và hỗ trợ kiểm tra từng phép tính. Tuy nhiên, NumPy không có cơ chế tự động lan truyền gradient hay tối ưu GPU như framework học sâu, nên việc mở rộng lên mô hình nhiều lớp sẽ phức tạp và chậm hơn. Project có bài kiểm tra finite-difference so sánh gradient tính thủ công với gradient xấp xỉ số để phát hiện sai sót.

### 3.6.2. Keras

Keras cung cấp giao diện mức cao để định nghĩa các lớp trong mạng nơ-ron. Trong project, Sequential được sử dụng để xây dựng TinyCNN gồm Conv2D, MaxPooling2D, Flatten và Dense. Hàm kích hoạt ReLU được đặt trực tiếp trong lớp Convolution; Cross-Entropy và SGD được khai báo trong bước compile. Quá trình huấn luyện có thể thực hiện theo từng batch với train_on_batch và dữ liệu ở định dạng NHWC.

Project dùng Keras 3 với PyTorch backend, vì vậy người dùng không bắt buộc phải cài TensorFlow. Dù Keras và PyTorch có cách viết khác nhau, cùng backend ở đây giúp giảm sự khác biệt do thư viện tính toán cơ sở. Điều quan trọng là các phép tính và thứ tự sắp xếp trọng số được chuẩn hóa để tương ứng với bản From Scratch.

### 3.6.3. PyTorch

Với PyTorch, kiến trúc được định nghĩa thành lớp TorchCNN kế thừa torch.nn.Module. Các thành phần tương ứng gồm nn.Conv2d, ReLU, nn.MaxPool2d và nn.Linear. Trong mỗi batch huấn luyện, mô hình dự đoán logits, CrossEntropyLoss tính sai lệch, backward tính gradient và SGD cập nhật trọng số. PyTorch cung cấp khả năng tùy chỉnh vòng lặp huấn luyện chi tiết nhưng vẫn hỗ trợ tự động tính đạo hàm.

So với Keras, PyTorch phù hợp khi cần tự kiểm soát từng bước của quá trình học, chẳng hạn bổ sung điều kiện dừng, loss tùy chỉnh hoặc quản lý thủ công việc chuyển thiết bị CPU/GPU. Trong project hiện tại, cấu hình ưu tiên CPU để phù hợp với laptop thông thường và bảo đảm kiểm thử độc lập với card đồ họa.

### 3.6.4. Cấu trúc code 3 app × 3 framework

Mỗi app đặt tại projects/mnist, projects/fashion hoặc projects/sign_language, gồm bốn file: scratch.py, keras.py, pytorch.py và train.py. Từng file framework định nghĩa tên dataset, cách huấn luyện và phương thức dự đoán. Phần toán học hoặc lớp nền tảng dùng chung được đặt trong shared/cnn_numpy.py và shared/backends.py nhằm tránh lặp lại code và bảo đảm ba app sử dụng đúng một kiến trúc CNN. File train_all.py thực hiện huấn luyện 9 phiên bản; app.py và pages/ chịu trách nhiệm điều hướng giao diện.

## 3.7. Đánh giá và so sánh thực nghiệm

### 3.7.1. Tiêu chí đánh giá

Trong bài toán phân loại nhiều lớp, Accuracy cho biết tỷ lệ ảnh được gán đúng nhãn. Precision đo mức độ chính xác của những dự đoán thuộc một lớp, Recall đo khả năng nhận diện đầy đủ các ảnh thật thuộc lớp đó, còn F1-score kết hợp Precision và Recall. Để so sánh giữa các bộ dữ liệu có phân bố lớp khác nhau, chương sử dụng thêm Macro-F1: tính F1 từng lớp rồi lấy trung bình không theo trọng số số mẫu. Confusion matrix giúp xác định cặp nhãn dễ bị nhầm.

Ngoài chất lượng dự đoán, cần xem xét thời gian huấn luyện, khả năng mở rộng và mức độ phức tạp của mã nguồn. Tốc độ không chỉ phụ thuộc thuật toán mà còn phụ thuộc phần cứng, framework và số lượng batch. Do đó, mọi kết luận về hiệu năng phải gắn với cấu hình đo cụ thể; không thể khẳng định mặc định framework nào luôn nhanh nhất.

### 3.7.2. Thiết lập đối chiếu ba phiên bản

Bảng 3.7. Các tham số huấn luyện mặc định

| Tham số | Giá trị |
|---|---|
| Kiến trúc | Conv 4×3×3 → ReLU → MaxPool → Dense |
| Batch size | 64 |
| Epoch | 3 |
| Optimizer | SGD |
| Learning rate | 0,05 |
| Random seed | 42 |
| Tập thực nghiệm | ≤1.600 Train; ≤450 Val; ≤900 Test |
| Metric | Accuracy, Precision, Recall, Macro-F1, CM |

Cùng một giá trị random seed được sử dụng để chọn mẫu và khởi tạo trọng số ban đầu. Ba phương pháp cũng sử dụng cùng thứ tự batch trong từng epoch. Cách thiết kế này giúp hạn chế các yếu tố gây nhiễu và làm cho phần so sánh phản ánh trực tiếp hơn đặc điểm cài đặt. Tuy nhiên, các phép tính số thực vẫn có thể có sai khác rất nhỏ do thứ tự phép toán hoặc môi trường runtime.

### 3.7.3. Bảng kết quả dành cho số liệu chạy trên CSV Kaggle

Bảng 3.8. Bảng so sánh Accuracy và Macro-F1 của ba cách cài đặt trên ba dataset (điền sau khi thực nghiệm)

| Dataset | NumPy Scratch | Keras | PyTorch |
|---|---|---|---|
| MNIST | Chờ chạy | Chờ chạy | Chờ chạy |
| Fashion-MNIST | Chờ chạy | Chờ chạy | Chờ chạy |
| Sign Language MNIST | Chờ chạy | Chờ chạy | Chờ chạy |

Mỗi ô ở bảng trên cần báo cáo cặp Accuracy/Macro-F1, kèm số mẫu Test và số epoch. **Chưa có số đo từ CSV Kaggle thật trong môi trường biên soạn báo cáo, nên không tự điền số liệu**. Sau khi chạy python download_kaggle.py và python train_all.py, chương trình tạo 9 file JSON kết quả riêng trong artifacts. Chạy python scripts/export_results.py sẽ sinh bảng Markdown và CSV gồm số liệu thực tế để chèn vào phần này. Giao diện Streamlit đồng thời biểu diễn bảng, biểu đồ cột so sánh Accuracy/F1 và confusion matrix của từng model.

### 3.7.4. Hướng phân tích kết quả

Nếu ba phiên bản đạt kết quả gần nhau, có thể kết luận rằng việc chuyển cùng kiến trúc sang ba công cụ khác nhau được thực hiện nhất quán. Nếu có sai lệch lớn, cần kiểm tra dữ liệu đầu vào, thứ tự kênh ảnh, thứ tự lớp nhãn, cách khởi tạo trọng số, hàm mất mát và cách cập nhật gradient. Việc kiểm tra ba mô hình có cùng đầu ra ngay khi mới khởi tạo giúp phát hiện lỗi chuyển thứ tự trọng số giữa Keras và PyTorch/NumPy.

Khi đánh giá trên MNIST, có thể kiểm tra những cặp chữ số dễ nhầm. Đối với Fashion-MNIST, cần chú ý nhóm trang phục có hình dạng gần giống nhau. Với Sign Language MNIST, cần xem xét nhầm lẫn giữa các chữ cái tay có hình dáng tương tự, đồng thời bảo đảm không sinh ra nhãn J/Z vốn không tồn tại trong bộ dữ liệu. Các nhận xét về lỗi cụ thể phải dựa trên ma trận nhầm lẫn xuất ra sau khi chạy, không dựa trên phỏng đoán.

## 3.8. Deploy ứng dụng CNN bằng Streamlit

### 3.8.1. Mục tiêu và thiết kế giao diện

Project được xây dựng dưới dạng một hệ thống Streamlit nhiều trang, gồm trang chủ và ba ứng dụng độc lập. Trang chủ hiển thị tổng quan nguồn dữ liệu Kaggle, số lượng mô hình đã huấn luyện và các lối tắt đến từng ứng dụng. Mỗi trang có bốn tab chính: Dataset Kaggle, Dự đoán CNN, Đánh giá & so sánh và Source code. Tông màu xanh đậm, các thẻ thống kê và biểu đồ trực quan giúp dễ trình bày trước giảng viên.

Trong tab Dataset Kaggle, hệ thống hiển thị link nguồn, kích thước, ảnh ví dụ, một số hàng CSV và phân bố lớp. Tab Dự đoán cho phép lựa chọn NumPy From Scratch, Keras hoặc PyTorch, sau đó chọn ảnh từ bộ dữ liệu hoặc tải ảnh PNG/JPG. Hệ thống chuyển ảnh về grayscale 28×28, hiển thị kết quả dự đoán và biểu đồ xác suất. Tab Đánh giá tổng hợp Accuracy, Precision, Recall, Macro-F1, biểu đồ so sánh và confusion matrix. Tab Source code cho phép xem file framework tương ứng cùng phần triển khai lõi.

### 3.8.2. Quy trình hoạt động của ứng dụng

Quy trình suy luận bắt đầu từ dữ liệu người dùng cung cấp, qua bước tiền xử lý, đưa vào CNN đã huấn luyện và chuyển logits thành xác suất. Sau đó giao diện hiển thị nhãn dự đoán cùng các lớp có xác suất cao. Khi chuyển giữa các framework, app sẽ tải file mô hình tương ứng: NPZ cho From Scratch, định dạng .keras cho Keras và .pt cho PyTorch.

Việc lưu mô hình riêng từng framework giúp kiểm tra trực tiếp kết quả của mỗi bản cài đặt, không phải hiển thị cùng một mô hình dưới ba tên khác nhau. Thông tin về lớp nhãn, kích thước train/val/test, cấu hình huấn luyện và các chỉ số đánh giá cũng được lưu trong file JSON để có thể truy vết và lập báo cáo.

### 3.8.3. Hướng dẫn chạy trên máy cá nhân

Bước đầu tiên là giải nén project và mở thư mục bằng VS Code hoặc PyCharm. Tại Terminal, tạo môi trường Python, cài các thư viện trong requirements.txt rồi chạy python download_kaggle.py để tải đúng ba dataset gốc. Sau đó chạy python train_all.py để huấn luyện 9 mô hình và ghi kết quả, chạy python tests/test_project.py để kiểm tra các thành phần phần mềm, cuối cùng chạy streamlit run app.py. Giao diện được truy cập tại địa chỉ http://localhost:8501.

Đối với người chỉ cần thử giao diện trước, có thể chạy riêng một dataset với --dataset mnist --max-train 500 --epochs 1 để tiết kiệm thời gian. Cần lưu ý kết quả của cấu hình chạy nhanh không thể so sánh trực tiếp với kết quả huấn luyện toàn bộ theo cấu hình khác. Mỗi bảng đánh giá đều cần ghi rõ số mẫu thực tế, số epoch và các tham số liên quan.

### 3.8.4. Triển khai trực tuyến và giới hạn

Ứng dụng có thể được đưa lên Streamlit Community Cloud bằng cách lưu source code và các artifact mô hình đã huấn luyện trong một repository GitHub. Người triển khai chọn app.py làm entrypoint và để nền tảng cài các thư viện dựa trên requirements.txt. Đối với các CSV Kaggle có kích thước lớn, nên tránh đưa toàn bộ dữ liệu lên repository chỉ để phục vụ suy luận. Có thể cung cấp vài ảnh ví dụ riêng và giữ các file mô hình đã train, hoặc bố trí kho dữ liệu riêng nếu muốn tab khám phá dữ liệu hoạt động đầy đủ.

Cần phân biệt giữa ứng dụng đã được xây dựng và website thực sự được deploy công khai. Trong phạm vi báo cáo, project có code Streamlit, cấu hình và các bước hướng dẫn; URL công khai chỉ xuất hiện sau khi người thực hiện hoàn thành triển khai trên một nền tảng hosting. Với Sign Language MNIST, mô hình chỉ dự đoán ảnh ký hiệu tay tĩnh và không thể được xem là công cụ dịch ngôn ngữ ký hiệu thực tế từ video.

## 3.9. Đánh giá chung và hướng phát triển

Kết quả đạt được của project nằm ở việc thống nhất ba app theo một quy trình có thể tái lập: lấy dữ liệu Kaggle, kiểm tra cấu trúc CSV, chuẩn hóa ảnh, phân tách train/validation/test, xây dựng CNN cùng kiến trúc theo ba phương pháp, lưu và tải mô hình, hiển thị dự đoán, đánh giá bằng cùng hệ chỉ số. Đây là cách tổ chức có ích trong học tập vì một thay đổi của thuật toán có thể kiểm tra lại trên nhiều bộ dữ liệu mà không cần viết lại toàn bộ chương trình.

Về hạn chế, kiến trúc TinyCNN sử dụng rất ít bộ lọc và chỉ một lớp tích chập. Kích thước đầu vào 28×28 giúp giảm chi phí nhưng cũng làm mất nhiều chi tiết so với ảnh gốc. Việc chỉ dùng một lần chia dữ liệu và các tập con để huấn luyện giúp tiết kiệm thời gian song chưa đủ cơ sở để kết luận về hiệu năng tổng quát. Ngoài ra, ảnh ngoài môi trường Kaggle có thể khác biệt lớn về góc chụp, độ sáng và cách đặt đối tượng, làm giảm chất lượng dự đoán.

Để mở rộng đề tài, có thể bổ sung các lớp tích chập, Dropout và Batch Normalization, thử Adam, đánh giá nhiều seed hoặc xây dựng mô hình với ảnh nhiều kênh màu. Với bài toán ngôn ngữ ký hiệu, bước phát triển tiếp theo cần dữ liệu thời gian thực, chuỗi khung hình và cách xử lý cử chỉ có chuyển động. Các cải tiến này nên được trình bày và đánh giá tách biệt với phiên bản TinyCNN cơ bản để tránh thay đổi nhiều yếu tố cùng lúc.

## 3.10. Kết luận chương

Chương 3 đã trình bày cơ sở lý thuyết của mạng nơ-ron tích chập, gồm phép tích chập, hàm kích hoạt ReLU, Max Pooling, Flatten, Softmax, Cross-Entropy và cách huấn luyện bằng lan truyền ngược. Trên cơ sở đó, báo cáo thiết kế TinyCNN và tổ chức project thành ba ứng dụng tương ứng với MNIST, Fashion-MNIST và Sign Language MNIST, sử dụng đúng các bộ dữ liệu CSV từ Kaggle.

Mỗi ứng dụng có ba phiên bản From Scratch, Keras và PyTorch, cùng hệ thống khảo sát dữ liệu, dự đoán và đánh giá trên giao diện Streamlit. Sự khác biệt giữa các phương pháp chủ yếu nằm ở khả năng biểu diễn, kiểm soát và mở rộng code trong khi kiến trúc CNN được giữ cố định để phục vụ so sánh. Những kiến thức này tạo nền tảng để tiếp tục nghiên cứu RNN và các kiến trúc xử lý dữ liệu tuần tự ở Chương 4.

## TÀI LIỆU THAM KHẢO

[1] LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-Based Learning Applied to Document Recognition. Proceedings of the IEEE, 86(11), 2278–2324. https://doi.org/10.1109/5.726791

[2] Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press. https://www.deeplearningbook.org/

[3] Kaggle. MNIST in CSV. https://www.kaggle.com/datasets/oddrationale/mnist-in-csv

[4] Harvard DS Book. Machine Learning in Practice – MNIST dataset. https://rafalab.dfci.harvard.edu/dsbook/machine-learning-in-practice.html

[5] Kaggle – Zalando Research. Fashion MNIST. https://www.kaggle.com/datasets/zalando-research/fashionmnist

[6] Kaggle – tecperson. Sign Language MNIST. https://www.kaggle.com/datasets/datamunge/sign-language-mnist

[7] Keras. Keras 3 API Documentation. https://keras.io/api/

[8] PyTorch. PyTorch Documentation. https://pytorch.org/docs/stable/index.html

[9] Streamlit. Streamlit Documentation. https://docs.streamlit.io/
