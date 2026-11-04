# CHƯƠNG 2. CÁC KỸ THUẬT MACHINE LEARNING CƠ BẢN

## 2.1. Giới thiệu chung

Machine Learning (học máy, viết tắt là ML) là một nhánh của trí tuệ nhân tạo, nghiên cứu phương pháp xây dựng chương trình có thể học các quy luật từ dữ liệu. Trong cách lập trình truyền thống, người phát triển phải thiết kế rõ các quy tắc xử lý từng tình huống. Với học máy, người phát triển lựa chọn dữ liệu, cách biểu diễn và thuật toán; hệ thống sử dụng quá trình huấn luyện để điều chỉnh tham số, từ đó tạo dự đoán cho những mẫu chưa xuất hiện trong quá trình học. Học máy đặc biệt hữu ích khi số lượng tình huống lớn và việc viết quy tắc thủ công gặp khó khăn.

Chẳng hạn, để phân loại một bông hoa, ta có thể đo chiều dài và chiều rộng của đài hoa, cánh hoa rồi để mô hình tìm mối liên hệ giữa bốn đặc trưng đó với loài hoa. Trong dự đoán giá bất động sản, dữ liệu đầu vào có thể gồm diện tích, vị trí, số phòng; kết quả cần dự đoán là một giá trị liên tục. Hai ví dụ cho thấy bài toán ML có thể khác nhau về dạng đầu ra, dù đều bắt đầu từ dữ liệu và một mục tiêu dự đoán xác định.

Chương này trình bày các nhóm phương pháp ML cơ bản, quy trình xử lý dữ liệu và đánh giá mô hình. Phần thực nghiệm lựa chọn ba bộ CSV công khai được tải từ Kaggle. Trên cùng một bài toán Logistic Regression, chương xây dựng ba chương trình bằng NumPy từ đầu (From Scratch), Keras 3 và PyTorch. Việc giữ nguyên thuật toán, dữ liệu và cách chia tập giúp so sánh *cách cài đặt* thay vì so sánh các thuật toán có độ phức tạp khác nhau. Cuối chương là một project Streamlit gồm ba trang ứng dụng độc lập tương ứng với Iris, Red Wine Quality và Breast Cancer. Mỗi trang hỗ trợ nhập mẫu, dự đoán CSV, xem mã nguồn, đánh giá và đối chiếu ba cách cài đặt.

## 2.2. Các hình thức học máy phổ biến

### 2.2.1. Học có giám sát

Học có giám sát (Supervised Learning) sử dụng các ví dụ đã có nhãn. Mỗi mẫu gồm vector đặc trưng đầu vào $x_i$ và nhãn đầu ra $y_i$. Từ những cặp dữ liệu $(x_i,y_i)$, mô hình học hàm ánh xạ $f(x)$ để khi nhận một đầu vào mới có thể dự đoán nhãn hoặc giá trị tương ứng. Mức độ chính xác không chỉ phụ thuộc thuật toán mà còn chịu ảnh hưởng từ chất lượng nhãn, số lượng mẫu và cách chia dữ liệu.

Hai bài toán tiêu biểu là **phân loại** và **hồi quy**. Phân loại dự đoán một nhãn rời rạc, ví dụ ba loài hoa Iris hoặc hai nhóm của dữ liệu Breast Cancer Wisconsin. Hồi quy dự đoán một đại lượng liên tục như giá nhà, lượng điện tiêu thụ hay thời gian vận chuyển. Một số thuật toán phân loại phổ biến gồm Logistic Regression, k-Nearest Neighbors (KNN), cây quyết định, Random Forest, Naive Bayes và Support Vector Machine (SVM). Trong hồi quy, Linear Regression, Ridge Regression và các mô hình dựa trên cây được dùng nhiều.

### 2.2.2. Học không giám sát

Khác với học có giám sát, học không giám sát (Unsupervised Learning) tìm cấu trúc trong tập dữ liệu không có nhãn mục tiêu. Bài toán thường gặp là phân cụm, chẳng hạn chia khách hàng thành các nhóm có thói quen mua sắm tương đồng. Thuật toán K-Means chọn số cụm $K$, xác định tâm cụm và lặp việc gán điểm dữ liệu vào tâm gần nhất rồi cập nhật tâm. Kết quả phân cụm cần được giải thích theo ngữ cảnh vì số cụm phù hợp không phải lúc nào cũng xác định trước.

Một hướng khác là giảm chiều dữ liệu. Principal Component Analysis (PCA) tạo các trục mới nhằm giữ lại phần lớn phương sai của dữ liệu, giúp trực quan hóa hoặc giảm số đặc trưng. Cần lưu ý PCA không tự động lựa chọn những biến có ý nghĩa nhất đối với nhãn dự đoán; nếu thực hiện PCA trước khi chia dữ liệu, thông tin từ tập kiểm tra có thể ảnh hưởng đến quá trình học.

### 2.2.3. Học tăng cường và những hình thức kết hợp

Học tăng cường (Reinforcement Learning) nghiên cứu cách một tác tử lựa chọn hành động trong môi trường nhằm tối đa hóa phần thưởng tích lũy. Thay vì cung cấp đáp án đúng cho từng mẫu như học có giám sát, môi trường trả về tín hiệu thưởng/phạt sau các hành động. Bài toán điều khiển robot, trò chơi hoặc tối ưu hành động theo chuỗi thời gian thường sử dụng cách tiếp cận này.

Ngoài ba nhóm trên, học bán giám sát kết hợp lượng nhỏ dữ liệu có nhãn với lượng lớn dữ liệu chưa gán nhãn; tự giám sát tạo tín hiệu học từ cấu trúc sẵn có của dữ liệu. Những hình thức này đặc biệt quan trọng khi việc thu thập nhãn thủ công tốn kém, nhưng nằm ngoài phạm vi thực nghiệm của chương.

## 2.3. Một số thuật toán ML cơ bản

### 2.3.1. Hồi quy tuyến tính

Linear Regression mô hình hóa quan hệ gần tuyến tính giữa biến đầu vào và đại lượng cần dự đoán. Với $d$ đặc trưng, một mô hình tuyến tính có dạng $\hat{y}=w^Tx+b$, trong đó $w$ là vector trọng số và $b$ là độ lệch. Mô hình có thể được huấn luyện bằng cách giảm trung bình bình phương sai số, thường gọi là Mean Squared Error (MSE). Ưu điểm của hồi quy tuyến tính là dễ hiểu, tốc độ huấn luyện nhanh và đóng vai trò đường chuẩn trong nhiều bài toán hồi quy. Hạn chế là khó mô tả mối quan hệ phi tuyến phức tạp nếu không mở rộng đặc trưng.

### 2.3.2. Logistic Regression

Dù có từ “Regression” trong tên, Logistic Regression thường được dùng cho phân loại. Trong phân loại hai lớp, mô hình biến tổ hợp tuyến tính của đặc trưng thành xác suất bằng hàm sigmoid. Trong phân loại nhiều lớp, hàm softmax chuyển vector điểm số thành xác suất của từng lớp. Kết quả dự đoán là lớp có xác suất lớn nhất. Mô hình đơn giản, dễ triển khai và phù hợp làm chuẩn so sánh trước khi dùng những mạng nơ-ron phức tạp.

Với dữ liệu đầu vào $X\in\mathbb{R}^{N\times d}$, trọng số $W\in\mathbb{R}^{d\times C}$ và bias $b\in\mathbb{R}^C$, điểm số là $Z=XW+b$. Xác suất lớp $c$ của mẫu $i$ được xác định bởi softmax: $p_{ic}=\exp(z_{ic})/\sum_{j=1}^{C}\exp(z_{ij})$. Hàm mất mát cross-entropy được tính từ xác suất của lớp đúng: $L=-\frac{1}{N}\sum_{i=1}^{N}\log p_{i,y_i}$. Khi tối ưu bằng gradient descent, trọng số và bias được điều chỉnh theo hướng làm giảm mất mát.

### 2.3.3. KNN, cây quyết định và Random Forest

KNN dự đoán nhãn dựa vào $K$ mẫu gần nhất trong dữ liệu huấn luyện. Thuật toán đơn giản, dễ thử nghiệm nhưng nhạy với thang đo đặc trưng, lựa chọn $K$ và lượng dữ liệu cần lưu. Cây quyết định phân chia dữ liệu nhiều lần theo các ngưỡng trên đặc trưng, tạo các quy tắc gần với cách suy luận “nếu – thì”. Ưu điểm là dễ trực quan hóa, song một cây quá sâu dễ ghi nhớ dữ liệu huấn luyện và dự đoán kém trên dữ liệu mới.

Random Forest khắc phục một phần nhược điểm của cây đơn lẻ bằng cách huấn luyện nhiều cây trên các tập mẫu và tập đặc trưng được chọn ngẫu nhiên rồi tổng hợp dự đoán. Phương pháp này thường cho kết quả ổn định trên dữ liệu dạng bảng và ít phụ thuộc vào việc chuẩn hóa thang đo hơn so với KNN hay Logistic Regression. Đổi lại, số lượng cây lớn làm tăng thời gian huấn luyện, dung lượng lưu trữ và khó giải thích từng quyết định riêng lẻ.

### 2.3.4. SVM và Naive Bayes

Support Vector Machine (SVM) tìm siêu phẳng phân tách các lớp với biên phân loại lớn. Khi dùng kernel, SVM còn có thể biểu diễn đường biên phi tuyến. Kết quả của SVM phụ thuộc nhiều vào việc chọn kernel, tham số điều chỉnh và chuẩn hóa dữ liệu. Naive Bayes dựa trên định lý Bayes và giả định độc lập có điều kiện giữa các đặc trưng. Giả định này thường không hoàn toàn đúng trong thực tế, nhưng phương pháp vẫn hữu ích trên một số bài toán văn bản và dữ liệu có nhiều chiều.

Có thể thấy không có thuật toán nào tốt nhất trong mọi trường hợp. Vì vậy, việc xác định đúng bài toán, đánh giá trên dữ liệu chưa dùng huấn luyện và so sánh với đường chuẩn là quan trọng hơn việc lựa chọn mô hình chỉ dựa trên mức độ phổ biến. Để tập trung vào quy trình và cách lập trình, phần thực nghiệm tiếp theo chọn Logistic Regression đa lớp làm mô hình chung cho ba cách cài đặt.

## 2.4. Quy trình xây dựng mô hình Machine Learning

Một quy trình ML thường bắt đầu bằng việc xác định mục tiêu: dự đoán đại lượng gì, dữ liệu được thu thập ở đâu và dự đoán sẽ hỗ trợ quyết định nào. Sau đó, dữ liệu được kiểm tra về kích thước, kiểu thuộc tính, nhãn, bản ghi trùng, giá trị thiếu và phân bố các lớp. Nếu bỏ qua bước khảo sát, mô hình có thể đạt chỉ số tốt trên dữ liệu thử nghiệm nhưng hoạt động sai khi triển khai thực tế.

Với dữ liệu dạng bảng, các bước tiền xử lý thường gồm loại bỏ bản ghi lỗi, xử lý giá trị thiếu, mã hóa đặc trưng phân loại thành số và chuẩn hóa đặc trưng có thang đo khác nhau. **Điểm quan trọng là phải chia dữ liệu trước khi học tham số tiền xử lý**. Ví dụ, trung bình và độ lệch chuẩn của StandardScaler chỉ được tính trên tập huấn luyện, sau đó áp dụng lại cho tập validation và test. Làm ngược lại có thể gây rò rỉ dữ liệu (data leakage), khiến kết quả đánh giá lạc quan không đúng thực tế.

Dữ liệu được tách thành train để cập nhật tham số, validation để lựa chọn cấu hình và test để báo cáo kết quả cuối. Trong thí nghiệm của chương, tỷ lệ mục tiêu là khoảng 70% train, 15% validation và 15% test, có phân tầng theo nhãn (stratified splitting) nhằm giữ phân bố lớp tương đối ổn định. Sử dụng `random_state=42` để dễ tái lập cùng một cách chia.

Quy trình huấn luyện gồm khởi tạo trọng số, tính đầu ra, tính hàm mất mát, xác định gradient và cập nhật tham số. Một vòng qua toàn bộ tập train được gọi là epoch. Sau huấn luyện, mô hình được dùng để dự đoán trên tập test và tính các chỉ số. Cuối cùng, cần lưu cả mô hình lẫn bộ chuẩn hóa và danh sách đặc trưng đầu vào; nếu chỉ lưu trọng số mà quên cách chuẩn hóa thì dự đoán sau triển khai có thể sai lệch đáng kể.

## 2.5. Giới thiệu và khảo sát ba bộ dữ liệu lấy từ Kaggle

Trong báo cáo này, dữ liệu thực nghiệm được lấy từ **ba dataset có trang nguồn trên Kaggle**, không dùng các hàm `load_iris`, `load_wine`, `load_breast_cancer` có sẵn của Scikit-learn. Các file CSV được tải bằng `download_kaggle.py` (sử dụng `kagglehub`), lưu dưới `data/raw/` và chỉ được đọc qua `shared/datasets.py`. Nhờ vậy có thể kiểm tra nguồn gốc, kích cỡ, tên cột và quá trình xử lý đầu vào. Mỗi ứng dụng có dữ liệu riêng, nhưng ba cách cài đặt Logistic Regression trong cùng ứng dụng nhận đúng một cách chia tập và cùng bộ chuẩn hóa. Dữ liệu có tên gọi tương tự nhưng khác phiên bản không được tự động sử dụng thay thế.

### 2.5.1. Bộ dữ liệu Iris Species (App 1)

**Nguồn Kaggle:** https://www.kaggle.com/datasets/uciml/iris. **CSV:** `data/raw/iris/Iris.csv`. Bộ dữ liệu gồm **150 bản ghi**, trong đó mỗi hàng tương ứng một mẫu hoa. File CSV có các trường `Id`, `SepalLengthCm`, `SepalWidthCm`, `PetalLengthCm`, `PetalWidthCm` và `Species`. Thuộc tính `Id` chỉ dùng để định danh nên không đưa vào mô hình; bốn phép đo còn lại là đầu vào. Nhãn `Species` được chuẩn hóa về `setosa`, `versicolor`, `virginica`.

**Bảng 2.1. Một số mẫu Iris từ dữ liệu gốc**

| SepalLengthCm | SepalWidthCm | PetalLengthCm | PetalWidthCm | Species |
|---:|---:|---:|---:|---|
| 5,1 | 3,5 | 1,4 | 0,2 | Iris-setosa |
| 4,9 | 3,0 | 1,4 | 0,2 | Iris-setosa |
| 4,7 | 3,2 | 1,3 | 0,2 | Iris-setosa |

![Hình 2.1. Phân bố nhãn Iris theo nguồn dữ liệu](figures/iris_class_balance.png)

**Phân bố:** mỗi lớp 50 mẫu, tương ứng 33,33%. Nhận xét: phân bố rất cân bằng, thuận lợi để so sánh Accuracy và Macro-F1. Tuy nhiên 150 mẫu là quy mô nhỏ; chỉ vài mẫu dự đoán sai trên tập Test có thể làm thay đổi điểm số đáng kể. Nhóm `setosa` thường dễ phân biệt hơn `versicolor` và `virginica`.

### 2.5.2. Red Wine Quality (App 2)

**Nguồn Kaggle:** https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009. **CSV:** `data/raw/wine/winequality-red.csv`. Đây là **Red Wine Quality** có **1.599 mẫu**, **11 thuộc tính hóa lý** như độ axit cố định, axit bay hơi, lượng đường dư, chloride, mật độ, độ pH, sulphate và nồng độ cồn; cột thứ 12 là `quality` (điểm chất lượng). File CSV được phân tách bằng dấu chấm phẩy (`;`), cần đọc đúng dấu phân cách. **Bộ dữ liệu này khác bộ Wine Recognition 178 mẫu đã sử dụng ở phiên bản báo cáo trước.**

Trong phiên bản app, đầu ra được chuyển thành bài toán phân loại nhị phân: nếu `quality` từ **6 điểm trở lên**, gán lớp **“Từ 6 điểm”**; ngược lại gán lớp **“Dưới 6 điểm”**. Cách quy đổi rõ ràng và giúp người dùng dễ nhập dữ liệu cho giao diện phân loại. Điểm chất lượng gốc vẫn được sử dụng để xác lập nhãn nhưng không đưa vào vector đầu vào, tránh rò rỉ nhãn.

**Bảng 2.2. Một số giá trị mẫu của Red Wine Quality (trích cột)**

| Fixed acidity | Volatile acidity | Citric acid | Alcohol | Quality | Nhãn sau xử lý |
|---:|---:|---:|---:|---:|---|
| 7,4 | 0,70 | 0,00 | 9,4 | 5 | Dưới 6 điểm |
| 7,8 | 0,88 | 0,00 | 9,8 | 5 | Dưới 6 điểm |
| 7,8 | 0,76 | 0,04 | 9,8 | 5 | Dưới 6 điểm |

![Hình 2.2. Phân bố nhãn Red Wine Quality theo ngưỡng 6](figures/wine_class_balance.png)

**Phân bố:** 744 mẫu dưới 6 điểm (46,53%) và 855 mẫu từ 6 điểm (53,47%). Nhận xét: hai nhóm tương đối cân bằng nhưng nhiều đặc trưng hóa lý có đơn vị và thang đo khác nhau, do đó phải chuẩn hóa trên Train. Việc đưa nhiều mức `quality` vào chỉ hai nhóm khiến một số mẫu sát ngưỡng khó phân biệt; Accuracy không nhất thiết quá cao dù code hoàn toàn đúng.

### 2.5.3. Breast Cancer Wisconsin (Diagnostic) (App 3)

**Nguồn Kaggle:** https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data. **CSV:** `data/raw/breast_cancer/data.csv`. Bộ dữ liệu có **569 mẫu**, **30 đặc trưng số** trích xuất từ mô tả nhân tế bào, nhãn gốc ở cột `diagnosis` với hai giá trị `M` (malignant) và `B` (benign). Hai cột `id` và `Unnamed: 32` được loại bỏ vì không mang thông tin đặc trưng phù hợp; nhãn được chuyển thành 0–1. Đây chỉ là bài tập ML, không phải hệ thống chẩn đoán bệnh.

**Bảng 2.3. Một số mẫu đầu từ dữ liệu nghiên cứu (trích cột)**

| radius_mean | texture_mean | perimeter_mean | diagnosis |
|---:|---:|---:|---|
| 17,99 | 10,38 | 122,80 | M |
| 20,57 | 17,77 | 132,90 | M |
| 19,69 | 21,25 | 130,00 | M |

![Hình 2.3. Phân bố hai nhãn của Breast Cancer Wisconsin](figures/breast_cancer_class_balance.png)

**Phân bố:** malignant 212 mẫu (37,26%), benign 357 mẫu (62,74%). Nhận xét: khác với Iris, dữ liệu nghiêng về benign. Vì vậy cần xem thêm Recall của malignant và ma trận nhầm lẫn để phân biệt lỗi bỏ sót với lỗi nhận nhầm; không được chỉ dựa vào Accuracy để nhận xét kết quả. Do tính chất y sinh, ứng dụng luôn hiện cảnh báo chỉ dùng cho học tập.

### 2.5.4. Tổng hợp dữ liệu và khả năng so sánh

**Bảng 2.4. Bảng tổng hợp ba dataset Kaggle**

| Ứng dụng | CSV | Số mẫu | Số đặc trưng | Số lớp | Phân bố |
|---|---|---:|---:|---:|---|
| Iris | Iris.csv | 150 | 4 | 3 | 50 / 50 / 50 |
| Red Wine Quality | winequality-red.csv | 1599 | 11 | 2 | 744 / 855 |
| Breast Cancer | data.csv | 569 | 30 | 2 | 212 / 357 |

Ba dataset có quy mô và đặc trưng khác nhau nhưng đều là bài toán phân loại. Vì vậy có thể dùng chung cách cài đặt Softmax Logistic Regression để đối chiếu ba framework. Lưu ý so sánh công bằng được thực hiện **bên trong từng dataset** (Scratch, Keras, PyTorch chạy cùng tập Test), không lấy Accuracy giữa hai dataset làm thước đo trực tiếp để tuyên bố một bài toán “dễ hơn” bài toán khác.

## 2.6. Thiết kế thực nghiệm và ba cách cài đặt

### 2.6.1. Cấu hình chung

Để so sánh công bằng, cả ba cách lập trình cùng sử dụng mô hình Logistic Regression với hàm softmax, không có tầng ẩn. Tham số được khởi tạo bằng 0, thuật toán tối ưu là Stochastic Gradient Descent ở chế độ **full batch** (mỗi lần cập nhật dùng toàn bộ dữ liệu train), tốc độ học bằng **0,15**, số epoch là **350**, không áp dụng regularization. Các giá trị này được cố định trong toàn bộ thí nghiệm nhằm hạn chế việc mỗi phiên bản chạy với một cấu hình riêng.

Phần tiền xử lý dùng StandardScaler, chỉ `fit` trên dữ liệu train, rồi `transform` validation và test. Cùng những chỉ số test được tính với scikit-learn. Keras 3 được chạy với PyTorch backend vì môi trường kiểm thử không cài TensorFlow; điều này có nghĩa **Keras và PyTorch là hai cách lập trình khác nhau, nhưng bản Keras sử dụng PyTorch làm nền tính toán**, chứ không phải hai động cơ tính toán hoàn toàn độc lập. Cách thiết kế phù hợp để so sánh API, sự thuận tiện và độ đúng của chương trình.

### 2.6.2. From Scratch bằng NumPy

Trong phiên bản From Scratch, chương trình tự viết hàm softmax, hàm tính gradient và vòng lặp cập nhật trọng số bằng NumPy. Với xác suất dự đoán $P$, ma trận one-hot của nhãn $Y$ và $N$ mẫu train, gradient theo điểm số là $G=(P-Y)/N$. Từ đó, gradient theo trọng số là $X^TG$; gradient theo bias là tổng các hàng của $G$. Quy tắc cập nhật là $W\leftarrow W-\eta X^TG$ và $b\leftarrow b-\eta\sum_iG_i$.

**Đoạn mã cốt lõi 2.1. Huấn luyện Logistic Regression từ đầu**

```python
W = np.zeros((X_train.shape[1], n_classes))
b = np.zeros(n_classes)
Y = np.eye(n_classes)[y_train]
for epoch in range(350):
    z = X_train @ W + b
    z -= z.max(axis=1, keepdims=True)
    exp_z = np.exp(z)
    p = exp_z / exp_z.sum(axis=1, keepdims=True)
    G = (p - Y) / len(y_train)
    W -= 0.15 * (X_train.T @ G)
    b -= 0.15 * G.sum(axis=0)
```

Ưu điểm của việc tự cài đặt là người học hiểu rõ ý nghĩa từng bước, từ tính xác suất đến điều chỉnh tham số. Nhược điểm là dễ mắc lỗi số học, đạo hàm hoặc tối ưu hiệu năng nếu mô hình lớn. Vì vậy, From Scratch phù hợp cho việc học thuật toán và kiểm chứng kết quả ở các bài toán nhỏ.

### 2.6.3. Cài đặt bằng Keras 3

Keras cung cấp API mức cao, giúp viết ngắn gọn quá trình khởi tạo mô hình, cấu hình hàm mất mát và huấn luyện. Với Logistic Regression nhiều lớp, chỉ cần một lớp Dense có số đầu ra bằng số lớp. Đầu ra được giữ ở dạng logits và `SparseCategoricalCrossentropy(from_logits=True)` xử lý đúng việc tính cross-entropy.

**Đoạn mã cốt lõi 2.2. Mô hình trong Keras**

```python
model = keras.Sequential([
    keras.layers.Input(shape=(n_features,)),
    keras.layers.Dense(n_classes, activation=None,
                       kernel_initializer="zeros",
                       bias_initializer="zeros")
])
model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.15),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True)
)
model.fit(X_train.astype("float32"), y_train,
          epochs=350, batch_size=len(X_train),
          shuffle=False, verbose=0)
```

So với cách tự lập trình, Keras giúp giảm số dòng mã cần viết, đồng thời thuận tiện khi mở rộng sang mạng nhiều tầng. Tuy nhiên, để phân tích sai số, điều chỉnh các bước cập nhật đặc biệt hoặc kiểm soát chi tiết huấn luyện, người dùng cần hiểu thêm cách Keras tổ chức quá trình tính toán.

### 2.6.4. Cài đặt bằng PyTorch

PyTorch hỗ trợ xây dựng lớp Linear và dùng autograd để tự động tính đạo hàm. Chương trình khai báo optimizer SGD cùng CrossEntropyLoss, sau đó lặp các bước `zero_grad`, `forward`, tính loss, `backward`, `step`. Khác với Keras, vòng lặp huấn luyện được viết trực tiếp nên thuận tiện khi cần tùy biến.

**Đoạn mã cốt lõi 2.3. Vòng lặp huấn luyện PyTorch**

```python
layer = torch.nn.Linear(n_features, n_classes)
torch.nn.init.zeros_(layer.weight)
torch.nn.init.zeros_(layer.bias)
optimizer = torch.optim.SGD(layer.parameters(), lr=0.15)
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(350):
    optimizer.zero_grad()
    logits = layer(X_train_tensor)
    loss = criterion(logits, y_train_tensor)
    loss.backward()
    optimizer.step()
```

PyTorch linh hoạt hơn khi muốn nghiên cứu hoặc thay đổi kiến trúc nhưng đòi hỏi người lập trình quản lý thiết bị, trạng thái huấn luyện và việc lưu tải mô hình. Cả Keras lẫn PyTorch đều có khả năng mở rộng sang các mạng phức tạp; riêng trong chương này, mô hình được giữ ở mức tuyến tính để so sánh đúng một thuật toán.


### 2.6.5. Tách ba ứng dụng thành chín tệp huấn luyện

Để đáp ứng yêu cầu mỗi ứng dụng có code riêng, project được tổ chức thành ba thư mục `projects/iris`, `projects/wine` và `projects/breast_cancer`. Mỗi thư mục chứa bốn tệp gồm `scratch.py`, `keras.py`, `pytorch.py` và `train.py`. Ba tệp đầu triển khai cùng công thức Logistic Regression nhưng bằng ba cách lập trình khác nhau. Tệp `train.py` đảm nhận chạy lần lượt ba phiên bản, lưu mô hình và tính toán chỉ số đánh giá. Việc tách cấu trúc này cho phép người học trình bày trực tiếp mã nguồn của từng phương pháp khi thuyết trình và không phải tìm trong một tệp quá dài.

**Bảng 2.5. Sơ đồ phân chia code theo ba ứng dụng**

| Ứng dụng | Từ đầu (NumPy) | Keras 3 | PyTorch |
|---|---|---|---|
| Iris | `projects/iris/scratch.py` | `projects/iris/keras.py` | `projects/iris/pytorch.py` |
| Wine | `projects/wine/scratch.py` | `projects/wine/keras.py` | `projects/wine/pytorch.py` |
| Breast Cancer | `projects/breast_cancer/scratch.py` | `projects/breast_cancer/keras.py` | `projects/breast_cancer/pytorch.py` |

Ba nhóm code chỉ dùng chung những thành phần thực sự giống nhau, chẳng hạn tải dữ liệu, chia tập, chuẩn hóa, đánh giá và giao diện. Các hàm `train` nằm trong từng tệp riêng nên hoàn toàn có thể mở trực tiếp để theo dõi phương pháp cập nhật trọng số. Đây là cách tổ chức thuận tiện khi làm bài tập lớn, vì tách biệt phần nghiên cứu thuật toán với phần triển khai ứng dụng.

Quy trình tái lập thực nghiệm như sau: chạy `python train_all.py`, chương trình tải lần lượt ba bộ dữ liệu, chia các tập theo seed 42, khởi tạo bộ StandardScaler trên train, gọi ba hàm huấn luyện, tính các chỉ số trên validation và test, rồi lưu ra các tệp mô hình. Sau đó, giao diện Streamlit chỉ cần tải các tệp đã lưu; việc dự đoán không yêu cầu huấn luyện lại mô hình ở mỗi lượt sử dụng.

## 2.7. Đánh giá mô hình và cách so sánh kết quả

### 2.7.1. Tiêu chí đánh giá

Sau khi CSV Kaggle được tải về, `python train_all.py` chạy 9 lượt huấn luyện (3 dataset × 3 cách cài đặt). Các lượt chạy chung một cách chia dữ liệu phân tầng theo nhãn, cố định seed 42 và sử dụng StandardScaler fit trên Train. Từng mô hình có cùng số epoch, tốc độ học và trọng số khởi tạo. Tập Validation được dùng kiểm tra cấu hình; **chỉ tập Test** được dùng để báo cáo kết quả cuối cùng. Điều này hạn chế tình trạng dùng lặp một mẫu để vừa học vừa đánh giá.

**Accuracy** biểu diễn phần trăm dự đoán đúng. **Precision**, **Recall** và **F1-score** được tính theo từng lớp, sau đó lấy trung bình macro để các lớp có trọng số như nhau. **Log-loss** phản ánh mức độ phù hợp của các xác suất dự đoán; trong khi đó **Confusion Matrix** cho thấy chi tiết số mẫu bị nhầm giữa các lớp. Các chỉ số này cần được kết hợp, nhất là với dữ liệu Breast Cancer có phân bố lệch.

### 2.7.2. Thiết lập bảng đánh giá trên ba dataset

Chương trình `shared/train_runner.py` tạo `artifacts/metrics.csv`, các file `*_metrics.json`, checkpoint theo framework và metadata chứa thông tin scaler, nguồn Kaggle, danh sách feature và SHA-256 của CSV. Nhờ lưu các thành phần này, việc đối chiếu có thể tái lập khi cùng dữ liệu và phiên bản thư viện.

**Bảng 2.6. Mẫu bảng dùng để điền kết quả từ file metrics.csv sau khi chạy trên Kaggle**

| Dataset Kaggle | Phiên bản | Accuracy | Macro-F1 | Log-loss | Thời gian |
|---|---|---|---|---|---|
| Iris | From Scratch | Sau khi chạy | Sau khi chạy | Sau khi chạy | Sau khi chạy |
| Iris | Keras | Sau khi chạy | Sau khi chạy | Sau khi chạy | Sau khi chạy |
| Iris | PyTorch | Sau khi chạy | Sau khi chạy | Sau khi chạy | Sau khi chạy |
| Red Wine Quality | From Scratch | Sau khi chạy | Sau khi chạy | Sau khi chạy | Sau khi chạy |
| Red Wine Quality | Keras | Sau khi chạy | Sau khi chạy | Sau khi chạy | Sau khi chạy |
| Red Wine Quality | PyTorch | Sau khi chạy | Sau khi chạy | Sau khi chạy | Sau khi chạy |
| Breast Cancer | From Scratch | Sau khi chạy | Sau khi chạy | Sau khi chạy | Sau khi chạy |
| Breast Cancer | Keras | Sau khi chạy | Sau khi chạy | Sau khi chạy | Sau khi chạy |
| Breast Cancer | PyTorch | Sau khi chạy | Sau khi chạy | Sau khi chạy | Sau khi chạy |

**Lưu ý về số liệu:** Bảng này là **mẫu trình bày**, không phải số liệu đã đo. Tại môi trường soạn báo cáo không truy cập được CSV gốc từ Kaggle nên chưa thể chạy kiểm chứng trên đúng ba nguồn đó. Các Accuracy 91,30% / 100% / 95,35% ở project phiên bản cũ được tính trên bộ Scikit-learn, **không được gán cho project Kaggle mới**. Sau khi chạy thật, dùng `python -m scripts.export_results` để sinh `reports/ket_qua_thuc_nghiem_Kaggle.md`, rồi cập nhật bảng trong báo cáo.

### 2.7.3. Hướng dẫn phân tích và nhận xét kết quả

Trên **Iris**, cần quan sát có nhầm giữa versicolor và virginica hay không; nếu Accuracy giống nhau ở cả ba cách triển khai, cần tiếp tục đối chiếu ma trận nhầm lẫn và xác suất chứ không chỉ nhìn điểm số. Trên **Red Wine Quality**, nhãn đã được gộp từ thang chất lượng nhiều mức, vì vậy những mẫu nằm gần ngưỡng 6 có thể tạo ra nhiều sai sót; cần đọc Precision, Recall, F1 và so sánh với dự đoán đa số lớp. Trên **Breast Cancer**, cần kiểm tra riêng số mẫu malignant bị dự đoán thành benign; kết quả chỉ nhằm minh họa rủi ro của việc đánh giá phân loại, không dùng trong thực hành y tế.

Thời gian chạy là chỉ số bổ sung để thảo luận về overhead của từng thư viện. Với dữ liệu nhỏ, phép tính NumPy có thể rất nhanh, trong khi Keras có thể tốn thời gian khởi tạo; song không thể suy rộng kết luận đó cho mạng sâu hoặc GPU. Keras 3 trong project sử dụng **PyTorch backend**, nghĩa là thí nghiệm so sánh cách viết code hơn là so sánh ba hệ tính toán tách biệt.

### 2.7.4. So sánh định tính ba cách lập trình

**Bảng 2.7. Khác biệt giữa From Scratch, Keras và PyTorch**

| Tiêu chí | From Scratch | Keras 3 | PyTorch |
|---|---|---|---|
| Huấn luyện gradient | Tự viết | Tự động | Tự động |
| Tốc độ viết code | Chậm hơn | Ngắn gọn | Linh hoạt |
| Kiểm soát vòng lặp | Toàn phần | Qua fit() | Toàn phần |
| Mục tiêu phù hợp | Học nguyên lý | Prototyping | Nghiên cứu & tùy biến |
| Checkpoint | .npz | .keras | .pt |

Ba code đều thực hiện cùng một thuật toán Logistic Regression. Nếu kết quả số gần nhau, đó là cơ sở kiểm tra tính nhất quán chứ không phải bằng chứng framework nào tốt hơn. Nếu có sai khác lớn, cần kiểm tra thứ tự lớp, bộ chuẩn hóa, hàm softmax, tốc độ học và kiểu số thực trước khi rút ra kết luận.

### 2.7.5. Kiểm thử tính đúng đắn

Sau khi tải Kaggle và hoàn tất huấn luyện, `python tests/test_project.py` sẽ kiểm tra số mẫu và số đặc trưng, đảm bảo tập Train/Validation/Test không giao nhau, kiểm tra tổng các xác suất của mỗi mẫu xấp xỉ 1 và so khớp các dự đoán giữa ba cách cài đặt. Bộ test cũng đối chiếu lại Accuracy từ checkpoint với giá trị trong file metrics để phát hiện việc lưu hoặc tải nhầm mô hình. **Mức độ hoàn thành của kiểm thử phải được ghi nhận theo đúng lệnh đã chạy**, không xem source code tồn tại là bằng chứng toàn bộ hệ thống đã được kiểm thử trên dữ liệu Kaggle.

## 2.8. Triển khai thử nghiệm ứng dụng (Deploy)

### 2.8.1. Mục tiêu và kiến trúc hệ thống

Phần triển khai nhằm chuyển các tệp mô hình đã huấn luyện thành hệ thống có thể sử dụng bằng giao diện web. Thay vì đặt cả ba bài toán vào một biểu mẫu chung, project được chia thành **ba ứng dụng Streamlit độc lập** và một trang tổng quan để điều hướng. Cách tổ chức này giúp mỗi ứng dụng có giao diện phù hợp với số đặc trưng, ý nghĩa dữ liệu và thông báo riêng, đồng thời người dùng có thể chọn một trong ba cách cài đặt từ đầu, Keras hoặc PyTorch trong cùng trang.

Project có trang khởi đầu `app.py` (ML Lab · 03), kèm ba trang con `pages/01_Iris.py`, `pages/02_Wine.py` và `pages/03_Breast_Cancer.py`. Một trang ứng dụng gồm bốn tab: **Khám phá dataset**, **Dự đoán trực tiếp**, **Đánh giá & so sánh**, **Xem source code**. Người sử dụng không phải hiểu lệnh Python vẫn có thể thử một mẫu dữ liệu và quan sát xác suất từng lớp; người xem báo cáo vẫn có thể kiểm tra mã nguồn và ma trận nhầm lẫn ngay từ giao diện.

**Sơ đồ luồng xử lý:** Người dùng chọn ứng dụng → chọn bộ cài đặt (Scratch/Keras/PyTorch) → nhập đặc trưng hoặc tải CSV → kiểm tra dữ liệu và thứ tự cột → StandardScaler đã học trên tập train → checkpoint tương ứng → xác suất Softmax → nhãn và biểu đồ kết quả.

### 2.8.2. Thiết kế giao diện theo từng ứng dụng

**Ứng dụng Iris** sử dụng bốn phép đo hình thái của hoa. Giao diện hiển thị 150 mẫu, bốn đặc trưng, ba lớp; bảng năm mẫu đầu; biểu đồ phân bố cân bằng; khu vực nhập dữ liệu mẫu; biểu đồ ba xác suất dự đoán; bảng so sánh Accuracy/F1; ma trận nhầm lẫn. Việc phân loại ba lớp cho phép minh họa Softmax một cách trực quan.

**Ứng dụng Wine** làm việc với 11 thuộc tính hóa lý của Red Wine Quality. Việc nhập tay nhiều thông số có thể gây nhầm lẫn, vì thế giao diện cho phép chọn mẫu có sẵn hoặc nạp CSV nhiều dòng; người dùng có thể tải một CSV mẫu đúng tên cột. Phần thống kê thể hiện phân bố 744 mẫu dưới 6 điểm và 855 mẫu từ 6 điểm, đồng thời giải thích lý do cần chuẩn hóa do các chỉ số có thang đo khác nhau.

**Ứng dụng Breast Cancer Wisconsin** chứa 30 đặc trưng và hai lớp. Giao diện sử dụng biểu đồ phân bố và confusion matrix để người học thấy số liệu không cân bằng hoàn toàn, thay vì chỉ đọc Accuracy. Ứng dụng có thông báo rõ về mục đích giảng dạy, không dùng để kết luận chẩn đoán hay thay thế đánh giá chuyên môn.

Giao diện theo phong cách dashboard: nền xanh đậm, khối tiêu đề lớn, ba thẻ chỉ số (số mẫu, đặc trưng, lớp), các tab chức năng, biểu đồ tương tác và nút dự đoán có màu nổi bật. Cấu trúc màu được thống nhất giữa ba trang để toàn bộ project có nhận diện chung nhưng vẫn giữ từng ứng dụng tách biệt về nội dung.

### 2.8.3. Các tệp source và checkpoint

Mỗi nhóm mô hình có một định dạng tệp phù hợp. NumPy From Scratch lưu ma trận trọng số W và bias vào `.npz`; Keras lưu mô hình dưới dạng `.keras`; PyTorch lưu `state_dict` vào `.pt`. Ngoài ra, mỗi dataset có một file metadata `.joblib` chứa StandardScaler, danh sách feature đúng thứ tự và tên các lớp. Số liệu đánh giá được ghi vào `.json` và một bảng tổng hợp `metrics.csv` phục vụ báo cáo.

**Bảng 2.8. Các thành phần chính của project**

| Thành phần | Vai trò |
|---|---|
| `app.py` | Dashboard điều hướng ba ứng dụng |
| `pages/01_Iris.py`, `02_Wine.py`, `03_Breast_Cancer.py` | Ba trang ứng dụng độc lập |
| `projects/<tên_dataset>/{scratch,keras,pytorch}.py` | Chín bản cài đặt thuật toán |
| `shared/datasets.py` | Đọc CSV Kaggle, ánh xạ nhãn, chia tập và chuẩn hóa |
| `shared/train_runner.py` | Chạy thực nghiệm, lưu mô hình, tính chỉ số |
| `shared/ui.py` | Trình bày bảng số liệu, biểu đồ và dự đoán trên từng trang |
| `artifacts/` | Chín checkpoint, scaler, metrics, mẫu dữ liệu (sau huấn luyện) |
| `download_kaggle.py` | Tải ba CSV từ Kaggle vào `data/raw/` |
| `tests/test_project.py` | Tự động kiểm tra độ nhất quán của chín mô hình |

### 2.8.4. Chạy thử trên máy cá nhân

Người thực hiện cần chuẩn bị Python và mở terminal tại thư mục gốc của project. Đầu tiên tạo môi trường ảo nếu cần, sau đó cài thư viện bằng lệnh `pip install -r requirements.txt`. Bước tiếp theo chạy `python download_kaggle.py` để tải ba bộ CSV từ Kaggle. Sau đó thực hiện `python train_all.py` để huấn luyện đồng thời chín phiên bản và lưu kết quả vào thư mục `artifacts`. Có thể huấn luyện từng ứng dụng bằng `python -m projects.iris.train`, `python -m projects.wine.train` hoặc `python -m projects.breast_cancer.train`.

Khi các checkpoint đã được tạo, chạy `streamlit run app.py`, sau đó truy cập địa chỉ `http://localhost:8501` trên trình duyệt. Tại trang tổng quan, chọn ứng dụng cần thử. Trong tab Dự đoán trực tiếp, chọn một trong ba mô hình, chọn mẫu gốc hoặc tải CSV, nhấn nút Dự đoán để xem nhãn và biểu đồ xác suất. Trong tab Đánh giá & so sánh, bảng chỉ số được tạo trực tiếp từ các giá trị tính trên tập test, kèm ma trận nhầm lẫn của mô hình được chọn.

### 2.8.5. Phương án triển khai lên môi trường web

Một phương án phù hợp cho mục đích học tập là **Streamlit Community Cloud**. Người thực hiện đưa mã nguồn, `requirements.txt`, các CSV Kaggle được phép phân phối và checkpoint đã huấn luyện lên một repository GitHub, sau đó kết nối repository với dịch vụ Streamlit Community Cloud. File khởi động được cấu hình là `app.py`. Khi cài thư viện thành công, dịch vụ có thể mở dashboard và các trang con thông qua liên kết truy cập công khai.

Một số vấn đề cần lưu ý khi triển khai là thời gian cài PyTorch tương đối dài, sự tương thích phiên bản Keras/PyTorch, dung lượng bộ nhớ của dịch vụ và việc bảo vệ dữ liệu người dùng gửi lên ứng dụng. Trong phiên bản bài tập này, đầu vào chỉ nên là dữ liệu mẫu công khai hoặc dữ liệu không nhạy cảm. Không nên đưa thông tin bệnh nhân thật hoặc dữ liệu cá nhân vào repository và ứng dụng demo công khai.

Trong môi trường xây dựng hiện tại không thể kết nối Kaggle để tải ba CSV; vì vậy kết quả huấn luyện và giao diện web **chưa được kiểm chứng trực tiếp trên dữ liệu Kaggle**. Source code đã được viết và cần chạy lại chuỗi `download_kaggle.py → train_all.py → tests/test_project.py → app.py` tại máy có mạng trước khi công bố kết quả. Sự phân biệt này cần thiết để báo cáo phản ánh đúng phạm vi thử nghiệm thực tế.

## 2.9. Kết luận chương

Chương 2 trình bày các kiến thức cốt lõi về Machine Learning: học có giám sát, không giám sát, tăng cường, một số thuật toán tiêu biểu và quy trình chuẩn hóa, huấn luyện, đánh giá. Phần thực hành được thiết kế thành **một project gồm ba ứng dụng phân loại độc lập**, sử dụng nguồn dữ liệu công khai trên Kaggle: Iris Species, Red Wine Quality và Breast Cancer Wisconsin (Diagnostic).

Với mỗi dataset, thuật toán Softmax Logistic Regression được cài đặt thành ba mã nguồn riêng theo phương pháp NumPy From Scratch, Keras 3 và PyTorch. Việc sử dụng cùng phép chia tập và cùng tham số tối ưu cho phép kiểm tra tính nhất quán và sự thuận tiện của từng cách lập trình. Giao diện Streamlit được tổ chức theo ba trang, mỗi trang có khu vực xem nguồn dữ liệu, các mẫu, phân bố nhãn, dự đoán CSV, bảng so sánh, ma trận nhầm lẫn và mã nguồn tương ứng.

Cần phân biệt giữa việc hoàn thiện thiết kế và việc xác minh thực nghiệm. Do tại thời điểm soạn báo cáo chưa thể tải CSV trực tiếp từ Kaggle trong môi trường hiện tại, các chỉ số Accuracy, F1 và thời gian được **để trống có chủ đích**. Người thực hiện cần chạy hệ thống trên các CSV Kaggle gốc, lấy kết quả thực từ `artifacts/metrics.csv` và cập nhật Bảng 2.6. Cách trình bày này bảo đảm báo cáo không trộn lẫn các nguồn dữ liệu hoặc tạo ra những kết quả chưa được kiểm chứng.

Sau khi hoàn tất kiểm thử và triển khai, project sẽ minh họa được toàn bộ chuỗi công việc học máy từ đọc dữ liệu Kaggle, phân tích phân bố, tiền xử lý, huấn luyện bằng ba framework cho đến so sánh và phục vụ dự đoán trên giao diện. Đây là tiền đề phù hợp để nghiên cứu sâu hơn CNN, RNN và các kiến trúc trí tuệ nhân tạo ở các chương tiếp theo.

## TÀI LIỆU THAM KHẢO

[1] UCI Machine Learning, *Iris Species*, Kaggle: https://www.kaggle.com/datasets/uciml/iris.

[2] UCI Machine Learning, *Red Wine Quality*, Kaggle: https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009.

[3] UCI Machine Learning, *Breast Cancer Wisconsin (Diagnostic)*, Kaggle: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data.

[4] Scikit-learn Developers, *Preprocessing and model selection*: https://scikit-learn.org/stable/modules/preprocessing.html.

[5] Keras Team, *Keras 3 Documentation*: https://keras.io/keras_3/.

[6] PyTorch Contributors, *PyTorch Documentation*: https://pytorch.org/docs/stable/index.html.

[7] Streamlit Team, *Get Started*: https://docs.streamlit.io/get-started.

[8] Kaggle, *KaggleHub*: https://github.com/Kaggle/kagglehub.
