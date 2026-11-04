# Kết quả kiểm tra — 03/11/2026

## Cập nhật giao diện GlucoCare

- Thiết kế lại màu sắc, kiểu chữ, bố cục hai cột, xem trước ảnh, bảng diễn giải, trạng thái phiên và kết quả mô hình; không thay đổi backend hoặc ngưỡng y khoa.
- Đã chạy lại `tests.check_ui`: đạt luồng tạo ảnh, OCR, đối chiếu, phân tích, lưu và phản hồi.
- `tests.check_chat_ui`: đạt hội thoại đầy đủ, lịch sử và tạo cuộc trò chuyện mới.
- `tests.check_floating_chat_ui`: đạt mở/đóng khung chat nổi, gửi tin nhắn và dùng chung hội thoại với tab Trợ lý AI; dùng database tạm và model fixture.
- Đã mở dashboard và khung trợ lý trên trình duyệt thực để kiểm tra nội dung và bố trí. Nút chat được đặt cố định góc dưới bên phải; hỗ trợ bố cục hẹp.

## Kiểm tra hệ thống trước thay đổi giao diện

Windows, Python 3.13, MySQL cài sẵn trên máy, Ollama 0.40.13, mô hình `qwen3:4b`. Các phiên bản thư viện được chốt trong `requirements.lock.txt`. Không cần Docker.

## Kiểm tra đã chạy

| Kiểm tra | Kết quả | Phạm vi |
|---|---|---|
| `unittest discover -s tests -p "test_*.py" -v` | **33/33 đạt** | Rule, biên chỉ số, dữ liệu thiếu/sai đơn vị, truy xuất, PDF, nguồn trùng, Agent soạn/kiểm tra lại, lịch sử, Q-learning, lỗi mô hình và JSON streaming, train/test |
| `tests.check_ocr` | **3/3 đạt** | Đọc đúng bốn chỉ số từ pixel của ba PNG mẫu và phân loại đúng khoảng theo Rule |
| `tests.check_ui` | **Đạt** | Sinh PNG → chọn ảnh → OCR → xác nhận → phân tích → lưu → phản hồi. Dùng DB SQLite tạm độc lập |
| `tests.check_chat_ui` | **Đạt** | Khung chat, hai lượt hỏi, lịch sử, vote, tạo cuộc trò chuyện mới. Dùng DB tạm và model fixture; không coi là kiểm thử LLM thật |
| `tests.check_chat_live` | **Đạt** | Ollama + MySQL thật: giải thích HbA1c dựa trên ca minh họa, câu hỏi tiếp nối về ăn uống, từ chối tự tăng liều insulin, ghi hội thoại, next_state, Q-value và chặn reward lặp; nhánh bất tỉnh/co giật trả Rule khẩn cấp |
| `tests.check_mysql` | **Đạt** | Kết nối `mutilagent`, kiểm tra đủ 10 bảng |
| Nạp web | **Đạt** | Tải và trích xuất 19.500 ký tự từ trang NIDDK Managing Diabetes. Kiểm thử đơn vị từ chối host/port/scheme ngoài danh sách cho phép |
| Nạp PDF | **Đạt** | Trích văn bản từ PDF thực tạo bởi pypdf, đưa vào kho tri thức test |
| `pip check`, compileall | **Đạt** | Không có phụ thuộc hỏng hoặc lỗi cú pháp |

Không đánh giá OCR trên phiếu xét nghiệm thực tế; ba PNG là ảnh minh họa do phần mềm tạo. Các hội thoại live được đặt tiêu đề `[KIỂM THỬ]`, không phải bệnh nhân thật.

## MySQL thực tế

Các bảng: `runs`, `feedback`, `training_samples`, `dataset_catalog`, `model_runs`, `kb_documents`, `kb_chunks`, `conversations`, `chat_turns`, `rl_qvalues`.

Đã lưu 9 tài liệu/đoạn kiến thức NIDDK ban đầu. Nạp lại tài liệu giống nguồn/nội dung không tạo bản trùng. Dữ liệu NHANES và metadata nguồn đã nhập MySQL; pipeline huấn luyện tải dữ liệu từ MySQL, không đi vòng qua CSV để fit.

Vấn đề cấu hình ban đầu đã xử lý: dùng `MYSQL_*` thay URL mẫu khi có `MYSQL_USER`, mật khẩu có ký tự đặc biệt được truyền qua `URL.create`, database `mutilagent` đã tạo và kết nối thành công. Không thay đổi mật khẩu MySQL. Cột nội dung KB được nâng thành MEDIUMTEXT để lưu tài liệu tiếng Việt dài.

## Deep Learning với dữ liệu công khai thật

Nguồn CDC NHANES: DEMO, BMX, GLU, GHB, DIQ; manifest từng XPT có URL và SHA-256 ở `data/nhanes_2015.metadata.json`, `data/nhanes_2017.metadata.json`.

- Train: **2.456** người lớn của chu kỳ 2015–2016.
- Test: **2.388** người lớn của chu kỳ 2017–2018 độc lập, gồm 405 nhãn dương và 1.983 nhãn âm.
- Kiểm tra không trùng participant_id giữa hai tập. Các người khác nhau có cùng bốn chỉ số vẫn được giữ lại; không ép nhãn lâm sàng thành một hàm tất định của bốn chỉ số.
- MLP 4→32→16→1; scaler fit trên train, cân bằng trọng số lớp khi fit; tập test ngoài không dùng để dừng sớm hoặc chọn ngưỡng. Với dữ liệu tùy chỉnh có participant_id, outer holdout chia theo bệnh nhân.

Kết quả phiên bản mô hình cuối, ngưỡng 0.5:

| Metric | Giá trị |
|---|---:|
| Accuracy | 0.9024 |
| Balanced accuracy | 0.8676 |
| ROC-AUC | 0.9365 |
| Sensitivity | 0.8148 |
| Specificity | 0.9203 |
| Confusion matrix (TN, FP / FN, TP) | 1825, 158 / 75, 330 |

Metrics đầy đủ, target, nguồn, khoảng dữ liệu train và các giới hạn ở `artifacts/metrics.json`, đồng thời được lưu vào `model_runs`. Mô hình hiện tại có `synthetic=false`.

**Đích dự đoán là DIQ010 tự báo cáo từng được nhân viên y tế thông báo mắc đái tháo đường**, không phải chẩn đoán mới từ xét nghiệm. Người điều trị có thể có chỉ số bình thường. Đây là đánh giá theo nhãn của tập dữ liệu; không phải hiệu quả chẩn đoán lâm sàng, không phải xác suất bệnh đã hiệu chuẩn, không dùng để ước lượng tỷ lệ bệnh trong dân số. Dữ liệu Mỹ complete-case, chưa kiểm định cho Việt Nam. [Codebook DIQ010 của CDC](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DIQ_J.htm).

## Chat, Agent và RL

Hội thoại thật đã đi qua Intake → Rule/Prediction khi có chỉ số → Retrieval → Advisor dùng Qwen → Review. Câu trả lời có nguồn phù hợp về ngưỡng xét nghiệm, ăn uống và giới hạn dùng thuốc. Bài kiểm thử soạn lại dùng draft có nguồn sai để xác minh vòng phản hồi Reviewer → Advisor; bài lỗi mô hình xác minh không lưu phản hồi giả thành công.

Đã khắc phục hai lỗi khi thử mô hình thật: template Qwen luôn mở phần suy nghĩ và JSON có thể tiếp tục sinh sau đối tượng đã hoàn chỉnh. Adapter Qwen dùng tiền tố kết thúc thinking, JSON Schema và kết thúc stream sau khi nhận đủ đối tượng hợp lệ. Chưa đánh giá tốc độ trên nhiều cấu hình máy; thời gian đáp ứng phụ thuộc tài nguyên cục bộ.

Q-learning được kiểm tra với chuyển trạng thái, công thức cập nhật, bootstrap và reward idempotent; cũng đã cập nhật Q-table từ phản hồi trong MySQL thật. Mới có phản hồi kiểm thử, **chưa chứng minh chính sách học tốt hơn một kiểu trả lời cố định trên người dùng thật**. Chính sách chỉ chọn cách trình bày, không chọn thuốc hoặc đổi ngưỡng xét nghiệm.

ReviewAgent kiểm tra cấu trúc, chỉ số dẫn nguồn và một số mẫu nội dung rủi ro; không bảo đảm mọi khẳng định của LLM đều được kiểm chứng ngữ nghĩa. Ứng dụng là hệ thống nghiên cứu/đồ án chạy localhost, chưa phải sản phẩm y tế hoặc dịch vụ nhiều người dùng.
