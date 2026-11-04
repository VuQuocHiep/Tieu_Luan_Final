# GỢI Ý GIẢI THÍCH MÃ NGUỒN CHO MỤC 6.3

Tài liệu này gắn nội dung lý thuyết với **mã nguồn thật** trong thư mục Agent. Các đoạn trích khi đưa vào Word nên được sao chép **liên tiếp từ một vị trí trong một file**, không nối các dòng nằm rời rạc trong mã nguồn.

## 6.3.1. App 1 - Agent hỏi đáp tài liệu

### Mục tiêu

Người dùng đặt câu hỏi về tài liệu PDF/TXT. Ứng dụng trích xuất văn bản, chia đoạn và tìm kiếm những đoạn liên quan bằng BM25. Khi Ollama hoạt động, Agent có thể lựa chọn và gọi công cụ tra cứu trước khi tạo câu trả lời. Kết quả đưa tên tài liệu và số trang để hỗ trợ kiểm chứng.

### Các tầng xử lý của ứng dụng

Tầng 1 - Giao diện: `app1_hoi_dap/app.py` sử dụng `st.file_uploader` nhận tệp và `st.chat_input` nhận câu hỏi.

Tầng 2 - Xử lý tài liệu: `rag_core.py`, hàm `extract_file()` kiểm tra kiểu dữ liệu và trích xuất văn bản từ PDF/TXT.

Tầng 3 - Chia đoạn: `rag_core.py`, hàm `split_text()` chia văn bản thành những đoạn có chồng lấn, còn `build_chunks()` ghi nhớ nguồn, số trang và mã đoạn.

Tầng 4 - Truy xuất: `rag_core.py`, lớp `BM25Index` tính trọng số từ khóa và sắp xếp các đoạn theo điểm phù hợp.

Tầng 5 - Công cụ: `app.py`, hàm `search_documents()` cung cấp khả năng tìm kiếm cho Agent. Cấu trúc `Tool` mô tả tên, ý nghĩa và tham số đầu vào của công cụ.

Tầng 6 - Ra quyết định, gọi công cụ: `common/agent_core.py`, lớp `AgentRunner` gửi danh sách công cụ cho Ollama, tiếp nhận yêu cầu gọi hàm và đưa kết quả công cụ trở lại mô hình. Đây là tầng giúp phân biệt Agent với một lượt gọi LLM đơn thuần.

Tầng 7 - Trình bày kết quả: `app.py` hiển thị câu trả lời, nguồn được đối chiếu và nhật ký sử dụng công cụ.

### Đoạn mã nên trích để phân tích trong báo cáo

Đoạn A: trích nguyên hàm `extract_file()` trong `rag_core.py`. Nêu rõ cách chọn PDF hoặc TXT và giữ số trang. Đoạn B: trích nguyên hàm `split_text()` và giải thích kích thước đoạn, overlap, lý do chia nhỏ tài liệu. Đoạn C: trích phương thức `BM25Index.search()` và giải thích từ khóa, IDF, xếp hạng. Đoạn D: trích đoạn tạo `tools` và gọi `AgentRunner` trong `app.py`, giải thích giao thức giữa LLM và công cụ. Đoạn E: trích vòng lặp `for` trong `AgentRunner.run()` và giải thích quy trình gọi công cụ nhiều bước.

### Điểm cần ghi chính xác

Phiên bản này **không sử dụng Qdrant hoặc embedding**. BM25 là thuật toán truy xuất theo từ khóa. Nếu cần báo cáo RAG với cơ sở dữ liệu vector, phải triển khai thêm mô hình embedding và vector DB thay vì mô tả sai mã nguồn hiện tại.

## 6.3.2. App 2 - Agent quản lý kho

### Mục tiêu

Người dùng yêu cầu kiểm tra tồn kho, lịch sử xuất hàng, dữ liệu cảm biến mẫu hoặc đề xuất số lượng nhập. Agent có thể lựa chọn công cụ đọc dữ liệu từ SQLite và giải thích kết quả. Các thao tác ghi dữ liệu phải do người dùng xác nhận bằng nút riêng trên giao diện.

### Các tầng xử lý của ứng dụng

Tầng 1 - Dữ liệu mẫu: `warehouse_core.py`, hàm `initialize()` khởi tạo bảng SQLite sản phẩm, phiếu xuất, cảm biến và phiếu nháp.

Tầng 2 - Data Access: `connect()` và các câu lệnh SQL lấy dữ liệu, luôn dùng tham số thay vì ghép chuỗi trực tiếp để tránh SQL injection.

Tầng 3 - Nghiệp vụ: `low_stock()` phát hiện sản phẩm đạt hoặc dưới ngưỡng tồn; `sales_summary()` tổng hợp lượng xuất 30 ngày.

Tầng 4 - Phân tích đề xuất: `reorder_suggestions()` sử dụng lượng xuất trung bình trong 30 ngày và mức tồn tối thiểu để gợi ý nhập thêm.

Tầng 5 - Công cụ Agent: `app.py` có các hàm `tool_low_stock`, `tool_sales`, `tool_reorder`, được công bố bằng danh sách `Tool`.

Tầng 6 - Điều phối LLM: `AgentRunner` gọi công cụ phù hợp tùy theo yêu cầu ngôn ngữ tự nhiên; công cụ trả kết quả JSON cho mô hình.

Tầng 7 - Giao diện và quyền hạn: `app.py` hiển thị sản phẩm, hội thoại và nút xác nhận tạo phiếu nháp. Agent không tự động thay đổi số lượng hàng.

### Đoạn mã nên trích

Đoạn A: `initialize()` mô tả bảng và dữ liệu. Đoạn B: `low_stock()` chứng minh điều kiện cảnh báo. Đoạn C: `sales_summary()` cho thấy câu truy vấn tổng hợp lượng xuất. Đoạn D: `reorder_suggestions()` giải thích công thức tính trung bình và mục tiêu tồn. Đoạn E: `tools` và `AgentRunner` trong giao diện cho thấy Agent lựa chọn công cụ. Đoạn F: khối xử lý nút `Xác nhận tạo phiếu nháp` chứng minh cơ chế kiểm soát thao tác ghi.

### Điểm cần ghi chính xác

Đây là **dữ liệu SQLite giả lập**. Chức năng đề xuất nhập hàng là **quy tắc nghiệp vụ**, không phải mô hình Machine Learning đã huấn luyện. Dữ liệu cảm biến được tạo mẫu; chưa có kết nối thời gian thực với ESP32.

## 6.3.3. App 3 - Multi-Agent tạo báo cáo

### Mục tiêu

Hệ thống chia việc tạo báo cáo thành ba vai trò có đầu vào và đầu ra riêng, được quản lý bởi `MultiAgentCoordinator`. Người dùng tải lên CSV hoặc sử dụng dữ liệu mẫu. Kết quả có chỉ số doanh thu, giá vốn, lợi nhuận gộp ước tính, bảng phân loại và báo cáo văn bản.

### Các tầng xử lý

Tầng 1 - Giao diện nhập: `app3_multi_agent/app.py` nhận CSV và lựa chọn chế độ có hoặc không có Ollama.

Tầng 2 - Kiểm tra dữ liệu: `DataAgent.run()` xác nhận đủ cột, xử lý kiểu dữ liệu, bỏ dòng sai và xây dựng tập dữ liệu chuẩn.

Tầng 3 - Phân tích số học: `AnalysisAgent.run()` tính doanh thu, giá vốn, biên lợi nhuận và các nhóm theo sản phẩm/danh mục bằng Python, không giao phép cộng số liệu cho LLM.

Tầng 4 - Phân tích ngôn ngữ: Khi người dùng bật Ollama, `AnalysisAgent` tạo nhận xét bằng một vai trò mô hình được chỉ dẫn riêng.

Tầng 5 - Soạn báo cáo: `ReportAgent.run()` sử dụng dữ liệu đã được thống kê để tạo báo cáo; khi không có LLM sẽ dùng mẫu văn bản cố định.

Tầng 6 - Điều phối Agent: `MultiAgentCoordinator.run()` gọi tuần tự ba Agent, nhận và chuyển dữ liệu cho Agent tiếp theo, đồng thời lưu lại trace.

Tầng 7 - Xuất kết quả: `app.py` hiển thị thống kê, nhật ký ba vai trò và cho phép xuất tệp TXT.

### Đoạn mã nên trích

Đoạn A: trích nguyên `DataAgent.run()` để giải thích làm sạch dữ liệu. Đoạn B: phần tính `revenue`, `cost`, `top_products` trong `AnalysisAgent.run()` nhằm chứng minh cách tính số liệu. Đoạn C: nguyên phương thức `ReportAgent.run()` để giải thích bước tạo báo cáo. Đoạn D: nguyên `MultiAgentCoordinator.run()` để chứng minh luồng phối hợp Agent. Đoạn E: mã nhận nút và gọi `MultiAgentCoordinator` trong `app.py`.

### Điểm cần ghi chính xác

Đây là **mô hình Multi-Agent điều phối tuần tự ba vai trò**. DataAgent dùng quy tắc kiểm tra dữ liệu; hai Agent còn lại có thể sử dụng LLM nếu người dùng bật chế độ Ollama. Không nên viết rằng ba LLM chạy song song hoặc rằng hệ thống tạo ra doanh thu thực tế.

## Gợi ý trình bày mục 6.3 trong Word

Đối với mỗi ứng dụng, trình bày theo thứ tự: mục tiêu ứng dụng, dữ liệu đầu vào, kiến trúc và luồng xử lý, các đoạn code quan trọng, kết quả chạy thử có thể quan sát và nhận xét về giới hạn. Chỉ đưa ảnh chụp giao diện hoặc kết quả thực nghiệm khi đã chạy thành công; không tự ghi kết quả kiểm thử chưa thực hiện.
