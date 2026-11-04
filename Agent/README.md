# AGENT - 3 ỨNG DỤNG MINH HỌA CHO BÁO CÁO HỌC PHẦN

Thư mục này chứa mã nguồn ba ứng dụng Python độc lập, có giao diện Streamlit và hỗ trợ Ollama (mặc định `qwen3:4b`). Có thể chạy các chức năng cốt lõi không cần Ollama để kiểm tra dữ liệu, nhưng chế độ **LLM Agent gọi công cụ** chỉ hoạt động khi Ollama có sẵn.

## 1. Cấu trúc thư mục

```text
Agent/
├── app1_hoi_dap/
│   ├── app.py                  Giao diện hỏi đáp PDF/TXT
│   ├── rag_core.py             Đọc tài liệu, chia đoạn, BM25
│   └── data/huong_dan_kho.txt Tài liệu mẫu
├── app2_quan_ly_kho/
│   ├── app.py                  Giao diện trợ lý kho
│   └── warehouse_core.py      SQLite, công cụ truy vấn, đề xuất nhập
├── app3_multi_agent/
│   ├── app.py                  Giao diện tạo báo cáo
│   └── agents.py               DataAgent, AnalysisAgent, ReportAgent
├── common/
│   ├── agent_core.py           Vòng lặp LLM quyết định và gọi công cụ
│   ├── ollama_client.py        API Ollama dùng chung
│   └── text_utils.py           Xử lý chữ tiếng Việt
├── data/
│   └── sales_sample.csv        Dữ liệu bán hàng giả lập
├── tests/test_core.py          Kiểm thử tự động
├── run_1.bat                   Chạy ứng dụng 1
├── run_2.bat                   Chạy ứng dụng 2
├── run_3.bat                   Chạy ứng dụng 3
└── requirements.txt
```

Cơ sở dữ liệu SQLite `data/warehouse.db` sẽ được tạo tự động trong lần chạy đầu ứng dụng 2, **không cần cài MySQL**.

## 2. Hướng dẫn cài trên Windows

Yêu cầu: Python 3.11 hoặc 3.12, kết nối internet khi cài thư viện. Có thể dùng Python 3.13 nếu những phiên bản thư viện đã cài tương thích.

Mở Command Prompt (CMD) tại thư mục `Agent`:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu chỉ có Python 3.12 thì thay `py -3.11` bằng `py -3.12`. Nếu `py` không tìm thấy, sử dụng `python -m venv .venv`.

### Bật Ollama (khuyến nghị cho phần trình diễn Agent thực sự)

Cài Ollama từ trang chính thức `https://ollama.com/` và chạy trong một CMD riêng:

```bat
ollama pull qwen3:4b
ollama serve
```

Nếu Ollama đang chạy sẵn ở nền, không cần chạy lệnh `ollama serve` lần nữa. Mặc định ứng dụng kết nối đến `http://127.0.0.1:11434`.

Để dùng tên mô hình hoặc địa chỉ khác, trong CMD chạy trước khi mở ứng dụng:

```bat
set OLLAMA_MODEL=qwen3:4b
set OLLAMA_URL=http://127.0.0.1:11434
```

Ứng dụng không cần API key đám mây. Việc chạy LLM cục bộ phụ thuộc vào RAM và tốc độ máy tính.

## 3. Cách chạy từng ứng dụng

Sau khi tạo `.venv`, nhấp đúp các file `run_1.bat`, `run_2.bat`, `run_3.bat`, hoặc trong CMD tại thư mục `Agent`:

```bat
.venv\Scripts\python.exe -m streamlit run app1_hoi_dap/app.py
```

```bat
.venv\Scripts\python.exe -m streamlit run app2_quan_ly_kho/app.py
```

```bat
.venv\Scripts\python.exe -m streamlit run app3_multi_agent/app.py
```

Mỗi ứng dụng mặc định mở ở địa chỉ `http://localhost:8501` (khi chỉ một ứng dụng chạy); nếu có ứng dụng khác đang dùng cổng này, Streamlit sẽ thông báo cổng tiếp theo.

## 4. Ứng dụng 1 - Agent hỏi đáp tài liệu

**Công nghệ:** Streamlit, pypdf, BM25, Ollama.

**Luồng xử lý:** PDF/TXT -> trích văn bản -> chia đoạn -> lập chỉ mục BM25 -> Agent gọi công cụ `search_documents` -> LLM trả lời, nêu tên tệp và trang. Có thể tải nhiều PDF/TXT, hoặc dùng tệp mẫu để thử ngay.

**Câu hỏi gợi ý:**
- `Khi tồn kho xuống dưới mức tối thiểu phải xử lý như thế nào?`
- `Có thể coi giá trị ADC của MQ-2 là ppm không?`
- `Quy trình nhập kho gồm những bước nào?`

**Lưu ý:** Truy xuất sử dụng thuật toán **BM25 theo từ khóa**, không phải embedding hay vector database. Đây là RAG đơn giản, không tự huấn luyện mô hình. PDF scan chỉ có hình ảnh có thể không đọc được vì không tích hợp OCR. Khi thiếu Ollama, ứng dụng chỉ hiển thị các đoạn tìm thấy và **không giả vờ đã tổng hợp câu trả lời bằng LLM**.

**Các đoạn code quan trọng để giải thích trong báo cáo:**
- `rag_core.py`: `extract_file()`, `split_text()`, `BM25Index.search()`.
- `app.py`: `search_documents()`, danh sách `tools`, `AgentRunner.run()`.
- `common/agent_core.py`: cơ chế nhận tool call, gọi hàm thật và đưa kết quả trở lại LLM.

## 5. Ứng dụng 2 - Agent hỗ trợ quản lý kho

**Công nghệ:** Streamlit, SQLite, Ollama.

Dữ liệu giả lập gồm sản phẩm, lịch sử xuất hàng và một lần ghi cảm biến. App tự khởi tạo database trong lần chạy đầu. Agent có các công cụ chỉ đọc: `low_stock`, `find_product`, `sensors`, `sales_summary`, `reorder_suggestions`.

**Câu hỏi gợi ý:**
- `Sản phẩm nào tồn thấp?`
- `Phân tích lượng xuất trong 30 ngày.`
- `Đề xuất số lượng nhập hàng để đủ tồn kho.`
- `Nhiệt độ và độ ẩm kho gần nhất là bao nhiêu?`

**Lưu ý:** Đề xuất nhập hàng dùng quy tắc từ lượng xuất trung bình trong 30 ngày, không phải AI dự báo đã huấn luyện. Chỉ số `gas_adc` là ADC thô, không phải ppm. Không có cảm biến ESP32 thật được kết nối. Agent không được sửa kho. Muốn tạo phiếu nhập **nháp** phải bấm nút xác nhận tại sidebar; thao tác đó không đặt mua hàng và không làm tăng tồn kho.

**Các đoạn code quan trọng:**
- `warehouse_core.py`: `initialize()`, `low_stock()`, `sales_summary()`, `reorder_suggestions()`.
- `app.py`: định nghĩa danh sách công cụ và `offline_answer()`.
- `common/agent_core.py`: vòng lặp gọi công cụ.

**Tích hợp vào hệ thống kho thật:** Thay các hàm truy vấn SQLite bằng HTTP API có phân quyền của backend. Lấy dữ liệu DHT22, MQ-2 từ API backend sau khi ESP32 gửi dữ liệu. Không truy cập trực tiếp vào database sản xuất và không tự gửi lệnh điều khiển thiết bị khi chưa có xác thực/ủy quyền.

## 6. Ứng dụng 3 - Hệ Multi-Agent tạo báo cáo

**Công nghệ:** Streamlit, CSV, Python thuần và Ollama tùy chọn.

Ứng dụng dùng **ba vai trò Agent độc lập** với bộ điều phối rõ ràng:

1. `DataAgent`: đọc và kiểm tra CSV, lọc dòng lỗi và bàn giao dữ liệu hợp lệ.
2. `AnalysisAgent`: dùng Python tính doanh thu, giá vốn, lãi gộp ước tính và số lượng; khi bật Ollama thì có thêm bước nhận xét bằng LLM với vai trò riêng.
3. `ReportAgent`: viết báo cáo dựa trên kết quả đã tính; có thể dùng LLM với chỉ dẫn riêng hoặc chế độ mẫu cố định.

`MultiAgentCoordinator` quản lý thứ tự và truyền dữ liệu giữa ba Agent. Đây là **Multi-Agent theo mô hình phối hợp vai trò**, trong đó Agent thu thập xử lý bằng quy tắc và hai Agent còn lại có thể sử dụng LLM, không phải ba LLM chạy đồng thời.

**CSV đầu vào gồm các cột:**

```text
date,sku,product,category,quantity,unit_price,unit_cost
```

`date` theo `YYYY-MM-DD`. `quantity` là số nguyên dương; giá tính bằng VND. Tệp `data/sales_sample.csv` có 42 dòng dữ liệu giả lập.

**Cách demo:** Mở ứng dụng 3 -> dùng CSV mặc định -> tích chọn Ollama (nếu muốn) -> bấm `Chạy 3 Agent và tạo báo cáo` -> quan sát từng bước, thống kê, nội dung báo cáo và tải tệp TXT.

**Các đoạn code quan trọng:**
- `agents.py`: `DataAgent.run()`, `AnalysisAgent.run()`, `ReportAgent.run()` và `MultiAgentCoordinator.run()`.
- `app.py`: luồng giao diện, hiển thị kết quả, tải báo cáo.

## 7. Kiểm thử

Tại CMD trong thư mục `Agent`:

```bat
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Các phép thử bao gồm: đọc và tìm kiếm văn bản, tra cứu sản phẩm, tạo phiếu nháp sau khi xác nhận, thống kê và phối hợp ba Agent, và giả lập vòng lặp gọi công cụ.

## 8. Giới hạn cần nêu trong báo cáo

- Tất cả dữ liệu kho và doanh thu trong gói này là **dữ liệu giả lập**, không phải dữ liệu của hệ thống IoT thực tế.
- Agent 1 tìm kiếm bằng BM25 chứ chưa dùng embedding/vector database.
- Agent 2 không tự điều khiển ESP32 và không thay thế dự báo nhu cầu bằng mô hình ML.
- Agent 3 không có ba mô hình LLM chạy song song; đây là ba vai trò chuyên trách với cơ chế điều phối tuần tự.
- Chế độ không có Ollama chỉ là cách **trình diễn quy trình và tính toán**, không tương đương với năng lực suy luận của LLM Agent.
- Cần xác thực tài khoản, phân quyền, bảo vệ dữ liệu và đánh giá thực nghiệm trước khi đưa lên máy chủ công khai.
