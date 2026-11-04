# Hệ đa Agent cho bệnh tiểu đường

Ứng dụng chạy cục bộ, triển khai các khối trong sơ đồ yêu cầu: **MySQL → Deep Learning → dự đoán/phân loại**, **web/sách → KB → RAG ↔ Agent**, **Rule + Reinforcement Learning → tư vấn và trợ giúp**.

Đầu vào hình ảnh ở dự án này là **ảnh bảng chỉ số xét nghiệm**. Phần mềm sinh PNG, OCR từ pixel, yêu cầu người dùng đối chiếu rồi phân tích chỉ số. Không phải mô hình đọc ảnh võng mạc hoặc chẩn đoán qua ảnh cơ thể.

## Chạy trên máy hiện tại — không cần Docker

MySQL và Ollama cần đang chạy. Workspace đã cài thư viện và có mô hình `qwen3:4b` trên máy này.

```cmd
cd /d D:\MultiAgent_Diabetes_System
.venv\Scripts\python.exe -m diabetes.cli init-db
.venv\Scripts\python.exe -m streamlit run app.py
```

Mở http://127.0.0.1:8501. Khi cập nhật mã hoặc `.env`, dừng CMD bằng Ctrl+C rồi chạy lại.

Giao diện GlucoCare có bố cục hai cột, thanh hướng dẫn theo bước và phần kết quả tách biệt giữa Rule và mô hình. **Nút Trợ lý AI cố định ở góc dưới bên phải** mở khung chat ngay trên trang hiện tại. Khung nổi và tab Trợ lý AI dùng chung hội thoại; phần “Ngữ cảnh & cuộc trò chuyện” trong khung nổi cho chọn báo cáo và tạo cuộc trò chuyện mới. Bấm lại nút hoặc bấm ra ngoài để thu khung chat.

`.env` cấu hình bằng các dòng riêng; mật khẩu để nguyên, kể cả `@`, không cần URL-encode:

```env
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_DATABASE=mutilagent
MYSQL_USER=YOUR_MYSQL_USER
MYSQL_PASSWORD=YOUR_MYSQL_PASSWORD
OLLAMA_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen3:4b
```

Khi có `MYSQL_USER`, các biến `MYSQL_*` được ưu tiên hơn `DATABASE_URL`. Nếu không có `MYSQL_USER`, có thể dùng URL SQLAlchemy. Không ghi mật khẩu thực vào `.env.example` hoặc đưa `.env` vào Git. Khi chuyển máy: tạo `.venv`, cài `requirements.txt`, cấu hình MySQL, cài Ollama và `ollama pull qwen3:4b`. `requirements.lock.txt` ghi phiên bản đã kiểm tra trên Windows/Python 3.13.

## Các màn hình

1. **Sinh ảnh bảng:** nhập glucose đói, HbA1c, tuổi, BMI; tạo và tải PNG có nhãn minh họa.
2. **Phân tích ảnh:** upload PNG/JPG hoặc dùng ảnh vừa tạo → OCR → đối chiếu giá trị/đơn vị → xác nhận nhịn ăn → phân tích với Rule và mạng nơ-ron. Lưu báo cáo MySQL, xuất JSON.
3. **Trợ lý AI:** chat nhiều lượt bằng Ollama thật. Chọn báo cáo xét nghiệm làm ngữ cảnh hoặc hỏi kiến thức chung. Có lịch sử MySQL, trích nguồn, luồng Agent, tạo/mở lại cuộc trò chuyện và phản hồi từng câu trả lời. Khi Ollama không hoạt động, ứng dụng báo lỗi thay vì giả lập câu trả lời AI.
4. **Kho tri thức:** xem nguồn, nạp PDF có văn bản/TXT/Markdown hoặc xem trước rồi nạp web NIDDK/CDC/ADA/WHO. Chia đoạn có chồng lấn, khử trùng và lưu văn bản/nguồn vào MySQL. PDF chỉ có ảnh phải OCR trước; tối đa 100 trang/10 MB, 60.000 ký tự mỗi tài liệu.
5. **Huấn luyện & RL:** nhập CSV có nhãn vào MySQL, chọn tập train và tập kiểm tra độc lập, huấn luyện MLP, xem confusion matrix, metrics, nguồn và Q-table.
6. **Kiến trúc:** sơ đồ các thành phần và luồng phối hợp.
7. **Lịch sử MySQL:** mở báo cáo phân tích đã lưu.

## Đối chiếu sơ đồ của thầy

| Khối | Cách triển khai | Mã nguồn |
|---|---|---|
| DB | MySQL `mutilagent`: dữ liệu huấn luyện, catalog nguồn, metrics, KB và chunks, báo cáo, hội thoại, reward, Q-table | `diabetes/storage.py` |
| Deep Learning | MLP hai lớp ẩn, kiến trúc 4→32→16→1; StandardScaler fit trên tập train; trọng số cân bằng lớp | `diabetes/learning.py` |
| Dự đoán / phân loại | Điểm mô hình và nhãn nhị phân ngưỡng 0.5, hiển thị riêng với phân loại theo ngưỡng xét nghiệm | `PredictionAgent`, `RuleAgent`, `app.py` |
| Web / sách → KB | Nạp HTML/PDF/TXT/MD, giữ tên nguồn, chunk và hash khử trùng; có 9 đoạn kiến thức ban đầu từ NIDDK | `diabetes/knowledge.py`, `knowledge/` |
| RAG | Kết hợp truy xuất TF-IDF word/character, cấp bằng chứng cùng ngữ cảnh cho LLM Ollama sinh trả lời có trích dẫn | `RetrievalAgent`, `AdvisorAgent`, `diabetes/llm.py` |
| Agent | Intake, Safety, Rule, Prediction, Retrieval, Advisor, Review được Coordinator điều phối; nếu draft không đạt kiểm tra thì gửi phản hồi soạn lại, tối đa hai vòng | `diabetes/chat.py`, `diabetes/agents.py` |
| Rule | Kiểm tra chỉ số, đơn vị, nhịn ăn và phạm vi người lớn không mang thai; nhánh tình huống khẩn cấp; chặn liều thuốc và nguồn không hợp lệ trong draft | `diabetes/rules.py`, `SafetyAgent`, `ReviewAgent` |
| Reinforcement Learning | Q-learning theo chủ đề và khoảng xét nghiệm; action là cách trình bày; chuyển trạng thái từ lượt hỏi sau, reward thực từ đánh giá người dùng; Q-table trong MySQL | `QLearningPolicy`, `Database.reward_turn` |
| Tư vấn / trợ giúp | Chat AI có bộ nhớ cuộc hội thoại và nguồn, sử dụng kết quả Rule/ML làm ngữ cảnh; không tự kê đơn | `diabetes/ui.py` |

Các Agent có trách nhiệm và giao thức rõ ràng; không cần mỗi Agent là một LLM hoặc tiến trình riêng để điều phối. Review hiện kiểm tra cấu trúc, chỉ số trích dẫn và một số mẫu câu nguy hiểm; **không phải bộ kiểm chứng ngữ nghĩa y khoa hoàn hảo**. Bản tham khảo ngắn dưới báo cáo vẫn dùng mẫu; phần hỏi đáp sinh bằng LLM nằm ở tab Trợ lý AI.

## Dữ liệu thật và mô hình hiện tại

Mô hình đang dùng dữ liệu công khai **CDC NHANES**, thay bản mô phỏng ban đầu:

- Train: chu kỳ **2015–2016**. Test: chu kỳ **2017–2018** riêng biệt; không dùng test để chọn ngưỡng hoặc dừng sớm.
- Ghép bằng `SEQN` từ DEMO, BMX, GLU, GHB, DIQ. Dùng người lớn có dữ liệu đủ, trọng số phân mẫu nhịn ăn dương; loại trường hợp ghi nhận mang thai.
- Bốn đầu vào: `LBXGLU` glucose huyết tương đói mg/dL, `LBXGH` HbA1c %, `RIDAGEYR` tuổi, `BMXBMI` BMI.
- **Nhãn:** DIQ010 tự báo cáo đã từng được nhân viên y tế thông báo mắc đái tháo đường. Chỉ dùng 1=yes và 2=no; loại borderline/unknown. Đây **không phải nhãn chẩn đoán mới từ xét nghiệm**. Người đã điều trị có thể có xét nghiệm bình thường.
- Raw XPT, CSV và manifest chứa URL/hash nằm trong `data/`; bản dữ liệu phục vụ train được lưu trong MySQL. Mô hình lưu `artifacts/model.pkl`, metrics ở `artifacts/metrics.json` và `model_runs` trong MySQL.

Tái lập luồng tải CDC → MySQL → train → test:

```cmd
.venv\Scripts\python.exe -m diabetes.cli prepare-nhanes
```

Lệnh này nạp lại hai dataset tên `nhanes_2015`, `nhanes_2017` và thay mô hình đang dùng sau khi train thành công. Dữ liệu Mỹ, complete-case, không dùng thiết kế mẫu để ước lượng tỷ lệ dân số; chưa kiểm định lâm sàng ở Việt Nam. Các metrics đo khả năng dự đoán **nhãn dataset**, không chứng minh hiệu quả chẩn đoán thực tế.

Nguồn: [CDC NHANES 2017–2018](https://wwwn.cdc.gov/nchs/nhanes/search/DataPage.aspx?Cycle=2017-2018), [DIQ010 codebook](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DIQ_J.htm), [glucose codebook](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/GLU_J.htm).

## Dữ liệu của bạn

CSV cần `fasting_glucose,hba1c,age,bmi,outcome`; glucose mg/dL, HbA1c %, tuổi năm, BMI kg/m², outcome 0/1 có nguồn rõ ràng. Nếu có `participant_id`, giữ lại để kiểm tra trùng bệnh nhân giữa train/test. Cần ít nhất 10 mẫu khác nhau mỗi lớp. Không tự gán nhãn bệnh từ một công thức rồi coi là dữ liệu thật.

```cmd
.venv\Scripts\python.exe -m diabetes.cli import-db data/clinical.csv --dataset clinical
.venv\Scripts\python.exe -m diabetes.cli train --dataset clinical
```

CLI `import-db` cùng tên sẽ thay dataset đó trong một transaction. Giao diện yêu cầu tên mới để giữ dataset cũ. Giao diện cho nhập mô tả nhãn và chỉ định dữ liệu mô phỏng. Dữ liệu giả lập vẫn có lệnh `demo-data` / `train --csv ... --synthetic` để kiểm tra pipeline, không dùng để báo cáo hiệu quả y khoa.

## Q-learning hoạt động như thế nào?

State = `chủ đề câu hỏi : khoảng xét nghiệm`; action = `concise`, `step_by_step`, `detailed`. ε-greedy với ε=0.1, α=0.2, γ=0.8:

`Q(s,a) ← Q(s,a) + α × [reward + γ × max Q(next_state, ·) − Q(s,a)]`

Reward +1 khi hữu ích, −1 khi chưa hữu ích. Lượt hỏi tiếp theo tạo `next_state`; phản hồi trước khi có lượt tiếp theo được coi là terminal lúc cập nhật. Mỗi câu trả lời nhận tối đa một reward để tránh bấm lặp làm sai dữ liệu. Chính sách này học cách diễn đạt; **không tối ưu thuốc, không đổi ngưỡng xét nghiệm**. Cần nhiều phản hồi thật mới đánh giá được ích lợi của chính sách. UCB1 của bản đầu chỉ còn dùng cho phần tham khảo báo cáo cũ; chat dùng Q-learning.

## Kiểm thử

```cmd
.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -p "test_*.py" -v
.venv\Scripts\python.exe -X utf8 -m tests.check_ocr
.venv\Scripts\python.exe -X utf8 -m tests.check_ui
.venv\Scripts\python.exe -X utf8 -m tests.check_chat_ui
.venv\Scripts\python.exe -X utf8 -m tests.check_floating_chat_ui
.venv\Scripts\python.exe -X utf8 -m tests.check_mysql
.venv\Scripts\python.exe -X utf8 -m tests.check_chat_live
```

Unit/UI tests dùng DB tạm độc lập; bài `check_chat_ui` dùng model fixture để kiểm tra giao diện, không coi là kiểm tra Ollama. `check_chat_live` dùng **Ollama và MySQL thật**, tạo hội thoại có nhãn KIỂM THỬ. Kết quả chạy cụ thể ghi trong `TEST_RESULTS.md`.

## Giới hạn vận hành

Ứng dụng nghiên cứu/đồ án chạy localhost, chưa có đăng nhập hoặc phân quyền cho nhiều người. Không đưa ra Internet như hệ thống bệnh viện. OCR có thể nhầm số/cột/đơn vị nên bắt buộc xác nhận thủ công. Điểm ML chưa hiệu chuẩn, không phải xác suất bệnh đã kiểm chứng. Cần bác sĩ và xét nghiệm phù hợp để xác nhận. Không kết luận type 1/type 2 hoặc kê thuốc. Các guardrail là biện pháp giảm lỗi, không bảo đảm LLM luôn đúng. Không nạp pickle từ nguồn ngoài.

Tài liệu kỹ thuật: [Ollama API](https://docs.ollama.com/api/chat), [Qwen3](https://qwenlm.github.io/blog/qwen3/), [RapidOCR](https://github.com/RapidAI/RapidOCR). Kiến thức sàng lọc: [NIDDK](https://www.niddk.nih.gov/health-information/diabetes/overview/tests-diagnosis). `compose.yaml` chỉ là lựa chọn khi chuyển máy; bản hiện tại dùng MySQL đã cài sẵn.
