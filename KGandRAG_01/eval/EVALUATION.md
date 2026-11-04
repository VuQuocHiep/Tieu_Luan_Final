# Đánh giá thực nghiệm retrieval — Diabetes KG + RAG

Mục tiêu: kiểm tra **bộ truy xuất** có đưa đúng đoạn tham khảo vào danh sách Top-K không. **Không** kiểm định độ đúng y khoa của câu trả lời LLM, không đánh giá chẩn đoán hay điều trị.

## Tập câu hỏi

- Dữ liệu: `eval/questions.json` — 30 câu hỏi tiếng Việt, chủ đề tổng quan, triệu chứng, xét nghiệm, thuốc, biến chứng và cấp cứu.
- Mỗi câu gắn `gold_docs`, ví dụ Q08 → `D08`, do người viết dataset tự xác định từ chính 30 chunk dữ liệu.
- Đây là **in-domain validation set**, có khả năng thiên lệch cao, **không phải** external validation set.
- Kết quả được tính bằng `python -m scripts.run_evaluation --k 3`. Số liệu phụ thuộc dataset, model và cấu hình hiện tại.

## Định nghĩa chỉ số

Với câu hỏi `q`, `Gold(q)` là tập ID tài liệu chuẩn, `TopK(q)` là các ID hệ thống trả về.

- `Hit@K = số câu có Gold ∩ TopK khác rỗng / tổng số câu`.
- `Recall@K = trung bình theo câu của |Gold ∩ TopK| / |Gold|`.
- `Precision@K = trung bình theo câu của |Gold ∩ TopK| / K`; ở đây mẫu số bằng K theo định nghĩa thử nghiệm.
- `MRR@K = trung bình 1 / vị trí tài liệu đúng đầu tiên`, hoặc 0 nếu không có tài liệu đúng trong K kết quả.

Các phương pháp so sánh dùng **cùng một** dataset và giá trị K. Đây là phép đo chất lượng tìm kiếm, không phải tỷ lệ câu trả lời y tế chính xác.

## Kết quả đo cục bộ (TF-IDF, 30 câu, K = 3)

| Phương pháp | Hit@3 | Recall@3 | Precision@3 | MRR@3 |
|---|---:|---:|---:|---:|
| Text TF-IDF | 96,7% | 96,7% | 33,3% | 0,911 |
| Knowledge Graph | 70,0% | 68,3% | 23,3% | 0,411 |
| Hybrid Text + KG | 100,0% | 100,0% | 34,4% | 0,950 |

> Điểm cao một phần vì câu hỏi và tài liệu chuẩn cùng được xây dựng theo chủ đề cụ thể. Không được trình bày như kết quả benchmark công bố hoặc chỉ số tin cậy y khoa. Tăng kích thước dữ liệu, tạo câu hỏi độc lập, câu hỏi gây nhiễu, biến thể ngôn ngữ, số liệu sai, đánh giá bởi chuyên gia mới đưa ra kết luận tổng quát được.

### Ý nghĩa các phương pháp

**Text TF-IDF** dùng đặc trưng ký tự và từ/ngữ để xếp hạng đoạn văn. **KG-only** dựa vào nhận diện thực thể/alias và mở rộng quan hệ đồ thị, thiếu khả năng diễn giải toàn bộ câu hỏi khi có nhiều tài liệu chung thực thể. **Hybrid** cộng hai điểm để bổ sung ngữ nghĩa truy vấn và cấu trúc thực thể.

Với tùy chọn `RETRIEVAL_ENCODER=sbert`, các điểm không còn khớp bảng trên và phải chạy lại. Nếu sửa dữ liệu/tập câu hỏi, phải chạy lại và cập nhật báo cáo này.

## Kiểm thử ngoài retrieval

`tests/test_engine.py` kiểm tra:

1. Tài liệu và đồ thị có ID hợp lệ và thống nhất.
2. Chuẩn hóa tiếng Việt và retrieval theo HbA1c.
3. KG trả về thực thể và quan hệ.
4. Câu hỏi ngoài lĩnh vực đủ khác biệt bị từ chối do thiếu bằng chứng.
5. Tình huống bất tỉnh/co giật được ưu tiên cảnh báo cấp cứu.
6. Yêu cầu chỉnh liều thuốc được chuyển sang cảnh báo an toàn.
7. Mặc định chạy không cần API key, có dẫn ID chứng cứ.
8. HTTP endpoints và kiểm tra validation.

Cơ chế nhận biết cấp cứu sử dụng từ khóa thủ công và **có thể bỏ sót** biểu hiện nguy hiểm được diễn đạt khác đi. Không sử dụng như hệ thống phân loại cấp cứu thực tế.

## Đề xuất đánh giá mở rộng cho đồ án

- Xây tập **ngoài phân phối** với câu hỏi không liên quan, câu hỏi thiếu dữ liệu và câu cố ý đánh lừa.
- Tách tập đánh giá do một nhóm khác viết, không nhìn nội dung chunk trước khi đặt câu hỏi.
- So sánh Vector vs KG vs Hybrid trên các K=1/3/5, thực hiện ablation bỏ entity-expansion.
- Nhờ người có chuyên môn đánh giá: **factual correctness**, **faithfulness to retrieved context**, **medical safety**, **citation correctness**, thang điểm 1–5 theo rubric; đánh giá mù và có đồng thuận.
- Đo latency p50/p95 và bộ nhớ trên cùng cấu hình máy, thêm giám sát tỉ lệ không trả lời khi thiếu thông tin.
