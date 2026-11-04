# Đánh giá thực nghiệm: Knowledge Graph + RAG về chứng khoán và vàng

## Mục đích và bộ dữ liệu

Đo độ đúng của **truy xuất thông tin** (retrieval), không đo hiệu quả giao dịch, chất lượng dự báo giá hoặc chất lượng lời khuyên đầu tư. `eval/questions.json` có **40 câu hỏi tiếng Việt** và **40 nhãn tài liệu D01…D40** (mỗi câu 1 tài liệu chuẩn) được viết thủ công dựa trên cùng corpus. Các câu hỏi bao quát: cổ phiếu, lệnh giao dịch, P/E, ETF, vàng, USD, lãi suất, lạm phát, rủi ro, đa dạng hóa và dữ liệu thời gian thực.

Tập này là **in-domain, tác giả tự gán nhãn**, nên có thể lạc quan và không nên dùng làm bằng chứng rằng mô hình tốt hơn các hệ thống ngoài thị trường. Không có kiểm định từ chuyên gia tài chính độc lập.

## Các chỉ số

Ký hiệu Gold(q) = tập ID tài liệu chuẩn, TopK(q) = các ID được truy xuất đầu tiên.

- **Hit@K** = tỷ lệ câu hỏi có ít nhất một tài liệu đúng trong Top-K.
- **Recall@K** = trung bình số tài liệu đúng được lấy lại chia cho số tài liệu đúng.
- **Precision@K** = trung bình số tài liệu đúng trong Top-K chia cho K.
- **MRR@K** = trung bình nghịch đảo vị trí tài liệu đúng đầu tiên, hoặc 0 nếu không có trong Top-K.

Mỗi câu hỏi có đúng **1 gold document**, nên Precision@3 bị chặn trên ở 33,3% theo định nghĩa này, dù Hit@3 đạt 100%.

## Kết quả chạy trực tiếp trên phiên bản hiện tại

Lệnh chạy:

```bash
python -m scripts.run_evaluation --k 3
```

| Phương pháp | Hit@3 | Recall@3 | Precision@3 | MRR@3 |
|---|---:|---:|---:|---:|
| Text TF-IDF | 100,0% | 100,0% | 33,3% | 0,975 |
| Knowledge Graph only | 87,5% | 87,5% | 29,2% | 0,733 |
| Hybrid Text + KG | 100,0% | 100,0% | 33,3% | 0,975 |

**Giải thích trung thực:** trên tập câu hỏi này, **Text và Hybrid có chất lượng bằng nhau theo các chỉ số**, chưa chứng minh Hybrid tốt hơn. KG-only có kết quả thấp hơn vì quan hệ một-hop và alias không hiểu mọi chi tiết của câu hỏi. Hybrid vẫn hữu ích để minh họa kết hợp chứng cứ từ hai cấu trúc và cung cấp graph explanations.

## Tệp kết quả

- `eval/reports/summary.csv`: CSV UTF-8-BOM để mở bằng Excel.
- `eval/reports/details.csv`: 120 hàng dữ liệu câu hỏi × ba phương pháp, gồm Gold IDs, các ID được tìm và điểm từng trường hợp.
- `eval/reports/results.json`: cấu trúc đầy đủ gồm thời điểm UTC, điểm tổng và chi tiết.
- Tab **Bảng đánh giá** trong ứng dụng: chạy đánh giá trực tiếp ở K=1/3/5 và xuất CSV.

Các số trên chỉ đúng với cấu hình **RETRIEVAL_ENCODER=tfidf**, bộ JSON và mã nguồn hiện tại. Bật SBERT, sửa trọng số/graph/docs hoặc thêm dữ liệu thì **bắt buộc chạy lại** và cập nhật các bảng trình bày.

## Đề xuất kiểm chứng nâng cao cho báo cáo nghiên cứu

1. Cho người khác xây tập câu hỏi **blind test** mà không xem trước các đoạn tài liệu.
2. Thêm câu hỏi gây nhiễu/ngoài lĩnh vực, câu không đủ dữ liệu và câu bị đảo ngữ nghĩa.
3. So sánh Hit/Recall/MRR ở K=1,3,5; chạy ablation `vector`, `graph`, `hybrid`, `SBERT`.
4. Đo thời gian phản hồi p50/p95 và chi phí bộ nhớ với cùng cấu hình.
5. Với câu trả lời LLM, đo **faithfulness, citation accuracy, abstention**, kiểm tra nguy cơ bịa giá hoặc gợi ý mua bán không có căn cứ.
6. Chỉ xem xét dữ liệu thị trường thực khi có nguồn API uy tín, giấy phép hợp lệ và dấu thời gian; kết quả đầu tư thực cần backtest có phòng tránh lookahead/survivorship bias.

Đây là mô hình demo học thuật, **không phải mô hình giao dịch hay lời khuyên đầu tư**.
