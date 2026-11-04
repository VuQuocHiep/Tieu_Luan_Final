"""A 3-role multi-agent report pipeline: collector -> analyst -> writer.
Collector: deterministic validation and collection.
Analyst: exact numerical calculations plus optional LLM insight.
Writer: report formatting and optional LLM drafting.
"""
import csv
import io
import json
from collections import defaultdict
from datetime import date
from pathlib import Path
from common.ollama_client import OllamaClient, OllamaError

REQUIRED_COLUMNS = {"date", "sku", "product", "category", "quantity", "unit_price", "unit_cost"}
SAMPLE_CSV = Path(__file__).resolve().parents[1] / "data" / "sales_sample.csv"


class DataAgent:
    """Retrieve, check and normalize data into a trusted hand-off."""
    name = "Agent thu thập dữ liệu"

    def run(self, csv_data: bytes, start: date | None = None, end: date | None = None):
        try:
            decoded = csv_data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("CSV phải dùng mã hóa UTF-8.") from exc
        reader = csv.DictReader(io.StringIO(decoded))
        if not reader.fieldnames or not REQUIRED_COLUMNS.issubset(set(reader.fieldnames)):
            raise ValueError("Thiếu cột CSV. Cần có: " + ", ".join(sorted(REQUIRED_COLUMNS)))
        rows = []
        invalid = []
        total = 0
        for line, raw in enumerate(reader, start=2):
            total += 1
            try:
                d = date.fromisoformat(raw["date"].strip())
                qty = int(raw["quantity"])
                price = int(raw["unit_price"])
                cost = int(raw["unit_cost"])
                if qty <= 0 or price < 0 or cost < 0:
                    raise ValueError("Giá trị số không hợp lệ")
                sku = (raw["sku"] or "").strip()
                product = (raw["product"] or "").strip()
                category = (raw["category"] or "").strip()
                if not sku or not product:
                    raise ValueError("Thiếu mã hoặc tên sản phẩm")
                if (start is None or d >= start) and (end is None or d <= end):
                    rows.append({"date": d.isoformat(), "sku": sku, "product": product,
                                 "category": category, "quantity": qty,
                                 "unit_price": price, "unit_cost": cost})
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                invalid.append({"line": line, "error": str(exc)})
        if not rows:
            raise ValueError("Không có dòng dữ liệu hợp lệ trong khoảng ngày đã chọn.")
        return {"rows": rows, "source_rows": total, "included_rows": len(rows),
                "skipped_rows": invalid, "date_range": [min(x["date"] for x in rows), max(x["date"] for x in rows)]}


class AnalysisAgent:
    """Compute trustworthy metrics with Python, then optionally create bounded observations."""
    name = "Agent phân tích dữ liệu"

    def run(self, collection: dict, client: OllamaClient | None = None):
        rows = collection["rows"]
        units = sum(x["quantity"] for x in rows)
        revenue = sum(x["quantity"] * x["unit_price"] for x in rows)
        cost = sum(x["quantity"] * x["unit_cost"] for x in rows)
        by_product = defaultdict(lambda: {"units": 0, "revenue": 0})
        by_category = defaultdict(lambda: {"units": 0, "revenue": 0})
        by_date = defaultdict(int)
        for r in rows:
            rev = r["quantity"] * r["unit_price"]
            k = f"{r['sku']} - {r['product']}"
            by_product[k]["units"] += r["quantity"]
            by_product[k]["revenue"] += rev
            by_category[r["category"]]["units"] += r["quantity"]
            by_category[r["category"]]["revenue"] += rev
            by_date[r["date"]] += rev
        top_products = sorted([{"product": k, **v} for k, v in by_product.items()],
                              key=lambda x: x["revenue"], reverse=True)
        categories = sorted([{"category": k, **v} for k, v in by_category.items()],
                            key=lambda x: x["revenue"], reverse=True)
        metrics = {"period": collection["date_range"], "transaction_rows": len(rows),
                   "units": units, "revenue_vnd": revenue, "estimated_cost_vnd": cost,
                   "estimated_gross_profit_vnd": revenue - cost,
                   "estimated_gross_margin_pct": round((revenue - cost) / revenue * 100, 2) if revenue else 0,
                   "top_products": top_products[:5], "categories": categories,
                   "revenue_by_day": [{"date": d, "revenue": by_date[d]} for d in sorted(by_date)],
                   "notes": "Dữ liệu giả lập, doanh thu = số lượng x đơn giá; lợi nhuận gộp ước tính chưa trừ vận hành, thuế và các chi phí khác."}
        biggest = top_products[0] if top_products else None
        insights = (f"Số lượng bán trong kỳ: {units}. "
                    f"Sản phẩm có doanh thu cao nhất: {biggest['product']}" if biggest else "Chưa có sản phẩm.")
        if client is not None:
            try:
                message = client.chat([
                    {"role": "system", "content": "Bạn là Agent PHÂN TÍCH SỐ LIỆU. Nhận JSON thống kê đã tính bằng Python. Viết 3-5 nhận xét dựa hoàn toàn vào số liệu. Không tự tạo con số mới, không nhầm lẫn doanh thu với lợi nhuận thực. Trả lời tiếng Việt ngắn gọn."},
                    {"role": "user", "content": json.dumps(metrics, ensure_ascii=False)}
                ])
                insights = message.get("content", "").strip() or insights
            except OllamaError:
                pass
        return {"metrics": metrics, "insights": insights}


class ReportAgent:
    """Compose the final report; optionally use a separately prompted LLM role."""
    name = "Agent tạo báo cáo"

    def run(self, analysis: dict, client: OllamaClient | None = None) -> str:
        m = analysis["metrics"]
        period = f"{m['period'][0]} đến {m['period'][1]}"
        top_lines = "\n".join(
            f"- {x['product']}: {x['units']} sản phẩm, doanh thu {x['revenue']:,} VND"
            for x in m["top_products"])
        cat_lines = "\n".join(
            f"- {x['category']}: {x['units']} sản phẩm, doanh thu {x['revenue']:,} VND"
            for x in m["categories"])
        fallback = (
            "BÁO CÁO PHÂN TÍCH BÁN HÀNG (DỮ LIỆU MẪU)\n\n"
            f"1. Thời gian: {period}. Số dòng dữ liệu: {m['transaction_rows']}.\n\n"
            f"2. Tổng quan: Số lượng {m['units']} sản phẩm; doanh thu {m['revenue_vnd']:,} VND; "
            f"giá vốn {m['estimated_cost_vnd']:,} VND; lợi nhuận gộp ước tính "
            f"{m['estimated_gross_profit_vnd']:,} VND; biên gộp {m['estimated_gross_margin_pct']}%.\n\n"
            f"3. Sản phẩm nổi bật:\n{top_lines}\n\n4. Theo danh mục:\n{cat_lines}\n\n"
            f"5. Nhận xét:\n{analysis['insights']}\n\n6. Lưu ý:\n{m['notes']}"
        )
        if client is None:
            return fallback
        try:
            response = client.chat([
                {"role": "system", "content": "Bạn là Agent VIẾT BÁO CÁO tiếng Việt dành cho sinh viên. Viết báo cáo có 6 mục: Tổng quan, Thống kê, Sản phẩm, Danh mục, Nhận xét, Hạn chế. Chỉ dùng số liệu JSON dưới đây, KHÔNG suy đoán số liệu hay hiện trạng kinh doanh thực. Ghi rõ đây là dữ liệu minh họa. Hãy viết đoạn văn rõ ràng, không trang trí."},
                {"role": "user", "content": json.dumps(analysis, ensure_ascii=False)}
            ])
            return response.get("content", "").strip() or fallback
        except OllamaError:
            return fallback


class MultiAgentCoordinator:
    """Explicit state passing and clear responsibility boundaries among 3 agents."""
    def run(self, csv_data: bytes, start: date | None = None,
            end: date | None = None, client: OllamaClient | None = None):
        collected = DataAgent().run(csv_data, start, end)
        analyzed = AnalysisAgent().run(collected, client)
        report = ReportAgent().run(analyzed, client)
        trace = [
            {"agent": DataAgent.name, "result": f"Kiểm tra {collected['source_rows']} dòng; đưa {collected['included_rows']} dòng hợp lệ vào kỳ báo cáo."},
            {"agent": AnalysisAgent.name, "result": "Tính chính xác các tổng số và lập nhận xét theo dữ liệu."},
            {"agent": ReportAgent.name, "result": "Soạn báo cáo từ kết quả phân tích; không chỉnh sửa số liệu nguồn."},
        ]
        return {"data": collected, "analysis": analyzed, "report": report, "trace": trace}
