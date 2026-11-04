"""App 3: a controlled multi-agent report workflow."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from common.ollama_client import OllamaClient
from app3_multi_agent.agents import MultiAgentCoordinator, SAMPLE_CSV

st.set_page_config(page_title="Agent 3 - Multi-Agent", page_icon="📊", layout="wide")
st.title("Ứng dụng 3: Multi-Agent tạo báo cáo")
st.caption("Ba vai trò: Agent thu thập dữ liệu, Agent phân tích, Agent viết báo cáo.")

upload = st.file_uploader("Chọn dữ liệu bán hàng CSV (hoặc dùng dữ liệu mẫu)", type=["csv"])
data = upload.getvalue() if upload else SAMPLE_CSV.read_bytes()
model = OllamaClient()
use_llm = st.checkbox(f"Dùng Ollama ({model.model}) cho phân tích nhận xét và viết báo cáo", value=False,
                      help="Nếu tắt, cả ba Agent vẫn xử lý theo các quy tắc có thể kiểm tra được.")
if use_llm and not model.available():
    st.warning("Chưa có Ollama/mô hình này; hệ thống sẽ tạo báo cáo theo quy tắc.")
    use_llm = False
st.write("**Cấu trúc CSV:** date, sku, product, category, quantity, unit_price, unit_cost.")
st.caption("Đơn giá và giá vốn tính bằng VND. Dữ liệu mẫu là giả lập để minh họa, không phải số liệu thực tế.")

if st.button("Chạy 3 Agent và tạo báo cáo", type="primary"):
    with st.spinner("Đang kiểm tra dữ liệu, tính toán và soạn báo cáo..."):
        try:
            output = MultiAgentCoordinator().run(data, client=model if use_llm else None)
            st.session_state.report3 = output
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
if "report3" in st.session_state:
    result = st.session_state.report3
    metrics = result["analysis"]["metrics"]
    a, b, c = st.columns(3)
    a.metric("Sản phẩm đã xuất", f"{metrics['units']:,}")
    b.metric("Doanh thu mẫu", f"{metrics['revenue_vnd']:,} VND")
    c.metric("Lợi nhuận gộp ước tính", f"{metrics['estimated_gross_profit_vnd']:,} VND")
    with st.expander("Nhật ký phối hợp 3 Agent", expanded=True):
        for step in result["trace"]:
            st.write(f"**{step['agent']}:** {step['result']}")
        invalid = result["data"]["skipped_rows"]
        if invalid:
            st.warning(f"Bỏ qua {len(invalid)} dòng không hợp lệ.")
            st.json(invalid[:20])
    st.subheader("Báo cáo")
    st.text(result["report"])
    st.download_button("Tải báo cáo TXT", result["report"].encode("utf-8"),
                       file_name="bao_cao_multi_agent.txt", mime="text/plain")
    with st.expander("Dữ liệu thống kê do Python tính"):
        st.json(metrics)
else:
    st.info("Bấm nút 'Chạy 3 Agent và tạo báo cáo' để bắt đầu.")
