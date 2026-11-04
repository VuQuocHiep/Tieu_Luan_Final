"""App 2: warehouse agent, with explicit UI confirmation for drafts."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from common.agent_core import AgentRunner, Tool
from common.ollama_client import OllamaClient, OllamaError
from common.text_utils import normalize
from app2_quan_ly_kho import warehouse_core as db

st.set_page_config(page_title="Agent 2 - Quản lý kho", page_icon="📦", layout="wide")
db.initialize()
st.title("Ứng dụng 2: Agent hỗ trợ quản lý kho")
st.caption("Dữ liệu SQLite mẫu, các công cụ tra cứu và lập đề xuất nhập hàng. Không kết nối thiết bị ESP32 thật.")

with st.sidebar:
    st.subheader("Tạo phiếu đề xuất nhập hàng")
    st.write("Chỉ lưu phiếu nháp khi người dùng bấm xác nhận, Agent không tự thay đổi dữ liệu kho.")
    sku = st.selectbox("Mã sản phẩm", [p["sku"] for p in db.products()])
    qty = st.number_input("Số lượng đề nghị", min_value=1, max_value=100000, value=10)
    if st.button("Xác nhận tạo phiếu nháp"):
        draft_id = db.add_draft(sku, qty)
        st.success(f"Đã tạo phiếu nháp #{draft_id}. Chưa mua hàng hoặc cập nhật tồn kho.")
    with st.expander("Phiếu nháp đã tạo"):
        st.dataframe(db.get_drafts(), hide_index=True)

left, right = st.columns(2)
with left:
    st.subheader("Danh sách sản phẩm")
    st.dataframe(db.products(), use_container_width=True, hide_index=True)
with right:
    st.subheader("Sản phẩm tồn thấp")
    st.dataframe(db.low_stock(), use_container_width=True, hide_index=True)

st.divider()
client = OllamaClient()
ready = client.available()
st.caption(f"Mô hình {client.model}: " + ("đã sẵn sàng" if ready else "chưa có; sử dụng chế độ phân tích theo quy tắc"))

def tool_low_stock():
    return {"products": db.low_stock()}

def tool_product(keyword: str):
    return {"products": db.find_product(keyword)}

def tool_sensors():
    return db.sensor_status()

def tool_sales(days: int = 30):
    return db.sales_summary(days)

def tool_reorder(lead_days: int = 7):
    return db.reorder_suggestions(lead_days)

tools = [
    Tool("low_stock", "Lấy danh sách sản phẩm có số lượng tồn thấp hơn hoặc bằng mức tối thiểu.",
         {"type": "object", "properties": {}}, tool_low_stock),
    Tool("find_product", "Tra cứu sản phẩm theo mã SKU hoặc tên.",
         {"type": "object", "properties": {"keyword": {"type": "string"}}, "required": ["keyword"]}, tool_product),
    Tool("sensors", "Lấy lần ghi nhiệt độ, độ ẩm, khí gas ADC gần nhất từ dữ liệu mẫu.",
         {"type": "object", "properties": {}}, tool_sensors),
    Tool("sales_summary", "Thống kê lượng xuất hàng trong số ngày được yêu cầu.",
         {"type": "object", "properties": {"days": {"type": "integer"}}}, tool_sales),
    Tool("reorder_suggestions", "Tạo đề xuất số lượng nhập dựa trên quy tắc tồn kho và mức xuất trung bình, không dùng ML.",
         {"type": "object", "properties": {"lead_days": {"type": "integer"}}}, tool_reorder),
]
prompt = """Bạn là Agent hỗ trợ quản lý kho, trả lời bằng tiếng Việt. Dữ liệu từ SQLite trong ứng dụng là DỮ LIỆU MẪU.
Luôn gọi công cụ phù hợp trước khi nêu số lượng, nhiệt độ, tồn kho hoặc kết quả phân tích.
Không bịa số liệu, không gọi ADC là ppm, không nói đang theo dõi cảm biến trực tiếp.
Đề xuất nhập kho là quy tắc đơn giản dựa trên lượng xuất 30 ngày, KHÔNG phải dự báo ML.
Bạn KHÔNG có quyền sửa dữ liệu, tạo đơn mua thực tế hoặc gửi lệnh tới thiết bị.
Trả lời rõ kết quả, nêu giới hạn dữ liệu khi phù hợp."""


def offline_answer(query: str):
    q = normalize(query)
    if any(w in q for w in ["nhiet", "do am", "khi gas", "cam bien", "moi truong"]):
        s = db.sensor_status()
        return (f"**Thông số lần ghi mẫu:** {s['temperature']} °C, độ ẩm {s['humidity']}%, "
                f"gas ADC {s['gas_adc']} (không phải ppm). Thời gian: {s['recorded_at']}. "
                "Đây là dữ liệu mô phỏng, không phải đo theo thời gian thực.")
    if any(w in q for w in ["nhap", "bo sung", "du doan", "de xuat"]):
        res = db.reorder_suggestions()
        items = res["suggestions"]
        lines = [f"- {x['sku']} - {x['name']}: tồn {x['current']}, đề xuất thêm {x['suggested_units']}" for x in items]
        return "**Đề xuất bổ sung theo quy tắc (không phải ML):**\n" + ("\n".join(lines) if lines else "Không có sản phẩm cần bổ sung.")
    if any(w in q for w in ["xuat", "doanh thu", "ban hang", "tieu thu"]):
        data = db.sales_summary()["products"]
        return "**Lượng xuất 30 ngày (dữ liệu mẫu):**\n" + "\n".join(
            f"- {x['name']}: {x['units']} sản phẩm" for x in data)
    if any(w in q for w in ["ton thap", "sap het", "canh bao", "ton kho", "san pham"]):
        items = db.low_stock()
        return "**Danh sách tồn thấp:**\n" + ("\n".join(
            f"- {x['sku']} - {x['name']}: còn {x['quantity']}, tối thiểu {x['min_stock']}" for x in items)
            if items else "Không có sản phẩm tồn thấp.")
    return "Bạn có thể hỏi: 'Sản phẩm nào tồn thấp?', 'Đề xuất nhập hàng', 'Nhiệt độ kho?', hoặc 'Lượng xuất 30 ngày?'."

if "messages_app2" not in st.session_state:
    st.session_state.messages_app2 = []
if st.button("Xóa cuộc trò chuyện"):
    st.session_state.messages_app2 = []
for m in st.session_state.messages_app2:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
question = st.chat_input("Ví dụ: Kiểm tra tồn kho và đề xuất nhập hàng")
if question:
    prior = st.session_state.messages_app2[-6:]
    st.session_state.messages_app2.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        if ready:
            try:
                result = AgentRunner(client, prompt, tools, max_steps=5).run(question, prior)
                answer = result["answer"]
                if not result["trace"]:
                    st.warning("LLM chưa gọi công cụ. Không sử dụng các số liệu chưa được xác minh.")
                with st.expander("Nhật ký gọi công cụ"):
                    for item in result["trace"]:
                        st.json(item)
            except OllamaError as exc:
                st.warning(str(exc))
                answer = offline_answer(question)
        else:
            answer = offline_answer(question)
        st.markdown(answer)
    st.session_state.messages_app2.append({"role": "assistant", "content": answer})
