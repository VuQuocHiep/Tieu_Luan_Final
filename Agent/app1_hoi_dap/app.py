"""App 1: document QA agent with real tool calls and BM25 retrieval."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from common.agent_core import AgentRunner, Tool
from common.ollama_client import OllamaClient, OllamaError
from app1_hoi_dap.rag_core import BM25Index, build_chunks, sample_file

st.set_page_config(page_title="Agent 1 - Hỏi đáp tài liệu", page_icon="📄", layout="wide")
st.title("Ứng dụng 1: Agent hỏi đáp tài liệu")
st.caption("Tìm kiếm BM25 trong PDF/TXT, sau đó Ollama sử dụng công cụ để tổng hợp câu trả lời kèm nguồn.")

with st.sidebar:
    st.subheader("Tài liệu")
    uploaded = st.file_uploader("Thêm PDF hoặc TXT", type=["pdf", "txt"], accept_multiple_files=True)
    include_sample = st.checkbox("Kèm tài liệu kho mẫu", value=True)
    k = st.slider("Số đoạn tìm kiếm", min_value=1, max_value=6, value=3)
    st.info("PDF scan chỉ chứa hình ảnh có thể không trích xuất được chữ. Ứng dụng không dùng OCR.")

sources = [sample_file()] if include_sample else []
sources.extend((file.name, file.getvalue()) for file in uploaded)
if not sources:
    st.warning("Hãy tải tài liệu lên hoặc bật tài liệu mẫu.")
    st.stop()
try:
    chunks = build_chunks(sources)
except Exception as exc:
    st.error(f"Không đọc được tài liệu: {exc}")
    st.stop()
if not chunks:
    st.error("Không trích xuất được văn bản từ tài liệu.")
    st.stop()
index = BM25Index(chunks)
client = OllamaClient()
ready = client.available()
st.write(f"Đã lập chỉ mục **{len(chunks)} đoạn** từ **{len(sources)} tài liệu**.")
st.caption(f"Mô hình: {client.model}. " + ("Ollama đã sẵn sàng." if ready else "Chưa có Ollama/mô hình; vẫn dùng được tìm kiếm trích đoạn."))


def search_documents(query: str, top_k: int = k):
    return {"results": index.search(query, top_k), "retrieval_method": "BM25"}


def list_documents():
    return {"sources": [{"filename": name, "size_bytes": len(data)} for name, data in sources]}


tools = [
    Tool("search_documents", "Tìm các đoạn liên quan trong tài liệu đã tải lên. Phải dùng công cụ này trước khi trả lời câu hỏi về nội dung tài liệu.",
         {"type": "object", "properties": {"query": {"type": "string"}, "top_k": {"type": "integer", "minimum": 1, "maximum": 6}},
          "required": ["query"]}, search_documents),
    Tool("list_documents", "Liệt kê tên tài liệu đang có trong bộ nhớ.",
         {"type": "object", "properties": {}}, list_documents),
]
system_prompt = """Bạn là Agent tra cứu tài liệu tiếng Việt. Dùng công cụ search_documents để lấy đoạn gốc trước khi trả lời nội dung tài liệu.
Chỉ khẳng định những điều có bằng chứng từ công cụ, nêu nguồn theo dạng [tên tệp, trang X].
Nếu không tìm ra thông tin, nói rõ không tìm thấy. Không tự bịa nội dung, tên trang, trích dẫn.
Bạn có thể gọi công cụ nhiều lần nếu kết quả đầu chưa đủ. Trả lời rõ ràng, ngắn gọn và dễ hiểu."""

if "messages_app1" not in st.session_state:
    st.session_state.messages_app1 = []
if st.button("Xóa lịch sử hội thoại"):
    st.session_state.messages_app1 = []
for item in st.session_state.messages_app1:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])

question = st.chat_input("Ví dụ: Khi hàng tồn kho dưới mức tối thiểu thì phải làm gì?")
if question:
    prior = [{"role": x["role"], "content": x["content"]} for x in st.session_state.messages_app1[-6:]]
    st.session_state.messages_app1.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        hits = index.search(question, k)
        if ready:
            try:
                out = AgentRunner(client, system_prompt, tools, max_steps=4).run(question, prior)
                answer, trace = out["answer"], out["trace"]
                st.markdown(answer)
                with st.expander("Các bước sử dụng công cụ"):
                    for entry in trace:
                        st.json(entry)
                if not trace:
                    st.warning("Mô hình chưa sử dụng công cụ truy xuất; cần kiểm chứng câu trả lời trước khi sử dụng.")
            except OllamaError as exc:
                st.warning(str(exc))
                ready = False
        if not ready:
            if hits:
                answer = "**Chế độ không có LLM:** Dưới đây là những đoạn gần câu hỏi nhất, chưa phải câu trả lời được tổng hợp.\n\n"
                answer += "\n\n".join(f"**{h['source']} - trang {h['page']}**\n\n{h['text']}" for h in hits)
            else:
                answer = "Không tìm thấy đoạn văn bản phù hợp. Hãy thay đổi câu hỏi hoặc tải tài liệu khác."
            st.markdown(answer)
        with st.expander("Đối chiếu nguồn tìm kiếm BM25"):
            for hit in hits:
                st.write(f"**{hit['source']} - trang {hit['page']} (điểm {hit['score']})**")
                st.write(hit["text"])
    st.session_state.messages_app1.append({"role": "assistant", "content": answer})
