"""Streamlit screens for consultation, KB and MySQL-backed model training."""
import json
from pathlib import Path
import streamlit as st
from .chat import ChatCoordinator
from .knowledge import seed_knowledge, ingest_text, read_upload, fetch_web
from .llm import LocalLLM, ModelUnavailable
from .config import ARTIFACTS, FEATURES
from .design import section

def get_db(factory):
    try:
        return factory()
    except Exception:
        st.error("Chưa kết nối được MySQL. Kiểm tra MYSQL_* trong .env và khởi động lại ứng dụng.")
        return None

def render_chat(factory, compact=False):
    prefix = "floating_" if compact else ""
    if compact:
        st.markdown('<div class="assistant-head"><div class="assistant-avatar">✧</div><div><strong>Trợ lý GlucoCare</strong><small>Cùng bạn hiểu chỉ số · Trả lời có nguồn</small></div></div>', unsafe_allow_html=True)
    else:
        section("Bước 03 / Đồng hành", "Một cuộc trò chuyện có cơ sở", "Hỏi về chỉ số, thói quen sinh hoạt hoặc bước tiếp theo. Trợ lý sử dụng nguồn tham khảo và báo cáo bạn lựa chọn.")
    llm = LocalLLM()
    if not compact:
        st.caption(f"Mô hình: {llm.model} · Chạy cục bộ · Tư vấn giáo dục, không thay thế khám bệnh.")
    if not compact and st.button("Kiểm tra mô hình AI"):
        status = llm.status()
        if status["ready"]:
            st.success(f"Sẵn sàng: {status['model']}")
        else:
            st.warning("Chưa sẵn sàng. Mở Ollama và kiểm tra mô hình bằng ollama list.")
    db = get_db(factory)
    if db is None:
        return
    if not compact:
        conversations = db.conversation_list()
        ids = [row["id"] for row in conversations]
        labels = {row["id"]: f"{row['title']} · {row['created_at']}" for row in conversations}
        selected = st.selectbox("Mở cuộc trò chuyện đã lưu", ids, index=None,
            format_func=lambda key: labels[key], key="conversation_picker")
        if selected and selected != st.session_state.get("last_picker"):
            st.session_state.chat_id = selected
            st.session_state.last_picker = selected
    def reset_chat():
        st.session_state.pop("chat_id", None)
        st.session_state.pop("last_picker", None)
        st.session_state.pop("conversation_picker", None)
    if not compact:
        st.button("Cuộc trò chuyện mới", on_click=reset_chat, key=prefix + "chat_new")
    cid = st.session_state.get("chat_id")
    history = db.chat_history(cid) if cid else []
    latest = st.session_state.get("result")
    context_options = ["Không dùng chỉ số xét nghiệm"]
    if latest and latest.get("patient"):
        context_options.append("Dùng báo cáo vừa phân tích")
    if history and json.loads(history[-1]["response"]).get("context"):
        context_options.append("Dùng chỉ số trong cuộc trò chuyện này")
    controls = st.expander("Ngữ cảnh & cuộc trò chuyện", expanded=False) if compact else st.container()
    with controls:
        if compact:
            st.button("Cuộc trò chuyện mới", on_click=reset_chat, key=prefix + "chat_new")
        context_choice = st.selectbox("Ngữ cảnh tư vấn", context_options,
            index=len(context_options) - 1, key=prefix + "chat_context_choice")
    report = None
    if context_choice == "Dùng báo cáo vừa phân tích":
        report = latest
    elif context_choice == "Dùng chỉ số trong cuộc trò chuyện này":
        report = json.loads(history[-1]["response"])["context"]
    if report:
        with st.expander("Các chỉ số sẽ cung cấp cho trợ lý"):
            st.json(report["patient"])
    with st.container(height=220 if compact else "content", key=prefix + "chat_messages"):
        if not history:
            st.markdown('<div class="chat-welcome"><div class="welcome-mark">✧</div><strong>Mình có thể giúp gì cho bạn?</strong><p>Bạn có thể hỏi “HbA1c có ý nghĩa gì?”, “Nên chuẩn bị gì khi tái khám?” hoặc trao đổi về báo cáo vừa phân tích.</p></div>', unsafe_allow_html=True)
        _render_messages(db, history, prefix)
    if compact:
        st.caption("Thông tin tham khảo · Không dùng để tự chẩn đoán hoặc đổi thuốc.")
    prompt = st.chat_input("Hỏi về kết quả, ăn uống, vận động, theo dõi hoặc chăm sóc...", max_chars=2000, key=prefix + "consultation_prompt")
    if prompt:
        try:
            if not cid:
                cid = db.create_conversation(prompt)
                st.session_state.chat_id = cid
            with st.spinner("Đang kiểm tra thông tin và soạn câu trả lời có nguồn..."):
                ChatCoordinator(db).run(cid, prompt, report)
            st.rerun()
        except ModelUnavailable as exc:
            st.error(str(exc))
            st.caption("Chưa có phản hồi từ mô hình. Bạn có thể gửi lại câu hỏi.")
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Chưa hoàn tất hoặc lưu được cuộc tư vấn ({type(exc).__name__}). Vui lòng thử lại.")

def _render_messages(db, history, prefix):
    for row in history:
        with st.chat_message("user"):
            st.write(row["question"])
        response = json.loads(row["response"])
        with st.chat_message("assistant"):
            st.write(response["answer"])
            if response.get("follow_up"):
                st.write(response["follow_up"])
            if response.get("mode") == "llm_rag":
                st.caption(f"Sinh bởi {response['model']} · trình bày {response['style']}")
            elif response.get("mode") == "safety_rule":
                st.caption("Phản hồi khẩn cấp từ Rule an toàn; không chờ mô hình sinh.")
            else:
                st.caption("Chưa có câu trả lời LLM đủ điều kiện; xem lý do ở trên.")
            with st.expander("Nguồn và hoạt động của các Agent"):
                for i, source in enumerate(response.get("sources", []), 1):
                    url = source["url"]
                    if url.startswith("https://"):
                        st.markdown(f"[{i}] [{source['title']}]({url})")
                    else:
                        st.write(f"[{i}] {source['title']} — {url}")
                    st.caption(source["text"])
                st.json(response.get("trace", []))
            if row["reward"] is not None:
                st.caption("Đã ghi nhận phản hồi cho câu trả lời này.")
            else:
                columns = st.columns(2)
                for column, label, reward in ((columns[0], "Hữu ích", 1), (columns[1], "Chưa hữu ích", -1)):
                    if column.button(label, key=f"{prefix}chat_vote_{row['id']}_{reward}"):
                        db.reward_turn(row["id"], reward)
                        st.rerun()

def render_assistant_widget(factory):
    with st.container(key="ai_launcher"):
        popup = st.popover("Trợ lý AI", icon=":material/chat_bubble:", key="assistant_popup",
                           on_change="rerun", help="Mở trợ lý để trao đổi ngay trên trang hiện tại")
        if popup.open:
            with popup:
                render_chat(factory, compact=True)

def render_knowledge(factory):
    section("Bằng chứng & nguồn", "Kho tri thức có thể kiểm tra", "Quản lý tài liệu tham khảo, bổ sung kiến thức từ sách hoặc nguồn y khoa và giữ lại thông tin xuất xứ.")
    db = get_db(factory)
    if db is None:
        return
    if not db.document_list():
        seed_knowledge(db)
    st.dataframe(db.document_list(), hide_index=True)
    st.caption("Tài liệu được chia thành các đoạn chồng lấn, lưu nguồn trong MySQL và truy xuất khi chat. PDF quét ảnh cần chuyển thành văn bản trước.")
    if st.button("Đồng bộ tài liệu KB đi kèm"):
        seed_knowledge(db)
        st.success("Đã đồng bộ; tài liệu trùng được bỏ qua.")
        st.rerun()
    with st.expander("Nạp sách / tài liệu PDF, TXT, Markdown"):
        uploaded = st.file_uploader("Tệp tài liệu", type=["pdf", "txt", "md"], key="kb_upload")
        title = st.text_input("Tên tài liệu", key="kb_title")
        source = st.text_input("Nguồn / tên sách, chương, URL", key="kb_source")
        reviewed = st.checkbox("Tôi đã kiểm tra nội dung và nguồn tài liệu", key="kb_reviewed")
        if st.button("Nạp tài liệu vào MySQL"):
            try:
                if not uploaded or not reviewed:
                    raise ValueError("Cần chọn tài liệu và xác nhận đã kiểm tra nguồn.")
                content = read_upload(uploaded.name, uploaded.getvalue())
                doc_id = ingest_text(db, content, title, source)
                st.success(f"Đã nạp tài liệu #{doc_id}.")
            except Exception as exc:
                st.error(str(exc) if isinstance(exc, ValueError) else "Không đọc hoặc lưu được tài liệu.")
    with st.expander("Nạp trang web y khoa"):
        url = st.text_input("URL HTTPS của NIDDK, CDC, ADA hoặc WHO", key="kb_web_url")
        web_title = st.text_input("Tên trang tài liệu", key="kb_web_title")
        if st.button("Đọc trang web để xem trước"):
            try:
                text, final_url = fetch_web(url)
                st.session_state.web_preview = {"text": text, "url": final_url, "input_url": url}
            except Exception as exc:
                st.error(str(exc) if isinstance(exc, ValueError) else "Không tải được trang web. Kiểm tra kết nối.")
        preview = st.session_state.get("web_preview")
        if preview and preview["input_url"] == url:
            st.text_area("Nội dung trích xuất", preview["text"], height=180, disabled=True)
            if st.button("Lưu trang đã xem vào KB"):
                try:
                    doc_id = ingest_text(db, preview["text"], web_title, preview["url"])
                    st.success(f"Đã lưu tài liệu #{doc_id}.")
                except ValueError as exc:
                    st.error(str(exc))

def render_training(factory):
    import pandas as pd
    from .learning import train
    section("Không gian nghiên cứu", "Đánh giá mô hình bằng dữ liệu", "Xem nguồn dữ liệu, kết quả kiểm tra độc lập và quá trình học từ phản hồi. Các số liệu dưới đây phản ánh nhãn của dataset.")
    db = get_db(factory)
    if db is None:
        return
    metrics_path = ARTIFACTS / "metrics.json"
    if metrics_path.exists():
        metadata = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metadata.get("synthetic"):
            st.warning("Mô hình hiện tại dùng dữ liệu mô phỏng.")
        st.write(metadata.get("target_description", ""))
        cols = st.columns(4)
        for column, key, label in zip(cols, ("accuracy", "roc_auc", "sensitivity", "specificity"),
                                      ("Accuracy", "ROC-AUC", "Sensitivity", "Specificity")):
            column.metric(label, f"{metadata.get(key, 0):.3f}")
        if metadata.get("confusion_matrix"):
            st.dataframe(pd.DataFrame(metadata["confusion_matrix"], index=["Thực tế 0", "Thực tế 1"], columns=["Dự đoán 0", "Dự đoán 1"]))
        st.caption("Chỉ số đánh giá mô hình theo nhãn dataset; không phải hiệu quả chẩn đoán đã được kiểm định lâm sàng.")
        with st.expander("Nguồn dữ liệu, cách chia tập và metrics đầy đủ"):
            st.json(metadata)
        st.download_button("Tải báo cáo mô hình", json.dumps(metadata, indent=2, ensure_ascii=False), "model_metrics.json", "application/json")
    datasets = db.dataset_list()
    st.dataframe(datasets, hide_index=True)
    with st.expander("Nạp CSV huấn luyện vào MySQL"):
        file = st.file_uploader("CSV chỉ số và nhãn outcome 0/1", type=["csv"], key="training_upload")
        name = st.text_input("Tên dataset", "clinical", key="training_name")
        target = st.text_input("Nhãn outcome được xác định như thế nào?", key="training_target")
        synthetic = st.checkbox("Đây là dữ liệu mô phỏng", key="training_synthetic")
        if st.button("Lưu dataset vào MySQL"):
            try:
                if not file or not name or len(name) > 64 or not target:
                    raise ValueError("Cần CSV, tên dataset tối đa 64 ký tự và mô tả nhãn.")
                frame = pd.read_csv(file)
                if not set(FEATURES + ["outcome"]).issubset(frame.columns):
                    raise ValueError("CSV cần " + ", ".join(FEATURES + ["outcome"]))
                if name in [d["dataset"] for d in datasets]:
                    raise ValueError("Tên dataset đã tồn tại; chọn tên mới để giữ nguyên dữ liệu cũ.")
                db.import_samples(frame, name, {"synthetic": synthetic, "target_description": target})
                st.success("Đã lưu vào MySQL.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc) if isinstance(exc, ValueError) else "Không đọc hoặc lưu được CSV.")
    names = [row["dataset"] for row in datasets]
    if names:
        train_name = st.selectbox("Dataset huấn luyện", names, key="model_train_dataset")
        test_name = st.selectbox("Dataset kiểm tra ngoài (tùy chọn)", ["Chia holdout từ tập train"] +
            [name for name in names if name != train_name], key="model_test_dataset")
        if st.button("Huấn luyện MLP từ MySQL"):
            try:
                with st.spinner("Đang huấn luyện và đánh giá..."):
                    frame = db.load_samples(train_name)
                    external = None if test_name == "Chia holdout từ tập train" else db.load_samples(test_name)
                    manifest = db.dataset_metadata(train_name)
                    if external is not None:
                        manifest["test_dataset"] = db.dataset_metadata(test_name)
                    metrics = train(frame=frame, external_test=external, dataset_metadata=manifest)
                    db.save_model_metrics(metrics)
                st.success("Đã huấn luyện, thay mô hình đang dùng và lưu metrics vào MySQL.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc) if isinstance(exc, ValueError) else "Huấn luyện chưa thành công; mô hình cũ được giữ nếu lỗi xảy ra trước khi xuất model.")
    with st.expander("Reinforcement Learning — Q-learning"):
        st.write("State: chủ đề câu hỏi + khoảng xét nghiệm. Action: ngắn / từng bước / chi tiết. Reward: hữu ích +1 hoặc chưa hữu ích −1. α=0.2, γ=0.8, ε=0.1. Chuyển trạng thái được ghi từ lượt hỏi kế tiếp. Chính sách học cách trình bày, không lựa chọn thuốc hoặc thay ngưỡng bệnh.")
        st.dataframe(db.qtable(), hide_index=True)
