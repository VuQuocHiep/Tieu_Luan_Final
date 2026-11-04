import hashlib
import json
import streamlit as st
from diabetes.schema import Patient
from diabetes.imaging import render_table, OCRAgent
from diabetes.agents import Coordinator
from diabetes.storage import Database
from diabetes.ui import render_chat, render_knowledge, render_training, render_assistant_widget
from diabetes.design import apply_design, hero, section, empty_state, result_banner, sidebar

st.set_page_config(page_title="Diabetes · Multi Agent", page_icon="🧬", layout="wide")
apply_design()
hero()

@st.cache_resource
def ocr_engine():
    return OCRAgent()

@st.cache_resource
def database():
    return Database()

def show_result(result):
    result_banner(result["rule"])
    evidence_panel, model_panel = st.columns([1.2, 1], gap="large")
    with evidence_panel:
        st.subheader("Cơ sở diễn giải")
        if result["rule"]["evidence"]:
            import pandas as pd
            evidence = pd.DataFrame(result["rule"]["evidence"]).rename(columns={"test": "Xét nghiệm", "value": "Giá trị", "unit": "Đơn vị", "level": "Khoảng phân loại"})
            evidence["Khoảng phân loại"] = evidence["Khoảng phân loại"].map({0: "Tham chiếu", 1: "Tiền đái tháo đường", 2: "Ngưỡng nghi ngờ"})
            st.dataframe(evidence, hide_index=True, width="stretch")
        for note in result["rule"]["notes"]:
            st.caption("• " + note)
    prediction = result["prediction"]
    with model_panel:
        st.subheader("Góc nhìn từ mô hình")
        if prediction["available"]:
            if prediction["synthetic"]:
                st.warning("Mô hình học trên dữ liệu MÔ PHỎNG, không có giá trị đánh giá lâm sàng.")
            st.metric("Điểm mạng nơ-ron · chưa hiệu chuẩn", f"{prediction['score']:.3f}")
            st.caption(f"Phân loại mô hình: nhãn {prediction['class']} · ngưỡng 0.5")
            st.caption(prediction["note"])
            with st.expander("Ý nghĩa nhãn của mô hình"):
                st.write(prediction.get("target_description", ""))
            if prediction.get("outside_training_range"):
                st.warning("Có chỉ số nằm ngoài khoảng dữ liệu huấn luyện: " + ", ".join(prediction["outside_training_range"]))
        else:
            st.caption("Deep Learning: " + prediction["reason"])
    st.subheader("Thông tin tham khảo ban đầu từ KB")
    st.write(result["rag"]["answer"])
    st.info("Bấm Trợ lý AI ở góc dưới bên phải để trao đổi thêm về báo cáo này.")
    for i, source in enumerate(result["rag"]["sources"], 1):
        st.markdown(f"[{i}] [{source['title']}]({source['url']})")
    with st.expander("Luồng Agent và báo cáo"):
        st.json(result)
    st.download_button("Tải báo cáo JSON", json.dumps(result, indent=2, ensure_ascii=False), "analysis.json", "application/json")
    if "run_id" in result:
        st.caption("Phản hồi điều chỉnh độ chi tiết tư vấn; không thay đổi ngưỡng y khoa.")
        left, right = st.columns(2)
        for column, label, reward in ((left, "Hữu ích", 1), (right, "Chưa hữu ích", 0)):
            if column.button(label):
                try:
                    database().feedback(result["run_id"], reward)
                    st.success("Đã lưu phản hồi trong MySQL.")
                except Exception:
                    st.error("Không lưu được phản hồi. Kiểm tra MySQL.")

tab_generate, tab_analyze, tab_chat, tab_kb, tab_train, tab_arch, tab_history = st.tabs([
    "01 · Tạo ảnh bảng", "02 · Phân tích ảnh", "03 · Trợ lý AI", "Kho tri thức", "Mô hình & RL", "Kiến trúc", "Lịch sử"])
with tab_generate:
    section("Bước 01 / Chuẩn bị", "Bắt đầu từ những chỉ số của bạn", "Tạo ảnh bảng chỉ số để trải nghiệm quy trình, hoặc chuyển sang Phân tích ảnh nếu đã có phiếu xét nghiệm.")
    form_panel, preview_panel = st.columns([1, 1.13], gap="large")
    with form_panel:
        with st.form("generate"):
            st.subheader("Thông tin xét nghiệm")
            st.caption("Nhập chỉ số và đơn vị tương ứng. Ảnh tạo ra luôn có nhãn minh họa.")
            c1, c2 = st.columns(2)
            glucose = c1.number_input("Glucose đói (mg/dL)", 20., 1000., 145.)
            hba1c = c2.number_input("HbA1c (%)", 2., 25., 7.1)
            age = c1.number_input("Tuổi", 18, 120, 45)
            bmi = c2.number_input("BMI (kg/m²)", 8., 100., 27.)
            st.caption("Glucose đói và HbA1c là hai xét nghiệm khác nhau. Đối chiếu đúng tên và đơn vị trên phiếu gốc.")
            create = st.form_submit_button("Sinh ảnh bảng xét nghiệm", type="primary", width="stretch")
        if create:
            st.session_state.generated_image = render_table(Patient(glucose, hba1c, age, bmi))
    with preview_panel:
        if "generated_image" in st.session_state:
            st.caption("BẢN XEM TRƯỚC · ẢNH MINH HỌA")
            st.image(st.session_state.generated_image, width="stretch")
            st.download_button("Tải ảnh PNG", st.session_state.generated_image, "lab_table_demo.png", "image/png", width="stretch")
            st.caption("Tiếp theo: mở Phân tích ảnh và chọn Dùng ảnh vừa sinh.")
        else:
            empty_state("Ảnh bảng sẽ xuất hiện tại đây", "Điền các chỉ số ở bên trái và chọn Sinh ảnh bảng xét nghiệm. Bạn có thể tải ảnh về sau khi tạo.")

with tab_analyze:
    section("Bước 02 / Khám phá", "Từ hình ảnh đến thông tin rõ ràng", "Tải ảnh bảng xét nghiệm, đọc chỉ số và kiểm tra lại trước khi phân tích. Các trường chưa nhận dạng được sẽ để trống.")
    source_panel, confirmation_panel = st.columns([1, 1.12], gap="large")
    with source_panel:
        st.subheader("01 / Ảnh đầu vào")
        uploaded = st.file_uploader("Tải ảnh bảng xét nghiệm (PNG / JPG)", type=["png", "jpg", "jpeg"])
        use_demo = st.checkbox("Dùng ảnh vừa sinh", disabled="generated_image" not in st.session_state)
        content = st.session_state.get("generated_image") if use_demo else (uploaded.getvalue() if uploaded else None)
        if content is None and st.session_state.get("image_digest"):
            for key in ("image_digest", "ocr", "result", "edit_glucose", "edit_a1c", "edit_age", "edit_bmi", "fasting", "pregnant", "confirmed"):
                st.session_state.pop(key, None)
        if content:
            digest = hashlib.sha256(content).hexdigest()
            if digest != st.session_state.get("image_digest"):
                st.session_state.image_digest = digest
                for key in ("ocr", "result", "edit_glucose", "edit_a1c", "edit_age", "edit_bmi", "fasting", "pregnant", "confirmed"):
                    st.session_state.pop(key, None)
            st.image(content, width="stretch")
            if st.button("Đọc chỉ số bằng OCR", width="stretch"):
                try:
                    with st.spinner("Đang nhận dạng chữ từ pixel ảnh..."):
                        st.session_state.ocr = ocr_engine().read(content)
                    for key in ("edit_glucose", "edit_a1c", "edit_age", "edit_bmi", "confirmed"):
                        st.session_state.pop(key, None)
                except Exception as exc:
                    st.error(f"OCR không thực hiện được: {type(exc).__name__}. Có thể nhập chỉ số thủ công bên dưới.")
        else:
            st.caption("Ảnh nên rõ nét, nhìn đủ tên xét nghiệm, kết quả và đơn vị. Chỉ dùng ảnh bảng chỉ số xét nghiệm.")
    extracted = st.session_state.get("ocr", {})
    if extracted:
        with source_panel, st.expander("Văn bản OCR — hãy đối chiếu với ảnh", expanded=False):
            st.code(extracted["text"] or "Không đọc được văn bản.")
            for warning in extracted["warnings"]:
                st.warning(warning)
    values = extracted.get("values", {})
    with confirmation_panel, st.form("confirm"):
        st.subheader("02 / Xác nhận chỉ số")
        st.caption("Glucose nhập bằng mg/dL; giá trị OCR từ mmol/L được quy đổi tự động.")
        c1, c2 = st.columns(2)
        g = c1.number_input("Glucose đói (mg/dL)", 20., 1000., value=values.get("fasting_glucose"), key="edit_glucose")
        a = c2.number_input("HbA1c (%)", 2., 25., value=values.get("hba1c"), key="edit_a1c")
        age_input = c1.number_input("Tuổi (bỏ trống nếu chưa rõ)", 1., 120., value=values.get("age"), key="edit_age")
        bmi_input = c2.number_input("BMI (bỏ trống nếu chưa rõ)", 8., 100., value=values.get("bmi"), key="edit_bmi")
        fasting = st.checkbox("Xác nhận xét nghiệm glucose sau nhịn ăn ít nhất 8 giờ", key="fasting")
        pregnant = st.checkbox("Đang mang thai", key="pregnant")
        question = st.text_input("Câu hỏi cho Agent", "Giải thích kết quả và bước tiếp theo")
        confirmed = st.checkbox("Tôi đã đối chiếu chỉ số và đơn vị với phiếu gốc", key="confirmed")
        persist = st.checkbox("Lưu chỉ số và báo cáo vào MySQL", value=True)
        submit = st.form_submit_button("Phân tích với các Agent", type="primary", width="stretch")
    if submit:
        st.session_state.pop("result", None)
        if not confirmed:
            st.warning("Cần đối chiếu và xác nhận chỉ số trước khi phân tích.")
        else:
            try:
                patient = Patient(g, a, age_input, bmi_input, fasting, pregnant).validate()
                st.session_state.result = Coordinator(database()).run(patient, question, persist=persist)
            except ValueError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Không hoàn tất phân tích ({type(exc).__name__}). Kiểm tra MySQL và cấu hình .env.")
    if "result" in st.session_state:
        show_result(st.session_state.result)

with tab_chat:
    render_chat(database)
with tab_kb:
    render_knowledge(database)
with tab_train:
    render_training(database)

with tab_arch:
    section("Nền tảng nghiên cứu", "Các thành phần phối hợp như thế nào?", "Theo dõi luồng dữ liệu, bằng chứng và phản hồi giữa các thành phần trong hệ thống.")
    st.graphviz_chart('''digraph G {
      rankdir=LR; bgcolor="transparent";
      node [shape=box, style="rounded,filled", fillcolor="#e9f5f1", color="#a8d2c9", fontname="Arial", fontsize=11, fontcolor="#173c50", margin=".18,.12"];
      edge [color="#799ba9", arrowsize=.7];
      MySQL -> NeuralNetwork -> PredictionAgent -> Coordinator;
      BooksWeb -> KB -> KnowledgeAgent;
      KB -> RetrievalAgent -> AdvisorAgent -> ReviewAgent -> Report;
      Coordinator -> RetrievalAgent; ReviewAgent -> AdvisorAgent;
      LLM -> AdvisorAgent; ChatHistory -> Coordinator;
      Image -> OCRAgent -> HumanConfirmation -> Coordinator;
      RuleAgent -> Coordinator; Feedback -> QLearning -> Coordinator;
      Coordinator -> Report; Report -> MySQL; Report -> Feedback;
    }''')
    st.write("MLP 4→32→16→1. RAG truy xuất kết hợp TF-IDF từ/cụm từ và ký tự → Ollama sinh trả lời → ReviewAgent kiểm tra nguồn và giới hạn. Coordinator điều phối Intake, Safety, Rule, Prediction, Retrieval, Advisor và Review; tối đa hai vòng soạn/kiểm tra. Q-learning học cách trình bày từ phản hồi, lưu trạng thái/chuyển trạng thái/reward vào MySQL.")
    st.code("python -m diabetes.cli init-db\npython -m diabetes.cli import-db data/clinical.csv --dataset clinical\npython -m diabetes.cli train --dataset clinical")
with tab_history:
    section("Hồ sơ phân tích", "Lịch sử có thể xem lại", "Mở những báo cáo đã lưu để kiểm tra kết quả và thông tin nguồn tại thời điểm phân tích.")
    if st.button("Đọc lịch sử từ MySQL"):
        try:
            for run_id, created, raw in database().recent():
                with st.expander(f"#{run_id} · {created}"):
                    st.json(json.loads(raw))
        except Exception as exc:
            st.error(f"Chưa kết nối được MySQL ({type(exc).__name__}). Xem README để khởi tạo.")

sidebar()
st.markdown('<div class="research-note">GlucoCare · Hệ đa Agent cho bệnh tiểu đường. Dành cho nghiên cứu và giáo dục; không thay thế chẩn đoán hoặc chỉ định điều trị.</div>', unsafe_allow_html=True)
render_assistant_widget(database)
