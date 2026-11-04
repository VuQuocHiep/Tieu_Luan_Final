"""Shared visual language for the clinical research workspace."""
from html import escape
import streamlit as st

def apply_design():
    st.markdown("""<style>
    :root { --ink:#122f45; --muted:#63798c; --teal:#087f78; --line:#e1e9ef; }
    .stApp {background:#f4f7fa;color:var(--ink);}
    .block-container {max-width:1440px;padding:4.5rem 2.6rem 7rem;}
    [data-testid="stHeader"] {background:rgba(244,247,250,.94);}
    h1,h2,h3,h4 {color:var(--ink);letter-spacing:-.035em;}
    h1 {font-weight:750!important;} h3 {font-size:1.28rem!important;}
    p,li,label {line-height:1.65;} [data-testid="stCaptionContainer"] {color:var(--muted);}
    [data-testid="stSidebar"] {background:#102c40;border-right:0;}
    [data-testid="stSidebar"] [data-testid="stSidebarContent"] {padding:1rem .35rem;}
    [data-testid="stSidebar"] p {color:#c6d8e2;}
    .brand {display:flex;gap:12px;align-items:center;margin-bottom:25px;}
    .brand-mark {width:43px;height:43px;border-radius:14px;background:#0d8e85;color:white;display:grid;place-items:center;font-size:27px;font-weight:700;box-shadow:0 6px 18px #001c2820;}
    .brand-name {font-size:22px;line-height:1.15;font-weight:750;color:var(--ink);letter-spacing:-.7px;}
    .brand-sub {font-size:10px;color:var(--muted);letter-spacing:1.8px;margin-top:5px;text-transform:uppercase;}
    .side-brand .brand-name {color:#f3fbff;} .side-brand .brand-sub {color:#8baaba;}
    .side-label {color:#7ea0b3;font-size:10px;font-weight:700;letter-spacing:1.9px;text-transform:uppercase;margin:26px 0 13px;}
    .side-row {padding:12px 0;border-bottom:1px solid #ffffff10;display:flex;gap:12px;align-items:center;color:#d1e1e9;font-size:13px;}
    .side-num {font-size:11px;color:#62c5b9;border:1px solid #416272;border-radius:9px;padding:3px 6px;}
    .side-note {padding:15px;border-radius:14px;border:1px solid #335365;background:#17384c;font-size:12px;line-height:1.8;color:#b6ccd9;margin-top:26px;}
    .side-status {font-size:12px;padding:9px 0;color:#b4cad7;display:flex;gap:9px;align-items:center;}
    .status-dot {width:7px;height:7px;display:inline-block;border-radius:50%;background:#587383;flex:none;}
    .status-dot.done {background:#5ed1bb;}
    .topline {display:flex;align-items:center;justify-content:space-between;gap:18px;margin-bottom:22px;}
    .topline .brand {margin:0;} .tag {font-size:11px;border:1px solid #cfdfdf;border-radius:30px;padding:7px 12px;color:#407273;background:#eaf5f2;}
    .hero {position:relative;overflow:hidden;background:#102f46;border-radius:23px;padding:33px 38px;margin:0 0 20px;min-height:218px;}
    .hero:before {content:'';position:absolute;inset:0;background-image:radial-gradient(#ffffff16 1px,transparent 1px);background-size:18px 18px;mask-image:linear-gradient(90deg,transparent,#000);}
    .hero-copy {position:relative;z-index:1;max-width:72%;}
    .hero-eyebrow {font-size:10px;letter-spacing:2.3px;color:#76d6c8;font-weight:700;}
    .hero h1 {font-size:clamp(25px,2.6vw,38px)!important;color:#fff;margin:11px 0 9px;line-height:1.3;padding:0;}
    .hero p {color:#c3d8e4;font-size:14px;max-width:620px;margin:0;line-height:1.85;}
    .hero-svg {position:absolute;width:240px;height:180px;right:18px;top:24px;opacity:.82;}
    .journey {display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:24px;}
    .journey-item {background:#fff;border:1px solid var(--line);border-radius:15px;padding:15px 18px;display:flex;gap:13px;align-items:center;}
    .journey-n {color:#0b827b;background:#edf7f4;padding:7px 9px;border-radius:10px;font-size:12px;font-weight:750;}
    .journey-item strong {font-size:13px;color:#203e52;} .journey-item small {display:block;color:var(--muted);font-size:11px;margin-top:3px;}
    [data-testid="stTabs"] [data-baseweb="tab-list"] {gap:5px;border-bottom:1px solid #dce5ed;padding:0 0 12px;overflow-x:auto;}
    [data-testid="stTabs"] [data-baseweb="tab"] {height:39px;border-radius:10px;padding:0 13px;font-size:12px;color:#5e7385;white-space:nowrap;}
    [data-testid="stTabs"] [aria-selected="true"] {color:white!important;background:#087f78;}
    [data-testid="stTabs"] [role="tab"] {border-radius:10px!important;}
    [data-testid="stTabs"] [data-baseweb="tab-highlight"], [data-testid="stTabs"] [data-baseweb="tab-border"] {display:none;}
    [data-testid="stTabContent"] {padding-top:23px;}
    .section-heading {margin:0 0 20px;}
    .section-heading .eyebrow {color:#0b827b;font-size:10px;letter-spacing:1.8px;text-transform:uppercase;font-weight:750;}
    .section-heading h2 {font-size:23px;margin:5px 0 6px;font-weight:750;padding:0;}
    .section-heading p {font-size:13px;color:var(--muted);margin:0;max-width:820px;}
    [data-testid="stForm"], [data-testid="stVerticalBlockBorderWrapper"]>div {border-radius:17px;}
    [data-testid="stForm"] {background:white;border:1px solid var(--line);padding:22px;}
    [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input {font-size:14px;}
    [data-baseweb="input"], [data-baseweb="select"]>div {border-radius:10px!important;}
    [data-testid="stButton"] button, [data-testid="stDownloadButton"] button, [data-testid="stFormSubmitButton"] button {border-radius:10px;min-height:40px;font-size:13px;font-weight:600;transition:box-shadow .18s,transform .18s;}
    [data-testid="stButton"] button:hover, [data-testid="stDownloadButton"] button:hover {border-color:#087f78;color:#087f78;box-shadow:0 3px 12px #1233470d;}
    [data-testid="stBaseButton-primaryFormSubmit"], [data-testid="stBaseButton-primary"] {background:#087f78!important;border-color:#087f78!important;color:white!important;}
    [data-testid="stFileUploaderDropzone"] {background:#f0f7f8;border:1px dashed #9dbfc6;border-radius:14px;padding:22px 16px;}
    [data-testid="stAlert"] {border-radius:12px;font-size:13px;}
    [data-testid="stMetric"] {background:white;border:1px solid var(--line);border-radius:15px;padding:17px 19px;}
    [data-testid="stMetricLabel"] {color:#647c8d;font-size:12px;}
    [data-testid="stMetricValue"] {color:#173c50;font-size:30px;font-weight:750;letter-spacing:-1px;}
    [data-testid="stDataFrame"] {border-radius:13px;overflow:hidden;}
    [data-testid="stExpander"] {background:#fff;border-radius:12px;}
    [data-testid="stChatMessage"] {border:1px solid #e1eaf0;border-radius:16px;background:#fff;padding:18px;}
    [data-testid="stChatMessage"] p {font-size:14px;}
    [data-testid="stChatInput"] {border-radius:15px;}
    [data-testid="stImage"] img {border:1px solid #dfe8f0;border-radius:14px;}
    .preview-empty {min-height:330px;background:linear-gradient(145deg,#edf6f6,#f9fbfd);border:1px dashed #bdd3da;border-radius:17px;display:flex;align-items:center;justify-content:center;flex-direction:column;text-align:center;padding:35px;}
    .preview-empty .empty-icon {font-size:37px;width:79px;height:79px;display:grid;place-items:center;background:#fff;border:1px solid #d8e7e9;border-radius:22px;color:#168b83;margin-bottom:20px;}
    .preview-empty strong {font-size:17px;letter-spacing:-.4px;color:#26485b;}
    .preview-empty p {font-size:12px;color:#6a8392;max-width:280px;margin:9px auto;}
    .result-banner {border:1px solid #c5e0d8;border-left:4px solid #0c8a77;background:#eff8f4;border-radius:15px;padding:21px 24px;margin:25px 0 20px;}
    .result-banner.amber {background:#fff9ec;border-color:#f1dbac;border-left-color:#cf9024;}
    .result-banner.rose {background:#fff3f1;border-color:#f1d5ce;border-left-color:#d66d58;}
    .result-banner small {text-transform:uppercase;letter-spacing:1.5px;font-size:10px;font-weight:750;color:#657a83;}
    .result-banner h3 {margin:7px 0 4px;padding:0;}
    .result-banner p {font-size:12px;color:#637786;margin:0;}
    .research-note {font-size:11px;color:#7c8e9a;margin-top:26px;border-top:1px solid #dce6ed;padding-top:16px;}
    .st-key-ai_launcher {position:fixed!important;right:25px;bottom:25px;width:auto!important;z-index:999990;}
    .st-key-ai_launcher [data-testid="stPopoverButton"] {background:#087f78!important;color:#fff!important;border:1px solid #19a99b!important;border-radius:50px!important;min-height:57px;padding:0 22px;box-shadow:0 8px 28px #075b5b45;font-weight:700;font-size:14px;}
    .st-key-ai_launcher [data-testid="stPopoverButton"]:hover {background:#066b66!important;transform:translateY(-2px);}
    [data-testid="stPopoverBody"] {width:430px!important;max-width:calc(100vw - 28px)!important;max-height:calc(100dvh - 110px)!important;overflow:auto;border-radius:20px!important;padding:20px!important;box-shadow:0 18px 75px #0a304533!important;border:1px solid #d9e7ed!important;}
    .assistant-head {display:flex;gap:12px;align-items:center;padding-bottom:13px;margin-bottom:12px;border-bottom:1px solid #e2ebef;}
    .assistant-avatar {width:43px;height:43px;border-radius:14px;background:#e3f4ee;color:#078276;font-size:24px;display:grid;place-items:center;}
    .assistant-head strong {font-size:16px;color:#173d50;display:block;}
    .assistant-head small {font-size:11px;color:#78909d;}
    .chat-welcome {text-align:center;padding:31px 18px;color:#547285;}
    .chat-welcome .welcome-mark {font-size:35px;color:#0a8b80;margin-bottom:10px;}
    .chat-welcome strong {display:block;font-size:17px;color:#264a5e;margin-bottom:10px;}
    .chat-welcome p {font-size:12px;line-height:1.8;}
    @media (max-width:900px) {.block-container {padding:4.3rem 1rem 7rem;}.hero {padding:26px;}.hero-copy {max-width:100%;}.hero-svg {opacity:.13;right:0;}.journey {gap:7px;}.journey-item {padding:12px 9px;}.journey-item small {display:none;}.journey-n {padding:5px;}.topline .tag {display:none;}}
    @media (max-width:550px) {.hero h1 {font-size:27px!important;}.hero p {font-size:12px;}.journey {grid-template-columns:1fr;}.journey-item {padding:9px 13px;}.journey-item small {display:inline;margin-left:7px;}.st-key-ai_launcher {right:15px;bottom:17px;}.st-key-ai_launcher [data-testid="stPopoverButton"] {min-height:52px;padding:0 17px;}}
    @media (prefers-reduced-motion:reduce) {* {transition:none!important;}}
    </style>""", unsafe_allow_html=True)

def hero():
    st.markdown("""<div class="topline"><div class="brand"><div class="brand-mark">+</div><div><div class="brand-name">GlucoCare<span style="color:#0b8b80">.</span></div><div class="brand-sub">Diabetes research workspace</div></div></div><span class="tag">Nghiên cứu & giáo dục sức khỏe</span></div>
    <div class="hero"><div class="hero-copy"><div class="hero-eyebrow">MULTI-AGENT DIABETES ASSISTANT</div><h1>Hiểu chỉ số.<br>Chủ động sức khỏe.</h1><p>Đọc bảng xét nghiệm, khám phá kết quả và trao đổi cùng trợ lý AI<br>dựa trên dữ liệu và nguồn kiến thức có thể kiểm tra.</p></div>
    <svg class="hero-svg" viewBox="0 0 260 190" fill="none" aria-hidden="true"><circle cx="139" cy="94" r="75" stroke="#438385" stroke-dasharray="3 8"/><circle cx="139" cy="94" r="55" stroke="#548e91" opacity=".5"/><path d="M10 104H63L78 83 98 129 121 51 145 115 161 94H245" stroke="#6ce4cc" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/><circle cx="139" cy="19" r="5" fill="#68d5c6"/><circle cx="210" cy="115" r="5" fill="#68d5c6"/><circle cx="93" cy="155" r="4" fill="#98bcd2"/><path d="M207 26v16m-8-8h16M39 143v12m-6-6h12" stroke="#80a9b8" stroke-width="2" stroke-linecap="round"/></svg></div>
    <div class="journey"><div class="journey-item"><span class="journey-n">01</span><div><strong>Chuẩn bị dữ liệu</strong><small>Tạo bảng hoặc tải ảnh xét nghiệm</small></div></div><div class="journey-item"><span class="journey-n">02</span><div><strong>Khám phá chỉ số</strong><small>Đối chiếu và xem kết quả phân tích</small></div></div><div class="journey-item"><span class="journey-n">03</span><div><strong>Trao đổi cùng AI</strong><small>Hỏi đáp có ngữ cảnh và dẫn nguồn</small></div></div></div>""", unsafe_allow_html=True)

def section(eyebrow, title, description=""):
    st.markdown(f'<div class="section-heading"><div class="eyebrow">{escape(eyebrow)}</div><h2>{escape(title)}</h2><p>{escape(description)}</p></div>', unsafe_allow_html=True)

def empty_state(title, description, icon="▤"):
    st.markdown(f'<div class="preview-empty"><div class="empty-icon">{escape(icon)}</div><strong>{escape(title)}</strong><p>{escape(description)}</p></div>', unsafe_allow_html=True)

def result_banner(rule):
    color = {1: "amber", 2: "rose"}.get(rule["level"], "" if rule["level"] == 0 else "amber")
    st.markdown(f'<div class="result-banner {color}"><small>Kết quả sàng lọc · cần xác nhận lâm sàng</small><h3>{escape(rule["label"])}</h3><p>Diễn giải dựa trên những chỉ số bạn đã đối chiếu. Đây không phải kết luận chẩn đoán.</p></div>', unsafe_allow_html=True)

def sidebar():
    with st.sidebar:
        st.markdown('''<div class="brand side-brand"><div class="brand-mark">+</div><div><div class="brand-name">GlucoCare.</div><div class="brand-sub">Multi-agent workspace</div></div></div><div class="side-label">Lộ trình của bạn</div><div class="side-row"><span class="side-num">01</span>Chuẩn bị ảnh xét nghiệm</div><div class="side-row"><span class="side-num">02</span>Đối chiếu & phân tích</div><div class="side-row"><span class="side-num">03</span>Trao đổi cùng trợ lý</div><div class="side-label">Phiên làm việc</div>''', unsafe_allow_html=True)
        for label, key in [("Ảnh bảng đã tạo", "generated_image"), ("Đã nhận dạng chỉ số", "ocr"), ("Báo cáo phân tích", "result")]:
            active = key in st.session_state
            st.markdown(f'<div class="side-status"><span class="status-dot {"done" if active else ""}"></span>{escape(label)} · {"Có" if active else "Chưa có"}</div>', unsafe_allow_html=True)
        st.markdown('''<div class="side-note"><strong style="color:#e8f3f7">Thông tin có cơ sở</strong><br>Luôn kiểm tra chỉ số và nguồn tham khảo. Kết quả cần được nhân viên y tế đánh giá.<br><br>Dùng các tab phía trên để mở kho tri thức, mô hình và lịch sử.</div>''', unsafe_allow_html=True)
