import streamlit as st
from shared.ui import theme
from shared.config import CONFIG

st.set_page_config(page_title='RNN Lab - 3 Kaggle Apps', page_icon='🧠', layout='wide')
theme()
st.markdown('''<div class="hero"><span class="pill">Chapter 4</span><span class="pill">RNN</span><span class="pill">Scratch · Keras · PyTorch</span>
<h1>RNN Lab · 3 ứng dụng NLP từ Kaggle</h1><p>Khảo sát dữ liệu, huấn luyện cùng một Simple RNN theo ba cách cài đặt, đánh giá và dự đoán trực tiếp.</p></div>''', unsafe_allow_html=True)
st.write('')
cols = st.columns(3)
for col,(key,s) in zip(cols,CONFIG.items()):
    with col:
        st.markdown(f'''<div class="card"><h3>{s['short']}</h3><p>{s['description']}</p><span class="pill">{len(s['classes'])} lớp</span></div>''', unsafe_allow_html=True)
st.info('Dùng menu Pages bên trái để mở từng ứng dụng. Trước khi chạy giao diện đầy đủ: download_kaggle.py → test_kaggle_data.py → train_all.py.')
