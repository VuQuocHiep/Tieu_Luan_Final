"""Home dashboard; three independent pages live in pages/."""
import streamlit as st
from shared.config import CONFIG,ART,METHODS
from shared.ui import theme,hero

st.set_page_config(page_title='CNN Vision Lab | Machine Learning',page_icon='🧠',layout='wide')
theme()
hero('CNN Vision Lab · 3 ứng dụng',
     'Thực nghiệm mạng nơ-ron tích chập trên 3 dataset Kaggle với ba cách triển khai: NumPy From Scratch, Keras và PyTorch.',
     'CHƯƠNG 3 · DEEP LEARNING / COMPUTER VISION')
cols=st.columns(3,gap='large')
links=['pages/01_MNIST.py','pages/02_Fashion_MNIST.py','pages/03_Sign_Language.py']
for (key,spec),col,link in zip(CONFIG.items(),cols,links):
    with col:
        st.markdown(f'<div class="appcard"><div style="font-size:2.6rem;margin-bottom:10px">{spec["emoji"]}</div><h3>{spec["short"]}</h3><p>{spec["title"]}<br>{spec["n_train"]:,} train · {spec["n_test"]:,} test · 28×28</p></div>',unsafe_allow_html=True)
        st.page_link(link,label=f'Mở ứng dụng {spec["short"]} →',use_container_width=True)
st.write('')
a,b,c=st.columns(3)
with a:st.metric('Nguồn dữ liệu','3 Kaggle datasets')
with b:st.metric('CNN implementations','9 models')
with c:
    ready=sum((ART/f'{k}_{m}.json').exists() for k in CONFIG for m in METHODS)
    st.metric('Mô hình đã huấn luyện',f'{ready}/9')
st.markdown('### Bắt đầu sử dụng')
st.code('python download_kaggle.py\npython train_all.py\npython tests/test_project.py\nstreamlit run app.py',language='bash')
st.info('Train/Validation/Test: tập test chính thức từ Kaggle được giữ riêng để đánh giá; các mô hình cài đặt cùng kiến trúc CNN 3×3 → ReLU → MaxPool → Dense.')
