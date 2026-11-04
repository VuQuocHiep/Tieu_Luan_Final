"""The beautiful dashboard that links to the 3 independent Streamlit application pages."""
import streamlit as st
from shared.ui import theme, hero, stat
from shared.datasets import SOURCES, dataset
from shared.inference import ARTIFACTS

st.set_page_config(page_title='ML Lab | 3 ứng dụng Machine Learning', page_icon='🧪', layout='wide')
theme()
hero('ML LAB · 03', 'CHƯƠNG 2  /  MACHINE LEARNING',
     '3 nguồn Kaggle CSV, 3 ứng dụng chuyên biệt, 9 phiên bản ML và dashboard trực quan.')
a,b,c=st.columns(3)
with a:stat('Ứng dụng','03')
with b:stat('Mô hình đã xây dựng','09')
with c:stat('Framework','NumPy / Keras / Torch')
st.markdown('### Khám phá từng ứng dụng')
files={'iris':'pages/01_Iris.py','wine':'pages/02_Wine.py','breast_cancer':'pages/03_Breast_Cancer.py'}
cols=st.columns(3)
for col,(key,spec) in zip(cols,SOURCES.items()):
    with col:
        try:
            b=dataset(key)
            dataset_detail=f'{len(b.target)} mẫu · {b.data.shape[1]} đặc trưng · {len(b.target_names)} lớp'
        except (FileNotFoundError, ValueError) as e:
            dataset_detail='Chưa có CSV Kaggle · Chạy python download_kaggle.py'
        st.markdown(f"""<div style="background:#10243B;border:1px solid #35506C;border-radius:19px;padding:22px;min-height:176px">
          <div style="font-size:30px">{spec["icon"]}</div>
          <div style="font-size:19px;font-weight:750;margin-top:12px">{spec["vi_name"]}</div>
          <div style="font-size:13px;color:#AFCCDD;margin-top:10px">{dataset_detail}</div>
        </div>""",unsafe_allow_html=True)
        st.page_link(files[key],label='Khởi động ứng dụng ↗',icon='🚀',use_container_width=True)
st.divider()
st.markdown('### Cách hoạt động')
st.write('**Dữ liệu công khai → chia Train/Validation/Test → StandardScaler (fit trên Train) → 3 bộ huấn luyện riêng → so sánh metrics → dự đoán CSV hoặc nhập tay.**')
if not (ARTIFACTS/'metrics.csv').exists():
    st.warning('Chưa có model. Chạy `python train_all.py` từ thư mục dự án.')
else:
    st.success('Đã có đủ thông tin đánh giá trong thư mục artifacts. Mở một trong ba app để xem kết quả.')
with st.expander('Hướng dẫn chạy project'):
    st.code('pip install -r requirements.txt\npython download_kaggle.py\npython train_all.py\nstreamlit run app.py',language='bash')
