"""Presentation and live inference for 3 independent Streamlit pages."""
import os
os.environ.setdefault('KERAS_BACKEND','torch')
from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from .datasets import SOURCES, dataset
from .inference import ARTIFACTS, BACKENDS, load_metadata, load_model, predict_proba

ROOT = Path(__file__).resolve().parents[1]
COLORS = ['#6DE6BF','#80BFFF','#E7B1FF','#FFCC89']

def theme():
    st.markdown("""<style>
    .stApp {background:radial-gradient(circle at 80% -10%,#182D4E 0%,#09172A 52%,#071221 100%);}
    [data-testid="stSidebar"]{background:#0B1D34;border-right:1px solid #284360;}
    h1,h2,h3 {letter-spacing:-.028em;}
    h1{font-size:2.4rem!important;}
    .block-container{padding-top:1.5rem;max-width:1230px;}
    .hero{padding:29px 31px;border:1px solid #375576;border-radius:23px;background:linear-gradient(115deg,#173455,#101E34 75%);margin-bottom:20px;}
    .hero .eyebrow{color:#80E6C5;letter-spacing:.16em;font-size:11px;font-weight:700;}
    .hero h1{margin:.45rem 0;color:#F3F8FF;}
    .hero p{color:#BCD1E3;font-size:1rem;margin-bottom:0;line-height:1.6;}
    .stat{border:1px solid #315172;border-radius:16px;background:#10243B;padding:16px 17px;margin:4px 0 14px;}
    .stat .k{color:#9FBDD2;font-size:12px;letter-spacing:.04em;}
    .stat .v{color:#F2F7FF;font-weight:750;font-size:26px;margin-top:5px;}
    div[data-testid="stTabs"] button[role="tab"]{font-weight:650;}
    div[data-testid="stMetric"]{background:#10243B;padding:15px;border-radius:15px;border:1px solid #294662;}
    .stButton button[kind="primary"]{border:0;background:linear-gradient(90deg,#43D3AF,#40AEE3);color:#092137;font-weight:800;}
    a{color:#81DBFF!important;}
    </style>""",unsafe_allow_html=True)

def stat(k,v):
    st.markdown(f'<div class="stat"><div class="k">{k}</div><div class="v">{v}</div></div>', unsafe_allow_html=True)

def hero(title,kicker,subtitle):
    st.markdown(f'<div class="hero"><div class="eyebrow">{kicker}</div><h1>{title}</h1><p>{subtitle}</p></div>',unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def cached_model(name,backend,n_features,n_classes):
    return load_model(name,backend,n_features,n_classes)

@st.cache_resource(show_spinner=False)
def cached_metadata(name):
    return load_metadata(name)

def metrics_for(name):
    f = ARTIFACTS/f'{name}_metrics.json'
    if not f.exists():
        return []
    return json.loads(f.read_text(encoding='utf-8'))

def nice_plot(fig, height=350):
    fig.update_layout(template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
                      font=dict(color='#DCE8F5'),height=height,margin=dict(l=12,r=12,t=30,b=26),
                      legend_title_text='')
    st.plotly_chart(fig,use_container_width=True)

def show_dataset(name,raw):
    spec = SOURCES[name]
    st.markdown('#### Nguồn dữ liệu và kích thước')
    st.write(spec['description'])
    st.link_button('Mở nguồn dữ liệu chính thức ↗',spec['source'])
    col1,col2,col3 = st.columns(3)
    with col1:stat('Số mẫu',str(len(raw.target)))
    with col2:stat('Đặc trưng',str(raw.data.shape[1]))
    with col3:stat('Lớp',str(len(raw.target_names)))
    st.markdown('#### 5 mẫu dữ liệu gốc')
    frame = pd.DataFrame(raw.data[:5], columns=raw.feature_names)
    frame['Nhãn'] = [str(raw.target_names[n]) for n in raw.target[:5]]
    st.dataframe(frame,hide_index=True,use_container_width=True)
    st.markdown('#### Phân bố nhãn')
    counts = np.bincount(raw.target)
    dist = pd.DataFrame({'Lớp':[str(n) for n in raw.target_names],
                         'Số mẫu':counts,'Tỷ lệ (%)':np.round(counts/counts.sum()*100,2)})
    col1,col2=st.columns([1.4,1])
    with col1:
        fig=px.bar(dist,x='Lớp',y='Số mẫu',color='Lớp',color_discrete_sequence=COLORS,
                   text='Số mẫu')
        fig.update_traces(textposition='outside')
        nice_plot(fig)
    with col2:
        st.dataframe(dist,hide_index=True,use_container_width=True)
        st.info('Nhận xét: '+spec['insight'])

def show_predict(name,raw):
    st.markdown('#### Dự đoán trực tiếp bằng mô hình đã huấn luyện')
    st.caption('Tất cả đầu vào nhập theo thứ tự và đơn vị gốc của dataset; hệ thống tự chuẩn hóa bằng scaler trên tập Train.')
    backend = st.radio('Cách cài đặt',list(BACKENDS),format_func=lambda x:BACKENDS[x],horizontal=True,key=f'model_{name}')
    mode=st.segmented_control('Chế độ',options=['Nhập một mẫu','CSV nhiều mẫu'],default='Nhập một mẫu',key=f'mode_{name}')
    if mode == 'CSV nhiều mẫu':
        st.write('CSV cần chứa các cột đặc trưng có tên trùng với dataset (không cần cột nhãn).')
        sample_csv=pd.DataFrame(raw.data[:5],columns=raw.feature_names).to_csv(index=False).encode()
        st.download_button('Tải CSV mẫu',sample_csv,file_name=f'{name}_sample.csv',mime='text/csv',key=f'csv_{name}')
        file=st.file_uploader('Chọn tập CSV',type=['csv'],key=f'upload_{name}')
        if file is None:
            st.info('Tải tệp CSV lên để bắt đầu dự đoán hàng loạt.')
            return
        inp=pd.read_csv(file)
    else:
        index=st.selectbox('Mẫu có sẵn để thử',range(len(raw.data)),format_func=lambda i:f'Mẫu #{i+1} — {raw.target_names[raw.target[i]]}',key=f'ex_{name}')
        inp=st.data_editor(pd.DataFrame([raw.data[index]],columns=raw.feature_names),
                           hide_index=True,use_container_width=True,num_rows='fixed',key=f'editor_{name}')
    st.caption('Chỉ số xác suất phản ánh kết quả mô hình; không được hiểu là độ chắc chắn tuyệt đối.')
    if st.button('✨  Chạy dự đoán',type='primary',key=f'predict_{name}'):
        try:
            meta=cached_metadata(name)
            required=meta['features']
            missing=[c for c in required if c not in inp.columns]
            if missing:
                raise ValueError('Thiếu cột: '+', '.join(missing))
            x=inp[required].apply(pd.to_numeric,errors='raise').to_numpy(dtype=float)
            if not len(x) or not np.isfinite(x).all():
                raise ValueError('Dữ liệu rỗng hoặc chứa giá trị thiếu/không hữu hạn')
            model=cached_model(name,backend,len(required),len(meta['classes']))
            scores=predict_proba(model,backend,meta['scaler'].transform(x))
            idx=scores.argmax(axis=1)
            out=pd.DataFrame({'Kết quả':[meta['classes'][i] for i in idx],
                              'Xác suất lớn nhất':[f'{n*100:.2f}%' for n in scores.max(axis=1)]})
            st.success(f'Đã dự đoán {len(out)} mẫu bằng {BACKENDS[backend]}')
            st.dataframe(out,use_container_width=True,hide_index=True)
            prob_df=pd.DataFrame({'Lớp':meta['classes'],'Xác suất':scores[0]})
            fig=px.bar(prob_df,x='Lớp',y='Xác suất',color='Lớp',color_discrete_sequence=COLORS)
            fig.update_yaxes(range=[0,1])
            nice_plot(fig,height=290)
            st.download_button('Tải kết quả dự đoán CSV',out.to_csv(index=False).encode('utf-8-sig'),
                               file_name=f'prediction_{name}_{backend}.csv',mime='text/csv',key=f'download_{name}')
        except (FileNotFoundError,ValueError,RuntimeError,TypeError) as e:
            st.error(f'Không thể chạy dự đoán: {e}')
    if name=='breast_cancer':
        st.warning('Bộ dữ liệu chỉ phục vụ giảng dạy và nghiên cứu; không dùng hệ thống này để chẩn đoán bệnh.')

def show_compare(name,raw):
    rows=metrics_for(name)
    if not rows:
        st.info('Chưa có số liệu. Chạy `python train_all.py` để tạo checkpoint và metrics.')
        return
    frame=pd.DataFrame(rows)
    frame['Cách cài đặt']=frame.backend.map(BACKENDS)
    st.markdown('#### So sánh mô hình trên cùng bộ Test')
    tabs=frame[['Cách cài đặt','accuracy','precision_macro','recall_macro','f1_macro','train_seconds']].copy()
    tabs.columns=['Mô hình','Accuracy','Precision (macro)','Recall (macro)','F1 (macro)','Train (giây)']
    st.dataframe(tabs,hide_index=True,use_container_width=True,
                 column_config={c:st.column_config.NumberColumn(format='%.4f') for c in tabs.columns[1:]})
    melted=frame.melt(id_vars=['Cách cài đặt'],value_vars=['accuracy','f1_macro'],var_name='Chỉ số',value_name='Giá trị')
    fig=px.bar(melted,x='Cách cài đặt',y='Giá trị',color='Chỉ số',barmode='group',
               color_discrete_map={'accuracy':'#6EE4C2','f1_macro':'#80BBFF'})
    fig.update_yaxes(range=[0,1.06])
    nice_plot(fig)
    selected=st.selectbox('Xem ma trận nhầm lẫn của',list(BACKENDS),format_func=lambda x:BACKENDS[x],key=f'cm_{name}')
    current=next(r for r in rows if r['backend']==selected)
    fig=px.imshow(np.array(current['confusion_matrix']),x=list(map(str,raw.target_names)),
                  y=list(map(str,raw.target_names)),text_auto=True,aspect='auto',
                  color_continuous_scale=[[0,'#132b44'],[1,'#58C8C7']])
    fig.update_layout(xaxis_title='Nhãn dự đoán',yaxis_title='Nhãn thực tế')
    nice_plot(fig,360)
    st.caption(f'Chia tập cố định: {rows[0]["train_size"]} train / {rows[0]["val_size"]} validation / {rows[0]["test_size"]} test; seed 42, 350 epoch, SGD learning rate 0.15. Thời gian huấn luyện phụ thuộc máy và có chi phí overhead của framework.')

def show_code(name):
    st.markdown('#### Code tách biệt của cả ba mô hình')
    tabs=st.tabs(['01 · From Scratch','02 · Keras 3','03 · PyTorch'])
    for tab,key in zip(tabs,BACKENDS):
        with tab:
            p=ROOT/'projects'/name/f'{key}.py'
            st.code(p.read_text(encoding='utf-8'),language='python',line_numbers=True)
            st.download_button('Tải file '+key,p.read_bytes(),file_name=f'{name}_{key}.py',key=f'src_{name}_{key}')
    st.caption('Một bài toán Logistic Regression, ba code độc lập; cùng dữ liệu, tham số, tiêu chí đánh giá.')

def render_app(name):
    st.set_page_config(page_title=f'{SOURCES[name]["vi_name"]} | ML Lab',page_icon='🧪',layout='wide')
    theme()
    spec=SOURCES[name]
    try:
        raw=dataset(name)
    except (FileNotFoundError, ValueError) as e:
        st.error(f'Dữ liệu Kaggle chưa sẵn sàng: {e}')
        st.code('python download_kaggle.py\npython train_all.py',language='bash')
        st.stop()
    hero(f'{spec["icon"]}  {spec["vi_name"]}',f'ML LAB  /  {spec["domain"].upper()}',spec['description'])
    if not (ARTIFACTS/f'{name}_metadata.joblib').exists():
        st.warning('Chưa tìm thấy checkpoint. Chạy `python train_all.py` để tạo 9 mô hình.')
    a,b,c,d=st.tabs(['📊  Khám phá dataset','🔮  Dự đoán trực tiếp','📈  Đánh giá & so sánh','🧑‍💻  Xem source code'])
    with a:show_dataset(name,raw)
    with b:show_predict(name,raw)
    with c:show_compare(name,raw)
    with d:show_code(name)
    st.divider()
    st.caption('Machine Learning Lab · Dữ liệu Kaggle · Iris / Red Wine Quality / Breast Cancer · From Scratch / Keras / PyTorch')
