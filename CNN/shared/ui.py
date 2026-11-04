"""Shared professional dashboard. Each Kaggle project lives on its own Streamlit page."""
import os
os.environ['KERAS_BACKEND']='torch'
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image,ImageOps,ImageEnhance
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from .config import CONFIG,ROOT,ART,METHODS
from .datasets import preview_data,class_distribution,file_paths
from .inference import load_model,predict_image,get_result

PALETTE=['#50E0C1','#68A9FF','#F4B16C','#AC8CFA','#F472B6','#A3E635','#FBBF24','#56CFE1','#F97373','#B2EAF5']


def theme():
    st.markdown("""<style>
        .stApp{background:radial-gradient(ellipse 1000px 600px at 95% -20%,#1d4162 0%,#0a1b2f 65%,#071426 100%);}
        [data-testid="stSidebar"] {background:#0a1c31;border-right:1px solid #2b4666;}
        .block-container{max-width:1240px;padding:2rem 2rem 4rem;}
        h1,h2,h3{letter-spacing:-.035em;}
        h1{font-size:2.4rem!important;}
        .hero{border:1px solid #3a5771;background:linear-gradient(110deg,rgba(30,66,97,.95),rgba(11,28,47,.95));border-radius:24px;padding:32px 36px;margin-bottom:23px;}
        .eyebrow{letter-spacing:.16em;text-transform:uppercase;color:#55dfc2;font-size:11px;font-weight:800;}
        .hero h1{margin:9px 0;color:#f0faff;line-height:1.18;}
        .hero p{color:#b8cbdf;font-size:1rem;line-height:1.65;margin-bottom:0;}
        .metric-card{border:1px solid #2a4964;border-radius:17px;padding:18px;background:#112940;}
        .metric-card .small{font-size:12px;color:#a8bed3;}
        .metric-card .big{font-size:25px;font-weight:800;color:#f1f8ff;line-height:1.7;}
        .appcard{border:1px solid #294963;border-radius:19px;padding:23px;background:linear-gradient(155deg,#152f4c,#0e233a);min-height:196px;}
        .appcard h3{margin:0 0 9px;font-size:1.18rem;}
        .appcard p{color:#a9c0d4;font-size:.88rem;line-height:1.6;}
        div[data-testid="stTabs"] button[role="tab"]{font-weight:650;}
        div[data-testid="stMetric"]{border:1px solid #31516d;border-radius:14px;padding:11px;background:#10263d;}
        div.stButton>button[kind="primary"]{background:linear-gradient(90deg,#4be0b8,#62bbf0);color:#092034;border:0;font-weight:800;border-radius:10px;}
        button[kind="secondary"]{border-radius:10px;}
        a{color:#72c9ff!important;}
    </style>""",unsafe_allow_html=True)


def hero(title,lead,tag='CNN / COMPUTER VISION'):
    st.markdown(f'<div class="hero"><div class="eyebrow">{tag}</div><h1>{title}</h1><p>{lead}</p></div>',unsafe_allow_html=True)


def small_card(title,value):
    st.markdown(f'<div class="metric-card"><div class="small">{title}</div><div class="big">{value}</div></div>',unsafe_allow_html=True)


def chart(fig,height=340):
    fig.update_layout(template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
                      font=dict(color='#dcecfb'),height=height,margin=dict(l=16,r=15,t=29,b=20),legend_title_text='')
    st.plotly_chart(fig,use_container_width=True)


@st.cache_resource(show_spinner=False)
def cached_model(key,method):
    return load_model(key,method)


def _fetch_metrics(key):
    rows=[]
    for method in METHODS:
        try:rows.append(get_result(key,method))
        except FileNotFoundError:pass
    return rows


def show_dataset(key):
    spec=CONFIG[key]
    st.markdown('### Thông tin dataset Kaggle')
    st.markdown(f"**Nguồn:** [{spec['kaggle']}]({spec['url']})")
    st.write(spec['description'])
    a,b,c=st.columns(3)
    with a:small_card('Tập train gốc',f"{spec['n_train']:,}")
    with b:small_card('Tập test gốc',f"{spec['n_test']:,}")
    with c:small_card('Kích thước ảnh / lớp',f"28×28 · {len(spec['classes'])}")
    st.write('')
    st.markdown('#### Ví dụ ảnh và nhãn từ CSV')
    try:
        images,labels,frame=preview_data(key)
        cols=st.columns(4)
        for i,(image,label) in enumerate(zip(images,labels)):
            with cols[i%4]:
                st.image(Image.fromarray((image[0]*255).astype(np.uint8)).resize((144,144),Image.Resampling.NEAREST),
                         caption=f'Mẫu {i+1}: {label}',width=144)
        st.markdown('#### Output một số mẫu dữ liệu (CSV)')
        st.dataframe(frame.head(5),use_container_width=True,hide_index=True)
        counts=class_distribution(key)
        values=pd.DataFrame({'Lớp':list(counts),'Số lượng':list(counts.values())})
        values['Tỷ lệ (%)']=np.round(values['Số lượng']/values['Số lượng'].sum()*100,2)
        st.markdown('#### Phân bố lớp (tập train gốc)')
        left,right=st.columns([1.8,1])
        with left:
            fig=px.bar(values,x='Lớp',y='Số lượng',color='Lớp',color_discrete_sequence=PALETTE)
            chart(fig,360)
        with right:st.dataframe(values,hide_index=True,use_container_width=True,height=370)
        st.info('Nhận xét: '+spec['note'])
    except (FileNotFoundError,ValueError) as e:
        st.warning(f'{e}. Hãy chạy `python download_kaggle.py` trước.')


def upload_image_widget(key):
    """Pick a real CSV sample or upload JPG/PNG; grayscale and resize on demand."""
    mode=st.radio('Nguồn ảnh', ['Ảnh ví dụ từ Kaggle','Tải ảnh PNG/JPG lên'],horizontal=True,key='mode_'+key)
    if mode=='Ảnh ví dụ từ Kaggle':
        try:
            images,labels,_=preview_data(key)
        except (FileNotFoundError,ValueError) as e:
            st.warning(str(e));return None,None
        idx=st.select_slider('Chọn ảnh ví dụ',options=list(range(len(images))),format_func=lambda i:f'Mẫu #{i+1}',key='sample_'+key)
        im=images[idx,0].copy()
        return im,labels[idx]
    uploaded=st.file_uploader('Tải ảnh chứa 1 đối tượng',type=['png','jpg','jpeg'],key='upload_'+key)
    if uploaded is None:return None,None
    try:
        im=Image.open(uploaded).convert('L').resize((28,28),Image.Resampling.LANCZOS)
        if st.checkbox('Đảo màu: nền tối, nét sáng',value=(key=='mnist'),key='invert_'+key):
            im=ImageOps.invert(im)
        return np.asarray(im,dtype=np.float32)/255.0,None
    except Exception as e:
        st.error(f'Ảnh không hợp lệ: {e}');return None,None


def show_predict(key):
    st.markdown('### Phân loại ảnh bằng CNN')
    st.caption('3 mô hình dùng cùng kiến trúc. Ảnh được đưa về grayscale 28×28 trước khi dự đoán.')
    method=st.segmented_control('Chọn cách cài đặt CNN',options=list(METHODS),format_func=lambda x:METHODS[x],default='pytorch',key='method_'+key)
    method=method or 'pytorch'
    left,right=st.columns([1,1.3],gap='large')
    with left:
        img,true_label=upload_image_widget(key)
        if img is not None:
            display=Image.fromarray(np.uint8(np.clip(img,0,1)*255)).resize((252,252),Image.Resampling.NEAREST)
            st.image(display,caption='Ảnh đầu vào đã chuẩn hóa (phóng to)',width=252)
            if true_label is not None:st.caption(f'Nhãn gốc trong CSV: **{true_label}**')
    with right:
        st.markdown('#### Kết quả dự đoán')
        if st.button('✨  Phân loại ngay',type='primary',key='predict_'+key,use_container_width=True):
            if img is None:
                st.warning('Chọn ảnh ví dụ hoặc tải ảnh lên.');return
            try:
                model=cached_model(key,method)
                prob=predict_image(model,method,img)[0]
                names=CONFIG[key]['classes']
                ix=int(prob.argmax())
                st.success(f'**Dự đoán:** {names[ix]} · {prob[ix]*100:.1f}%')
                order=np.argsort(prob)[::-1][:min(8,len(prob))]
                d=pd.DataFrame({'Lớp':[names[i] for i in order],'Xác suất':[float(prob[i]) for i in order]})
                fig=px.bar(d,x='Xác suất',y='Lớp',orientation='h',color='Xác suất',color_continuous_scale=['#3068a4','#50e0c1'])
                fig.update_layout(yaxis=dict(autorange='reversed'),coloraxis_showscale=False)
                chart(fig,325)
            except (FileNotFoundError,ValueError,RuntimeError) as e:
                st.warning(f'{e}\n\nHuấn luyện trước: `python train_all.py`')
        else:
            st.info('Chọn model và ảnh, sau đó nhấn Phân loại ngay để xem dự đoán và xác suất.')
    st.caption('Lưu ý: Xác suất của mô hình không phải mức độ chắc chắn tuyệt đối. Ảnh khác phân bố Kaggle có thể cho kết quả sai.')


def show_metrics(key):
    st.markdown('### Đánh giá & so sánh 3 cách cài đặt')
    rows=_fetch_metrics(key)
    if not rows:
        st.warning('Chưa có kết quả từ CSV Kaggle thật. Hãy chạy `python train_all.py` để tạo bảng so sánh. Không sử dụng số liệu minh họa làm kết quả thực nghiệm.')
        return
    names={r['method']:r['method_name'] for r in rows}
    d=pd.DataFrame([{'Mô hình':r['method_name'],'Accuracy':r['accuracy'],'Precision':r['precision_macro'],
                     'Recall':r['recall_macro'],'Macro-F1':r['f1_macro'],'Test':r['n_test']} for r in rows])
    st.dataframe(d.style.format({'Accuracy':'{:.2%}','Precision':'{:.2%}','Recall':'{:.2%}','Macro-F1':'{:.2%}'}),
                 hide_index=True,use_container_width=True)
    col1,col2,col3=st.columns(3)
    with col1:st.metric('Accuracy tốt nhất',f"{max(r['accuracy'] for r in rows):.2%}")
    with col2:st.metric('Macro-F1 tốt nhất',f"{max(r['f1_macro'] for r in rows):.2%}")
    with col3:st.metric('Số phiên bản đã chạy',f'{len(rows)}/3')
    metrics=['Accuracy','Precision','Recall','Macro-F1']
    long=d.melt(id_vars='Mô hình',value_vars=metrics,var_name='Chỉ số',value_name='Giá trị')
    fig=px.bar(long,x='Chỉ số',y='Giá trị',color='Mô hình',barmode='group',color_discrete_sequence=PALETTE,range_y=[0,1])
    chart(fig,350)
    sel=st.selectbox('Xem confusion matrix của',options=[r['method'] for r in rows],format_func=lambda m:names[m])
    chosen=next(r for r in rows if r['method']==sel)
    mat=np.asarray(chosen['confusion_matrix'])
    fig=px.imshow(mat,x=chosen['classes'],y=chosen['classes'],color_continuous_scale='Blues',
                  labels=dict(x='Dự đoán',y='Thực tế',color='Số ảnh'))
    chart(fig,460 if len(mat)>10 else 380)
    with st.expander('Cấu hình huấn luyện & lịch sử'):
        st.json(chosen['run_settings'])
        h=pd.DataFrame(chosen['history'])
        st.line_chart(h.set_index('epoch')[['val_accuracy']])
    st.caption('Train/Validation/Test tách đúng: Validation lấy từ CSV train; Test giữ từ CSV test Kaggle. Giới hạn mẫu được ghi trong JSON từng mô hình.')


def show_code(key):
    st.markdown('### Mã nguồn chia riêng theo framework')
    sel=st.selectbox('Cách cài đặt',options=list(METHODS),format_func=lambda m:METHODS[m],key='source_'+key)
    src=ROOT/'projects'/key/f'{sel}.py'
    st.code(src.read_text(encoding='utf-8'),language='python',line_numbers=True)
    st.markdown('**Mã nguồn lõi tái sử dụng giữa ba app**')
    core=ROOT/'shared'/('cnn_numpy.py' if sel=='scratch' else 'backends.py')
    with st.expander(core.name+' — Xem toàn bộ triển khai'):
        st.code(core.read_text(encoding='utf-8'),language='python',line_numbers=True)
    st.caption('Các app độc lập có cùng CNN; source lõi được chia sẻ để tránh sao chép và đảm bảo công bằng.')


def render_app(key):
    st.set_page_config(page_title=f"{CONFIG[key]['short']} | CNN Lab",page_icon='🧠',layout='wide')
    theme()
    s=CONFIG[key]
    hero(f"{s['emoji']}  {s['title']}",s['description'],f"ỨNG DỤNG CNN · {s['short'].upper()}")
    tabs=st.tabs(['📊  Dataset Kaggle','🧠  Dự đoán CNN','📈  Đánh giá & so sánh','💻  Source code'])
    with tabs[0]:show_dataset(key)
    with tabs[1]:show_predict(key)
    with tabs[2]:show_metrics(key)
    with tabs[3]:show_code(key)
