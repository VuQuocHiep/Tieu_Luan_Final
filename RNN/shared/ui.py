import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from .config import CONFIG, ARTIFACTS, ROOT
from .datasets import dataset_summary
from .train_runner import predict_text, artifact_paths

METHOD_LABELS = {'scratch': 'From Scratch (NumPy)', 'keras': 'Keras', 'pytorch': 'PyTorch'}


def theme():
    st.markdown('''
    <style>
      .stApp {background: linear-gradient(145deg,#07111f 0%,#0d1b2a 45%,#0b1320 100%); color:#eaf2ff;}
      .block-container {max-width:1180px; padding-top:2rem; padding-bottom:3rem;}
      .hero {padding:24px 28px;border:1px solid rgba(120,170,255,.25);border-radius:22px;background:linear-gradient(135deg,rgba(42,92,170,.22),rgba(18,30,50,.75));box-shadow:0 15px 45px rgba(0,0,0,.22);}
      .hero h1{font-size:2.15rem;margin:0 0 6px 0}.hero p{color:#b8c8dd;margin:0}
      .card{border:1px solid rgba(130,170,220,.18);background:rgba(14,28,46,.78);padding:18px;border-radius:18px;min-height:132px}
      .pill{display:inline-block;padding:6px 11px;border-radius:999px;background:#142c49;color:#a8d1ff;margin-right:6px;font-size:.84rem}
      [data-testid="stMetric"]{background:rgba(11,27,45,.8);padding:15px;border-radius:16px;border:1px solid rgba(130,170,220,.18)}
      .small{color:#9fb2c8;font-size:.92rem}
    </style>''', unsafe_allow_html=True)


def _metrics_rows(key):
    rows = []
    for method in METHOD_LABELS:
        p = artifact_paths(key, method)['metrics']
        if p.exists():
            m = json.loads(p.read_text(encoding='utf-8'))
            rows.append({'Method': METHOD_LABELS[method], 'method_key': method,
                         'Accuracy': m['accuracy'], 'Macro-F1': m['macro_f1'],
                         'Precision': m['macro_precision'], 'Recall': m['macro_recall'],
                         'N test': m['n_test']})
    return rows


def render_app(key):
    theme(); spec = CONFIG[key]
    st.markdown(f'''<div class="hero"><span class="pill">RNN</span><span class="pill">Kaggle</span>
    <h1>{spec['title']}</h1><p>{spec['description']}</p></div>''', unsafe_allow_html=True)
    tabs = st.tabs(['📚 Dataset Kaggle','🧠 Dự đoán RNN','📊 Đánh giá & so sánh','💻 Source code'])

    with tabs[0]:
        try:
            s = dataset_summary(key)
            c1,c2,c3 = st.columns(3)
            c1.metric('Tổng mẫu', f"{s['counts']['all']:,}")
            c2.metric('Số lớp', len(s['classes']))
            c3.metric('Train/Val/Test', f"{s['counts']['train']:,}/{s['counts']['val']:,}/{s['counts']['test']:,}")
            st.markdown(f"**Nguồn Kaggle:** {spec['url']}")
            st.subheader('Một số mẫu')
            sample_df = pd.DataFrame(s['samples'])
            st.dataframe(sample_df, use_container_width=True, hide_index=True)
            st.subheader('Phân bố lớp')
            dd = pd.DataFrame({'Lớp': list(s['distribution']), 'Số mẫu': list(s['distribution'].values())})
            st.plotly_chart(px.bar(dd, x='Lớp', y='Số mẫu', text='Số mẫu'), use_container_width=True)
            st.caption('Nhận xét: biểu đồ được tính trực tiếp từ dữ liệu Kaggle đã tải, không dùng số liệu giả.')
        except Exception as e:
            st.error(str(e)); st.code('python download_kaggle.py\npython tests/test_kaggle_data.py')

    with tabs[1]:
        method = st.selectbox('Chọn cách cài đặt', list(METHOD_LABELS), format_func=lambda x: METHOD_LABELS[x], key=f'm_{key}')
        examples = {
            'sms': 'Congratulations! You have won a free prize. Call now to claim your reward.',
            'imdb': 'The movie was engaging, well acted and surprisingly moving. I really enjoyed it.',
            'emotion': 'I feel so happy and grateful for everything that happened today.'
        }
        text = st.text_area('Nhập chuỗi cần phân loại', value=examples[key], height=150)
        if st.button('Dự đoán', type='primary', key=f'pred_{key}'):
            try:
                r = predict_text(key, method, text)
                st.success(f"Kết quả: **{r['label']}**")
                probs = pd.DataFrame({'Lớp': r['classes'], 'Xác suất': r['probabilities']})
                st.plotly_chart(px.bar(probs, x='Lớp', y='Xác suất', range_y=[0,1]), use_container_width=True)
            except Exception as e:
                st.error(str(e)); st.code(f'python train_all.py --dataset {key}')

    with tabs[2]:
        rows = _metrics_rows(key)
        if not rows:
            st.warning('Chưa có metrics. Hãy chạy train trước.'); st.code(f'python train_all.py --dataset {key}')
        else:
            df = pd.DataFrame(rows)
            best = df.sort_values(['Accuracy','Macro-F1'], ascending=False).iloc[0]
            c1,c2,c3 = st.columns(3)
            c1.metric('Accuracy tốt nhất', f"{best['Accuracy']*100:.2f}%")
            c2.metric('Macro-F1 tốt nhất', f"{df['Macro-F1'].max()*100:.2f}%")
            c3.metric('Số phiên bản đã chạy', f"{len(df)}/3")
            show = df[['Method','Accuracy','Macro-F1','Precision','Recall','N test']].copy()
            for col in ['Accuracy','Macro-F1','Precision','Recall']: show[col] = (show[col]*100).round(2).astype(str)+'%'
            st.dataframe(show, use_container_width=True, hide_index=True)
            long = df.melt(id_vars=['Method'], value_vars=['Accuracy','Macro-F1'], var_name='Metric', value_name='Score')
            st.plotly_chart(px.bar(long, x='Method', y='Score', color='Metric', barmode='group', range_y=[0,1]), use_container_width=True)
            method_cm = st.selectbox('Ma trận nhầm lẫn', df['method_key'].tolist(), format_func=lambda x: METHOD_LABELS[x], key=f'cm_{key}')
            mp = artifact_paths(key, method_cm)['metrics']
            m = json.loads(mp.read_text(encoding='utf-8'))
            fig = go.Figure(data=go.Heatmap(z=m['confusion_matrix'], x=m['classes'], y=m['classes'], colorscale='Blues', showscale=True))
            fig.update_layout(xaxis_title='Dự đoán', yaxis_title='Thực tế', height=520)
            st.plotly_chart(fig, use_container_width=True)

    with tabs[3]:
        method = st.radio('Chọn source', list(METHOD_LABELS), format_func=lambda x: METHOD_LABELS[x], horizontal=True, key=f'src_{key}')
        wrapper = ROOT / 'projects' / key / f'{method}.py'
        impl = ROOT / 'shared' / ('rnn_scratch.py' if method == 'scratch' else 'models.py')
        st.caption(f'Entry point: projects/{key}/{method}.py')
        st.code(wrapper.read_text(encoding='utf-8'), language='python')
        st.caption('Phần cài đặt dùng chung')
        st.code(impl.read_text(encoding='utf-8'), language='python')
