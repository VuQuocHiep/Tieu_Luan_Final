import streamlit as st
from shared.ui import render_app
st.set_page_config(page_title='RNN · Emotion', page_icon='🙂', layout='wide')
render_app('emotion')
