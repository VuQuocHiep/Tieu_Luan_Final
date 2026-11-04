import streamlit as st
from shared.ui import render_app
st.set_page_config(page_title='RNN · SMS Spam', page_icon='📩', layout='wide')
render_app('sms')
