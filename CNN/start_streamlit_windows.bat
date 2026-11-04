@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Opening CNN Vision Lab at http://localhost:8501
python -m streamlit run app.py
pause
