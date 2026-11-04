@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Chua tim thay .venv. Hay doc README.md de cai dat.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run app3_multi_agent/app.py
pause
