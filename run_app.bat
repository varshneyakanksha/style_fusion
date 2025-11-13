@echo off
echo Starting Style Fusion App...
cd /d "%~dp0"
call .venv\Scripts\activate
streamlit run app.py
pause
