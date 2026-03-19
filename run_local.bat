@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  py -3.11 -m venv .venv
)

echo Installing/updating dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt

echo Launching Streamlit dashboard...
".venv\Scripts\streamlit.exe" run dashboard.py --server.port 8501 --server.address 127.0.0.1
