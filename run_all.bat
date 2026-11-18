@echo off
cd /d "%~dp0"
if not exist "backend\.venv" (
  py -3.13 -m venv backend\.venv
)
call backend\.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
python -m pipeline.run_all
start "backend" cmd /k "cd /d %~dp0\backend && ..\.venv\Scripts\python -m uvicorn app.main:app --host 0.0.0.0 --port 8000"
start "frontend" cmd /k "cd /d %~dp0\frontend && npm install && npm run dev -- --host 0.0.0.0 --port 5173"
