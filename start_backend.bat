@echo off
cd /d "D:\RR\backend"
if errorlevel 1 (
  echo Could not change to D:\RR\backend.
  pause
  exit /b 1
)

echo Checking for a process listening on port 8000...
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R /C:":8000 .*LISTENING"') do (
  echo Stopping process %%P on port 8000...
  taskkill /F /PID %%P
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 (
  echo Could not activate backend\.venv. Run the setup steps in README.md first.
  pause
  exit /b 1
)

uvicorn app.main:app --port 8000
pause
