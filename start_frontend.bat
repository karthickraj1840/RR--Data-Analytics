@echo off
cd /d "D:\RR\frontend"
if errorlevel 1 (
  echo Could not change to D:\RR\frontend.
  pause
  exit /b 1
)

if not exist "node_modules" (
  echo Installing frontend dependencies...
  npm install
  if errorlevel 1 (
    echo Frontend dependency installation failed.
    pause
    exit /b 1
  )
)

npm run dev
pause
