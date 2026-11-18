@echo off
start "SupeRR Backend" cmd /k ""%~dp0start_backend.bat""
start "SupeRR Frontend" cmd /k ""%~dp0start_frontend.bat""
timeout /t 6 /nobreak >nul
start "" "http://localhost:5173"
