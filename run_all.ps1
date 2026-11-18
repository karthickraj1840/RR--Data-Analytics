$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Test-Path "backend/.venv")) {
    py -3.13 -m venv backend/.venv
}
& "backend/.venv/Scripts/python.exe" -m pip install --upgrade pip
& "backend/.venv/Scripts/python.exe" -m pip install -r backend/requirements.txt
& "backend/.venv/Scripts/python.exe" -m pipeline.run_all
Start-Process powershell -ArgumentList "-NoExit","-Command","Set-Location '$PSScriptRoot/backend'; & '../.venv/Scripts/python.exe' -m uvicorn app.main:app --host 0.0.0.0 --port 8000"
Start-Process powershell -ArgumentList "-NoExit","-Command","Set-Location '$PSScriptRoot/frontend'; npm install; npm run dev -- --host 0.0.0.0 --port 5173"
