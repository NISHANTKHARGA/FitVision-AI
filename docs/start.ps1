# FitVision AI — start both servers (Windows)
# Backend:  python -m uvicorn app.main:app --port 8017
# Frontend: npm run preview (production build) on 5173
# NOTE: use `python -m uvicorn`, NOT a bare `uvicorn` (that resolves to Python 3.14 here).
# Assumes the standard layout: <workspace>/{frontend,backend,ai,docs}

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)  # workspace (docs/ is one level deep)

Write-Host "Workspace: $root" -ForegroundColor DarkGray
Write-Host "Starting backend on http://127.0.0.1:8017 ..." -ForegroundColor DarkGray
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root\backend'; `$env:PYTHONIOENCODING='utf-8'; python -m uvicorn app.main:app --host 127.0.0.1 --port 8017"

Write-Host "Building frontend ..." -ForegroundColor DarkGray
Push-Location "$root\frontend"
npm run build
Pop-Location

Write-Host "Starting frontend on http://127.0.0.1:5173 ..." -ForegroundColor DarkGray
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root\frontend'; npm run preview -- --port 5173"

Write-Host "`nFitVision AI will be at: http://127.0.0.1:5173" -ForegroundColor Green
