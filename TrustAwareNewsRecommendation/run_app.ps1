# TrustNews AI — PowerShell Launch Script
Write-Host "=======================================================================" -ForegroundColor Cyan
Write-Host "  TrustNews AI  |  Advanced Multi-Objective Trust-Aware Framework" -ForegroundColor White
Write-Host "  Assam DownTown University  |  B.Tech CSE  |  2024-28" -ForegroundColor Gray
Write-Host "=======================================================================" -ForegroundColor Cyan
Write-Host ""

Set-Location $PSScriptRoot

# Activate virtual environment if present
if (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Host "[1/2] Activating .venv ..." -ForegroundColor Yellow
    & ".venv\Scripts\Activate.ps1"
} elseif (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "[1/2] Activating venv ..." -ForegroundColor Yellow
    & "venv\Scripts\Activate.ps1"
} else {
    Write-Host "[1/2] No venv found — using system Python" -ForegroundColor Gray
}

Write-Host ""
Write-Host "[2/2] Starting TrustNews AI at http://localhost:8501" -ForegroundColor Green
Write-Host "       Press Ctrl+C to stop." -ForegroundColor Gray
Write-Host ""

python -m streamlit run app.py --server.headless $false --browser.gatherUsageStats $false
