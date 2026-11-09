@echo off
TITLE TrustNews AI — Trust-Aware Personalized News Recommendation
echo =======================================================================
echo   TrustNews AI  |  Advanced Multi-Objective Trust-Aware Framework
echo   Assam DownTown University  |  B.Tech CSE  |  2024-28
echo =======================================================================
echo.
cd /d "%~dp0"

REM Try activating the virtual environment if it exists
IF EXIST ".venv\Scripts\activate.bat" (
    echo [1/2] Activating .venv virtual environment...
    call .venv\Scripts\activate.bat
) ELSE IF EXIST "venv\Scripts\activate.bat" (
    echo [1/2] Activating venv virtual environment...
    call venv\Scripts\activate.bat
) ELSE (
    echo [1/2] No venv found — using system Python...
)

echo.
echo [2/2] Launching TrustNews AI on http://localhost:8501
echo        Press Ctrl+C to stop the server.
echo.

REM Use python -m streamlit so it works even if 'streamlit' is not on PATH
python -m streamlit run app.py --server.headless false --browser.gatherUsageStats false

pause
