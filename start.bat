@echo off
title Gravitational Wave Analyzer
echo.
echo  ============================================================
echo   GRAVITATIONAL WAVE ANALYZER - Web Interface
echo   IMRPhenomD Waveform Model - Pure Python
echo  ============================================================
echo.

:: Activate virtual environment
call "%~dp0gw_env\Scripts\activate.bat"

:: Install web dependencies if needed
pip show fastapi >nul 2>&1
if errorlevel 1 (
    echo  Installing web dependencies...
    pip install fastapi uvicorn[standard] --quiet
    echo  Done.
    echo.
)

echo  Starting server at http://localhost:8000
echo  Press Ctrl+C to stop.
echo.

:: Open browser after a short delay
start "" "http://localhost:8000"

:: Start the FastAPI server
cd /d "%~dp0"
python -m uvicorn server:app --host 127.0.0.1 --port 8000
