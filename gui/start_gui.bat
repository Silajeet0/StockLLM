@echo off
:: start_gui.bat — Launch the Stock Market LLM Django GUI on Windows
:: Usage: start_gui.bat [port]  (default: 8000)

SET PORT=8000
IF NOT "%~1"=="" SET PORT=%~1

echo.
echo ==========================================================
echo        Stock Market LLM -- Django GUI Launcher
echo ==========================================================
echo.
echo   Port : %PORT%
echo.

:: Check Python
WHERE python >nul 2>&1
IF ERRORLEVEL 1 (
    echo [ERROR] Python not found. Please install Python 3.9+
    pause
    exit /b 1
)

echo [INFO] Checking dependencies...
python -m pip install -q django yfinance pandas numpy

echo.
echo [INFO] Starting server at http://127.0.0.1:%PORT%
echo        Press Ctrl+C to stop.
echo.

cd /d "%~dp0"
python manage.py runserver "0.0.0.0:%PORT%" --noreload
pause
