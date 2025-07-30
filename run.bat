@echo off
echo Starting Flask ML Trainer...
echo.
echo The application will start on http://localhost:5000
echo Press Ctrl+C to stop the server
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Virtual environment not found. Please run build.bat first.
    pause
    exit /b 1
)

REM Activate virtual environment and run app
call venv\Scripts\activate.bat && python app.py
