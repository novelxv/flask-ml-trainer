@echo off
echo Building Flask ML Trainer executable...
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install requirements
echo Installing requirements...
pip install -r requirements.txt

REM Build executable
echo Building executable with PyInstaller...
pyinstaller app.spec

echo.
echo Build complete! Check the 'dist' folder for the executable.
echo You can run FlaskMLTrainer.exe to start the application.
pause
