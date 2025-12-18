@echo off
echo Setting up CN6005 - Heart Disease Classification...
echo.

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Run the interactive CLI
echo.
echo Starting application...
python -m app.cli

pause
