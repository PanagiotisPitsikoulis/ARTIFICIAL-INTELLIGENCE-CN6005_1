@echo off
title CN6005 - Heart Disease Classification

REM Check if we're running in a proper console
REM If double-clicked, restart in a new cmd window
if "%RUNNING_IN_CMD%"=="" (
    set RUNNING_IN_CMD=1
    start "CN6005 - Heart Disease Classification" cmd /k "%~f0"
    exit
)

echo Setting up CN6005 - Heart Disease Classification...
echo.

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        echo Make sure Python is installed and in your PATH.
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies.
    pause
    exit /b 1
)

REM Run the interactive CLI
echo.
echo Starting application...
echo.
python -m app.cli

pause
