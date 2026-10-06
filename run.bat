@echo off
REM ==============================================================================
REM LLM Response Evaluation Framework - Windows One-Click Launcher
REM ==============================================================================

echo [INFO] Starting LLM Response Evaluation Platform...

REM Check if .venv exists
if exist ".venv\Scripts\python.exe" (
    echo [INFO] Detected virtual environment in .venv
    set PYTHON_EXE=.venv\Scripts\python.exe
) else if exist "venv\Scripts\python.exe" (
    echo [INFO] Detected virtual environment in venv
    set PYTHON_EXE=venv\Scripts\python.exe
) else (
    echo [INFO] Using system python
    set PYTHON_EXE=python
)

echo [INFO] Launching FastAPI Web Application and Dashboard...
echo [INFO] Open your browser at: http://localhost:8000
echo.

%PYTHON_EXE% -m uvicorn src.input_module.main:app --reload --host 0.0.0.0 --port 8000
pause
