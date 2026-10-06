#!/usr/bin/env bash
# ==============================================================================
# LLM Response Evaluation Framework - Linux / macOS Launcher
# ==============================================================================

set -e

echo "[INFO] Starting LLM Response Evaluation Platform..."

if [ -f ".venv/bin/python" ]; then
    echo "[INFO] Detected virtual environment in .venv"
    PYTHON_EXE=".venv/bin/python"
elif [ -f "venv/bin/python" ]; then
    echo "[INFO] Detected virtual environment in venv"
    PYTHON_EXE="venv/bin/python"
else
    echo "[INFO] Using system python"
    PYTHON_EXE="python3"
fi

echo "[INFO] Launching FastAPI Web Application and Dashboard..."
echo "[INFO] Open your browser at: http://localhost:8000"
echo ""

exec $PYTHON_EXE -m uvicorn src.input_module.main:app --reload --host 0.0.0.0 --port 8000
