#!/usr/bin/env bash
# Start backend (FastAPI) and frontend (Streamlit) together.
# Usage: bash run.sh

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "Starting ResearchLens backend on http://localhost:8000 ..."
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

echo "Starting ResearchLens frontend on http://localhost:8501 ..."
streamlit run "$ROOT/frontend/app.py" --server.port 8501 &
FRONTEND_PID=$!

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT INT TERM
wait
