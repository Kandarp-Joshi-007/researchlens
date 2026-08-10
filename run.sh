#!/usr/bin/env bash
# Start ResearchLens (FastAPI backend + Streamlit frontend).
#
# Usage:
#   bash run.sh              # localhost only (default)
#   HOST=0.0.0.0 bash run.sh # expose on the local network — see the warning below
#
# The API has no authentication. Binding it to 0.0.0.0 lets anyone who can
# reach this machine upload, read and delete papers, so it stays on loopback
# unless you explicitly opt in.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"   # uvicorn resolves backend.main from the project root

HOST="${HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-8501}"
OLLAMA="${OLLAMA_HOST:-http://localhost:11434}"

if [ "$HOST" != "127.0.0.1" ] && [ "$HOST" != "localhost" ]; then
  echo "WARNING: binding to $HOST exposes an unauthenticated API to the network."
  echo "         Set RESEARCHLENS_ALLOWED_ORIGINS to match your frontend origin."
fi

if ! curl -sf "$OLLAMA/api/tags" >/dev/null 2>&1; then
  echo "ERROR: Ollama is not reachable at $OLLAMA"
  echo "       Start it with 'ollama serve', then re-run this script."
  echo "       Required models: qwen2.5:7b and nomic-embed-text"
  exit 1
fi

# No --reload: it restarts the server whenever a file changes, which would
# abandon an analysis that takes several minutes to complete.
echo "Starting backend  on http://$HOST:$BACKEND_PORT ..."
python -m uvicorn backend.main:app --host "$HOST" --port "$BACKEND_PORT" &
BACKEND_PID=$!

echo "Starting frontend on http://$HOST:$FRONTEND_PORT ..."
python -m streamlit run "$ROOT/frontend/app.py" \
  --server.port "$FRONTEND_PORT" \
  --server.address "$HOST" \
  --server.headless true &
FRONTEND_PID=$!

cleanup() { kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo
echo "ResearchLens is running. Open http://localhost:$FRONTEND_PORT"
echo "Press Ctrl-C to stop."
wait
