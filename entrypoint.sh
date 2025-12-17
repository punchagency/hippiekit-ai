#!/usr/bin/env sh
set -e

# Default host/port; override with PORT env var if provided
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8001}
TIMEOUT_KEEP_ALIVE=${UVICORN_TIMEOUT_KEEP_ALIVE:-240}

# Enable reload in development when UVICORN_RELOAD=true
if [ "$UVICORN_RELOAD" = "true" ]; then
  python -m uvicorn main:app --host "$HOST" --port "$PORT" --reload --timeout-keep-alive "$TIMEOUT_KEEP_ALIVE"
else
  python -m uvicorn main:app --host "$HOST" --port "$PORT" --timeout-keep-alive "$TIMEOUT_KEEP_ALIVE"
fi
