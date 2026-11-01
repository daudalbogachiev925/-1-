#!/usr/bin/env bash
set -euo pipefail

export API_HOST="${API_HOST:-0.0.0.0}"
export API_PORT="${API_PORT:-8000}"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"

exec uvicorn src.api.main:app --host "${API_HOST}" --port "${API_PORT}" --log-level "${LOG_LEVEL,,}"
