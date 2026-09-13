#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/myenv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/myenv/bin/activate"
fi

cd "$ROOT/backend"
exec uvicorn main:app --reload --host 127.0.0.1 --port 8000
