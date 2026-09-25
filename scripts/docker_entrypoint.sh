#!/usr/bin/env bash
# Seed a tiny demo run (if none present) and start the FastAPI dashboard.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PROCESSED="${TACTIVISION_PROCESSED_ROOT:-$ROOT/data/processed}"
SEED_SRC="${TACTIVISION_SEED_DIR:-$ROOT/deploy/seed}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

mkdir -p "$PROCESSED" "$ROOT/data/raw/uploads"

# Copy seed runs into data/processed when the run folder is missing.
if [[ -d "$SEED_SRC" ]]; then
  for seed in "$SEED_SRC"/*; do
    [[ -d "$seed" ]] || continue
    name="$(basename "$seed")"
    dest="$PROCESSED/$name"
    if [[ ! -d "$dest" ]]; then
      echo "Seeding demo run: $name -> $dest"
      mkdir -p "$dest"
      cp -a "$seed"/. "$dest"/
    fi
  done
fi

echo "Starting TactiVision on ${HOST}:${PORT}"
exec python -m uvicorn tactivision.api.app:app --host "$HOST" --port "$PORT"
