#!/usr/bin/env bash
# Start the Postgres container and wait for it to be ready before returning.
#
# Idempotent: if the container is already running and healthy, exits quickly.
#
# Usage:
#   bash scripts/00_db_up.sh
#   bash scripts/00_db_up.sh --reset   # wipe data first (destructive)

set -euo pipefail

cd "$(dirname "$0")/.."

if [[ "${1:-}" == "--reset" ]]; then
    echo "[reset] stopping container and wiping ./postgres-data ..."
    docker compose down -v || true
    rm -rf ./postgres-data
fi

echo "[up] starting postgres ..."
docker compose up -d postgres

echo "[wait] polling pg_isready ..."
for i in {1..30}; do
    if docker compose exec -T postgres pg_isready -U "${POSTGRES_USER:-prism}" -d "${POSTGRES_DB:-prism_rag}" >/dev/null 2>&1; then
        echo "[ok] postgres is ready (after ${i}s)"
        exit 0
    fi
    sleep 1
done

echo "[fail] postgres did not become ready in 30s"
docker compose logs --tail=50 postgres
exit 1