"""Initialize / update the PRISM-RAG database schema.

init.sql is IDEMPOTENT and NON-DESTRUCTIVE: it creates what is missing and
evolves the schema in place (new enum values, new columns, new indexes). It
never drops tables, so re-running this is SAFE on a populated database — your
ingested chunks are preserved. This is the single source of truth; there is no
separate migrations file.

Prereqs:
    1. Postgres is up (docker compose up -d postgres) with pgvector available.
    2. PG_DSN env var (or .env) points at the right database.

Usage (from repo root):
    python scripts/01_init_db.py
"""
import sys
from pathlib import Path

# Allow imports of agents.ingestion.* without installing the project as a package.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import psycopg

from agents.ingestion.config import PG_DSN


SCHEMA_PATH = REPO_ROOT / "init.sql"


def main() -> None:
    sql = SCHEMA_PATH.read_text()
    # autocommit so ALTER TYPE ... ADD VALUE never trips the
    # "can't add enum value inside a transaction" restriction on any PG version.
    with psycopg.connect(PG_DSN, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute(sql)
    print(f"[ok] schema applied (idempotent) from {SCHEMA_PATH}")
    print(f"[ok] DSN: {PG_DSN}")


if __name__ == "__main__":
    main()