"""Initialize the PRISM-RAG database schema.

Run this ONCE after creating the database. Safe to re-run (uses IF NOT
EXISTS everywhere).

Prereqs:
    1. Create the database:  createdb prism_rag
    2. pgvector extension is installable (Postgres 14+ with pgvector package).
    3. PG_DSN env var (or .env) points at the right database.

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
    with psycopg.connect(PG_DSN) as conn, conn.cursor() as cur:
        cur.execute(sql)
        conn.commit()
    print(f"[ok] schema applied from {SCHEMA_PATH}")
    print(f"[ok] DSN: {PG_DSN}")


if __name__ == "__main__":
    main()