"""Embed all chunks as level-0 tree_nodes (BGE-M3 dense, 1024-d).

Usage (from repo root):
    python scripts/03_embed_chunks.py
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agents.ingestion.embed_leaves import embed_all_leaves
from agents.ingestion.db import connect


def verify():
    """Print Phase 2 verification stats."""
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM chunks")
            n_chunks = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM tree_nodes WHERE level=0")
            n_leaves = cur.fetchone()[0]
            print(f"\n=== Phase 2 verification ===")
            print(f"  chunks:  {n_chunks:,}")
            print(f"  leaves:  {n_leaves:,}")
            print(f"  missing: {n_chunks - n_leaves:,}")

            if n_leaves > 0:
                cur.execute("""
                    SELECT domain, source, COUNT(*)
                    FROM tree_nodes WHERE level=0
                    GROUP BY domain, source ORDER BY domain, source
                """)
                print(f"\n  leaves per (domain, source):")
                for dom, src, n in cur.fetchall():
                    print(f"    {dom:<10} {src:<25} {n:>10,}")


def build_hnsw():
    """Build HNSW index AFTER all embeddings are loaded."""
    print("\n[hnsw] building index (this may take a few minutes)...")
    with connect() as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SET maintenance_work_mem = '4GB'")
            cur.execute("SET max_parallel_maintenance_workers = 7")
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_tree_nodes_hnsw
                ON tree_nodes USING hnsw (embedding vector_ip_ops)
                WITH (m = 16, ef_construction = 200)
            """)
            cur.execute("ANALYZE tree_nodes")
    print("[hnsw] done.")


def main():
    print("[phase 2] embedding chunks as leaf nodes...")
    total = embed_all_leaves()
    print(f"[phase 2] embedded {total:,} new leaves")

    verify()
    build_hnsw()


if __name__ == "__main__":
    main()