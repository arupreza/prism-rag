"""Phase 3 — build hierarchical topic tree (UMAP + HDBSCAN + LLM summaries).

Prereqs:
  - Phase 2 complete (level-0 leaves embedded in tree_nodes)
  - vLLM serving Qwen2.5-32B-Instruct at SUMMARIZER_URL (OpenAI-compatible)

Usage (from repo root):
    python scripts/04_build_tree.py                  # all domains, rebuild
    python scripts/04_build_tree.py --domain medical # one domain
    python scripts/04_build_tree.py --no-rebuild     # keep existing internal nodes

New internal nodes are auto-indexed by the existing HNSW index on insert, so no
reindex is required; an ANALYZE is run at the end to refresh planner stats.
"""
import sys
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agents.tree_builder.build import build_all
from agents.ingestion.db import connect


def verify() -> None:
    with connect() as conn:
        with conn.cursor() as cur:
            print("\n=== Phase 3 verification ===")
            cur.execute(
                "SELECT level, COUNT(*) FROM tree_nodes GROUP BY level ORDER BY level"
            )
            print("  nodes per level:")
            for lvl, n in cur.fetchall():
                print(f"    level {lvl:<2} {n:>10,}")

            cur.execute(
                """
                SELECT domain, source, MAX(level)
                FROM tree_nodes GROUP BY domain, source ORDER BY domain, source
                """
            )
            print("  max level per (domain, source):")
            for dom, src, mx in cur.fetchall():
                print(f"    {dom:<10} {src:<25} {mx}")

            # every leaf must have a parent (noise reassignment guarantees this)
            cur.execute(
                "SELECT COUNT(*) FROM tree_nodes WHERE level = 0 AND parent_id IS NULL"
            )
            orphans = cur.fetchone()[0]
            print(f"  unparented leaves: {orphans:,}  (must be 0)")

            # descendant accounting: each top node's n_descendants should equal
            # its source's leaf count
            cur.execute(
                """
                SELECT t.domain, t.source, t.n_descendants,
                    (SELECT COUNT(*) FROM tree_nodes l
                        WHERE l.domain=t.domain AND l.source=t.source AND l.level=0)
                FROM tree_nodes t
                WHERE t.parent_id IS NULL AND t.level >= 1
                ORDER BY t.domain, t.source
                """
            )
            print("  top-node descendants vs leaf count:")
            for dom, src, ndesc, nleaf in cur.fetchall():
                flag = "" if ndesc == nleaf else "  <-- MISMATCH"
                print(f"    {dom:<10} {src:<25} {ndesc:>8,} / {nleaf:>8,}{flag}")

    with connect() as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("ANALYZE tree_nodes")
    print("  ANALYZE done.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", nargs="*", default=None,
                    help="subset of domains; default = all")
    ap.add_argument("--no-rebuild", action="store_true",
                    help="do not wipe existing internal nodes before building")
    args = ap.parse_args()

    print("[phase 3] building topic tree...")
    build_all(domains=args.domain, rebuild=not args.no_rebuild)
    verify()


if __name__ == "__main__":
    main()