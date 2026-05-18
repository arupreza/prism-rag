"""Ingest JSONL data into documents + chunks.

Default: PROTOTYPE_SAMPLE_PER_SOURCE docs per source (5,000). Pass --full
ONLY after you've validated every later phase on the prototype subset.

Usage (from repo root):
    python scripts/02_ingest.py                      # all 4 domains, sampled
    python scripts/02_ingest.py politics             # one domain
    python scripts/02_ingest.py politics medical     # multiple domains
    python scripts/02_ingest.py --full               # ignore the sample cap
"""
import sys
from pathlib import Path

# Allow imports of agents.ingestion.* without installing the project as a package.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agents.ingestion.config import (
    DATA_DIR,
    DOMAIN_SOURCES,
    PROTOTYPE_SAMPLE_PER_SOURCE,
)
from agents.ingestion.db import connect
from agents.ingestion.loader import ingest_jsonl


def parse_args(argv: list[str]) -> tuple[list[str], bool]:
    full = "--full" in argv
    domains = [a for a in argv if not a.startswith("--")]
    if not domains:
        domains = list(DOMAIN_SOURCES.keys())

    unknown = [d for d in domains if d not in DOMAIN_SOURCES]
    if unknown:
        raise SystemExit(
            f"Unknown domains: {unknown}. Valid: {list(DOMAIN_SOURCES)}"
        )
    return domains, full


def main(argv: list[str]) -> None:
    domains, full = parse_args(argv)
    sample = None if full else PROTOTYPE_SAMPLE_PER_SOURCE

    print(f"[start] domains={domains}  sample_per_source={sample}")

    with connect() as conn:
        for domain in domains:
            for source in DOMAIN_SOURCES[domain]:
                path = DATA_DIR / domain / f"{source}.jsonl"
                if not path.exists():
                    print(f"[skip] missing file: {path}")
                    continue
                print(f"\n[ingest] {domain}/{source}  path={path}")
                n_docs, n_chunks = ingest_jsonl(conn, path, domain, source, sample)
                print(f"[done]   {domain}/{source}  docs={n_docs:,}  chunks={n_chunks:,}")

        # ── Final stats ──────────────────────────────────────────────────────
        with conn.cursor() as cur:
            cur.execute("""
                SELECT domain, source, COUNT(*)
                FROM documents
                GROUP BY domain, source
                ORDER BY domain, source
            """)
            print("\n=== documents per (domain, source) ===")
            for dom, src, n in cur.fetchall():
                print(f"  {dom:<10} {src:<25} {n:>10,}")

            cur.execute("SELECT COUNT(*) FROM chunks")
            total_chunks = cur.fetchone()[0]
            print(f"\n=== total chunks: {total_chunks:,} ===")

            cur.execute("""
                SELECT MIN(n_tokens), AVG(n_tokens)::int, MAX(n_tokens)
                FROM chunks
            """)
            mn, avg, mx = cur.fetchone()
            print(f"=== chunk token stats: min={mn}  avg={avg}  max={mx} ===")


if __name__ == "__main__":
    main(sys.argv[1:])