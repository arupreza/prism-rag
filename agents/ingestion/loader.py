"""JSONL → documents + chunks.

Reads one .jsonl file and inserts each non-empty line as a document, splitting
long text into chunks. Idempotent thanks to UNIQUE (source, external_id) —
re-running after a crash just resumes from where it stopped.

Periodic commits (every 1000 docs) cap the loss window on hard failures.
"""
import json
from pathlib import Path

import psycopg

from .chunker import chunk_text, count_tokens


def ingest_jsonl(
    conn: psycopg.Connection,
    path: Path,
    domain: str,
    source: str,
    sample: int | None = None,
) -> tuple[int, int]:
    """Insert documents + chunks from a JSONL file.

    Args:
        conn:    psycopg connection (autocommit OFF).
        path:    path to .jsonl file.
        domain:  e.g. "politics".
        source:  e.g. "cc_news".
        sample:  cap on inserted documents (None = all).

    Returns:
        (n_docs_inserted, n_chunks_inserted)
    """
    n_docs = n_chunks = 0

    with open(path, encoding="utf-8") as f, conn.cursor() as cur:
        for line in f:
            if sample is not None and n_docs >= sample:
                break

            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue   # skip malformed lines

            text = (r.get("text") or "").strip()
            if not text:
                continue   # skip empty docs

            # Insert document. ON CONFLICT makes the call idempotent:
            # if (source, external_id) already exists, RETURNING returns nothing
            # and we skip chunking for that doc.
            cur.execute(
                """INSERT INTO documents
                    (external_id, domain, source, title, text, metadata, n_tokens)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (source, external_id) DO NOTHING
                RETURNING doc_id""",
                (
                    r.get("id"),
                    domain,
                    source,
                    r.get("title"),
                    text,
                    json.dumps(r.get("metadata", {})),
                    count_tokens(text),
                ),
            )
            row = cur.fetchone()
            if not row:
                continue   # already in DB from a previous run
            doc_id = row[0]
            n_docs += 1

            # Chunk and bulk-insert
            pieces = chunk_text(text)
            cur.executemany(
                """INSERT INTO chunks (doc_id, chunk_idx, text, n_tokens)
                VALUES (%s, %s, %s, %s)""",
                [(doc_id, idx, c, count_tokens(c)) for idx, c in enumerate(pieces)],
            )
            n_chunks += len(pieces)

            # Bound loss on crash: commit every 1000 docs.
            if n_docs % 1000 == 0:
                conn.commit()
                print(f"  [{source}] docs={n_docs:,}  chunks={n_chunks:,}")

    conn.commit()
    return n_docs, n_chunks