"""Embed chunks as level-0 tree_nodes.

Reads chunks that don't yet have a corresponding tree_node, encodes in
batches, inserts with all required fields. Idempotent — safe to re-run
after a crash.
"""
from __future__ import annotations

import psycopg
from psycopg.rows import dict_row

from .config import EMBED_BATCH
from .db import connect
from .encoder import BGEM3Encoder


READ_BATCH = 2000  # rows fetched from DB per round


def embed_all_leaves(encoder: BGEM3Encoder | None = None) -> int:
    """Embed every un-embedded chunk. Returns total leaves inserted."""
    if encoder is None:
        encoder = BGEM3Encoder()

    total = 0

    with connect() as conn:
        # server-side cursor for streaming large result sets
        cur = conn.cursor(name="chunk_stream", row_factory=dict_row, withhold=True)
        cur.execute("""
            SELECT c.chunk_id, c.text, c.n_tokens, c.doc_id,
                d.domain, d.source
            FROM chunks c
            JOIN documents d ON d.doc_id = c.doc_id
            WHERE NOT EXISTS (
                SELECT 1 FROM tree_nodes t WHERE t.chunk_id = c.chunk_id
            )
            ORDER BY c.chunk_id
        """)

        while True:
            rows = cur.fetchmany(READ_BATCH)
            if not rows:
                break

            texts = [r["text"] for r in rows]
            vecs = encoder.encode(texts)

            with conn.cursor() as w:
                w.executemany(
                    """INSERT INTO tree_nodes
                    (domain, source, level, is_leaf, chunk_id,
                        summary, embed_input, embedding, n_descendants)
                    VALUES (%(domain)s, %(source)s, 0, true, %(chunk_id)s,
                            %(text)s, %(text)s, %(vec)s, 1)
                    ON CONFLICT (chunk_id)
                    DO NOTHING""",
                    [
                        {
                            "domain":   r["domain"],
                            "source":   r["source"],
                            "chunk_id": r["chunk_id"],
                            "text":     r["text"],
                            "vec":      v,
                        }
                        for r, v in zip(rows, vecs)
                    ],
                )
            conn.commit()
            total += len(rows)
            print(f"  [embed] {total:,} leaves inserted", flush=True)

    return total