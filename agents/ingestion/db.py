"""
Postgres writer (psycopg3 + pgvector).
Parent-child aware: parent rows are storage-only, children are searchable.
Image chunks carry image_path so retrieval can display the figure.
"""
from __future__ import annotations
import os
import numpy as np
from contextlib import contextmanager

import psycopg
from psycopg.types.json import Json
from pgvector.psycopg import register_vector

DSN = os.getenv("PG_DSN", "postgresql://prism:prism@localhost:5433/prism_rag")


def _clean(s):
    """Strip NUL (0x00) bytes — illegal in Postgres text/jsonb. Other chars kept."""
    return s.replace("\x00", "") if isinstance(s, str) else s


@contextmanager
def conn():
    with psycopg.connect(DSN, autocommit=False) as c:
        register_vector(c)
        yield c


def upsert_document(c, *, domain, source_path, title, n_pages, sha256, metadata=None):
    row = c.execute(
        """
        INSERT INTO documents (domain, source_path, title, n_pages, sha256, metadata)
        VALUES (%s,%s,%s,%s,%s,%s)
        ON CONFLICT (source_path) DO UPDATE SET
            title=EXCLUDED.title, n_pages=EXCLUDED.n_pages, sha256=EXCLUDED.sha256
        RETURNING id;
        """,
        (domain, source_path, _clean(title), n_pages, sha256, Json(metadata or {})),
    ).fetchone()
    return row[0]


def insert_parent(c, *, document_id, domain, parent, mean_emb):
    """Insert non-searchable parent row. mean_emb = mean of children embeddings."""
    row = c.execute(
        """
        INSERT INTO chunks
          (document_id, domain, level, content, content_type, language,
           page_start, page_end, token_count, embedding,
           parent_chunk_id, is_searchable, image_path)
        VALUES (%s,%s,0,%s,%s,%s,%s,%s,%s,%s,NULL,FALSE,%s)
        RETURNING id;
        """,
        (
            document_id, domain,
            _clean(parent.content), parent.content_type, parent.language,
            parent.page_start, parent.page_end, parent.token_count,
            mean_emb, parent.image_path,
        ),
    ).fetchone()
    return row[0]


def insert_children(c, *, document_id, domain, parent_id, children, embeddings):
    """Insert searchable child rows. Returns list of new ids."""
    ids = []
    sql = """
        INSERT INTO chunks
            (document_id, domain, level, content, content_type, language,
            page_start, page_end, token_count, embedding,
            parent_chunk_id, is_searchable, image_path)
        VALUES (%s,%s,0,%s,%s,%s,%s,%s,%s,%s,%s,TRUE,%s)
        RETURNING id;
    """
    for ck, emb in zip(children, embeddings):
        rid = c.execute(
            sql,
            (
                document_id, domain,
                _clean(ck.content), ck.content_type, ck.language,
                ck.page_start, ck.page_end, ck.token_count,
                emb, parent_id, ck.image_path,
            ),
        ).fetchone()[0]
        ids.append(rid)
    return ids


def insert_summary(c, *, domain, level, cluster_id, children_ids, content, embedding):
    """RAPTOR summary node (level >= 1). Always searchable."""
    content = _clean(content)
    row = c.execute(
        """
        INSERT INTO chunks
            (domain, level, cluster_id, children_ids, content, content_type,
            token_count, embedding, is_searchable)
        VALUES (%s,%s,%s,%s,%s,'text',%s,%s,TRUE)
        RETURNING id;
        """,
        (domain, level, cluster_id, children_ids, content,
            len(content.split()), embedding),
    ).fetchone()
    c.execute(
        "UPDATE chunks SET parent_ids = parent_ids || ARRAY[%s]::bigint[] "
        "WHERE id = ANY(%s);",
        (row[0], children_ids),
    )
    return row[0]


def mean_pool(embs: list[list[float]]) -> list[float]:
    arr = np.array(embs, dtype=np.float32)
    v = arr.mean(axis=0)
    n = np.linalg.norm(v)
    return (v / n).tolist() if n > 0 else v.tolist()