"""
Retrieval: search children (is_searchable=TRUE), return parents.
Returns deduplicated parent paragraphs ranked by best child hit.
"""
from __future__ import annotations
import os
from dataclasses import dataclass

import psycopg
from pgvector.psycopg import register_vector

from agents.ingestion.encoder import embed

DSN = os.getenv("PG_DSN", "postgresql://prism:prism@localhost:5433/prism_rag")


@dataclass
class RetrievedPassage:
    parent_id: int
    content: str            # the FULL paragraph (parent text)
    content_type: str
    language: str | None
    page_start: int
    page_end: int
    score: float            # best child score for this parent
    matched_child_id: int
    n_children_hit: int     # how many of this parent's children matched in top-k


def retrieve(
    query: str,
    domain: str,
    *,
    top_k_children: int = 30,
    top_k_parents: int = 6,
):
    qv = embed([query])[0]

    sql = """
    WITH child_hits AS (
        SELECT
            c.id              AS child_id,
            c.parent_chunk_id AS parent_id,
            c.embedding <=> %s::vector AS dist
        FROM chunks c
        WHERE c.domain = %s::domain_t
          AND c.is_searchable = TRUE
          AND c.level = 0
          AND c.parent_chunk_id IS NOT NULL
        ORDER BY c.embedding <=> %s::vector
        LIMIT %s
    ),
    ranked AS (
        SELECT
            parent_id,
            MIN(dist)                     AS best_dist,
            COUNT(*)                      AS n_hits,
            (ARRAY_AGG(child_id ORDER BY dist))[1] AS best_child
        FROM child_hits
        GROUP BY parent_id
    )
    SELECT
        p.id, p.content, p.content_type::text, p.language,
        p.page_start, p.page_end,
        r.best_dist, r.best_child, r.n_hits
    FROM ranked r
    JOIN chunks p ON p.id = r.parent_id
    ORDER BY r.best_dist ASC
    LIMIT %s;
    """

    out: list[RetrievedPassage] = []
    with psycopg.connect(DSN) as c:
        register_vector(c)
        rows = c.execute(
            sql, (qv, domain, qv, top_k_children, top_k_parents)
        ).fetchall()
        for pid, content, ct, lang, ps, pe, dist, child, nhits in rows:
            out.append(RetrievedPassage(
                parent_id=pid, content=content, content_type=ct, language=lang,
                page_start=ps, page_end=pe,
                score=1.0 - float(dist),     # cosine sim
                matched_child_id=child, n_children_hit=nhits,
            ))
    return out


def build_llm_context(passages: list[RetrievedPassage]) -> str:
    """Pack retrieved parents into LLM context. Parents are already coherent paragraphs."""
    parts = []
    for i, p in enumerate(passages, 1):
        head = f"[{i}] (p.{p.page_start}, score={p.score:.3f})"
        body = p.content
        if p.content_type == "code":
            body = f"```{p.language or ''}\n{body}\n```"
        parts.append(f"{head}\n{body}")
    return "\n\n---\n\n".join(parts)