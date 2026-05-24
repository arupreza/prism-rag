"""Phase 3 orchestrator — build the hierarchical topic tree (RAPTOR).

Per (domain, source):
    load level-0 leaves
    repeat for level = 1..MAX_TREE_LEVELS:
        UMAP + HDBSCAN cluster current nodes
        summarize each cluster (Qwen-32B)  -> internal node text
        batch-embed summaries (BGE-M3)     -> internal node vectors
        insert internal nodes, link children -> parent
        stop when a level collapses to a single cluster
    next level operates on the summaries just created

Builds PER SOURCE, not per domain: the architecture routes domain → source →
clusters → leaves, so each source gets its own subtree and `source` stays
non-null on every internal node. (The schema's "source NULL only at root"
comment anticipates an optional domain-level root tying sources together — not
created here; Phase 4 retrieval filters by domain and walks parent_id, which
does not require level uniformity across sources.)
"""
from __future__ import annotations

import numpy as np
from psycopg.types.json import Jsonb

from agents.ingestion.config import (
    DOMAIN_SOURCES,
    MAX_TREE_LEVELS,
    MIN_CLUSTER_SIZE,
    UMAP_N_COMPONENTS,
)
from agents.ingestion.db import connect
from agents.ingestion.encoder import BGEM3Encoder
from agents.tree_builder.cluster import cluster_embeddings
from agents.tree_builder.summarizer import ClusterSummarizer


# ── DB helpers ───────────────────────────────────────────────────────────────
def _load_level0(conn, domain: str, source: str):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT node_id, embedding, summary, n_descendants
            FROM tree_nodes
            WHERE domain = %s AND source = %s AND level = 0
            ORDER BY node_id
            """,
            (domain, source),
        )
        rows = cur.fetchall()
    ids = [r[0] for r in rows]
    emb = np.asarray([r[1] for r in rows], dtype=np.float32)
    texts = [r[2] for r in rows]
    desc = [int(r[3]) for r in rows]
    return ids, emb, texts, desc


def reset_subtree(conn, domain: str, source: str) -> None:
    """Idempotent rebuild prep.

    FOOTGUN: tree_nodes.parent_id is `REFERENCES tree_nodes ON DELETE CASCADE`.
    Leaves point at level-1 nodes, so deleting internal nodes would CASCADE and
    delete the leaves too — wiping all of Phase 2. Break every link first, then
    delete internal nodes (now childless, so nothing cascades into leaves).
    """
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE tree_nodes SET parent_id = NULL WHERE domain = %s AND source = %s",
            (domain, source),
        )
        cur.execute(
            "DELETE FROM tree_nodes WHERE domain = %s AND source = %s AND level >= 1",
            (domain, source),
        )
    conn.commit()


def _insert_internal(
    conn, domain, source, level, title, summary, vec, n_desc, meta, child_ids
):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO tree_nodes
              (domain, source, level, is_leaf, parent_id, chunk_id,
               title, summary, n_descendants, cluster_meta, embedding, embed_input)
            VALUES (%s, %s, %s, false, NULL, NULL,
                    %s, %s, %s, %s, %s, %s)
            RETURNING node_id
            """,
            (domain, source, level, title, summary, n_desc, Jsonb(meta), vec, summary),
        )
        nid = cur.fetchone()[0]
        cur.execute(
            "UPDATE tree_nodes SET parent_id = %s WHERE node_id = ANY(%s)",
            (nid, child_ids),
        )
    return nid


# ── core build ───────────────────────────────────────────────────────────────
def build_subtree(domain, source, encoder, summarizer, *, rebuild: bool = True) -> None:
    with connect() as conn:
        if rebuild:
            reset_subtree(conn, domain, source)

        cur_ids, cur_emb, cur_texts, cur_desc = _load_level0(conn, domain, source)
        if len(cur_ids) < 2:
            print(f"  [{domain}/{source}] {len(cur_ids)} leaves — skip")
            return
        print(f"  [{domain}/{source}] {len(cur_ids):,} leaves")

        for level in range(1, MAX_TREE_LEVELS + 1):
            if len(cur_ids) <= 1:
                break

            labels, sil = cluster_embeddings(
                cur_emb,
                min_cluster_size=MIN_CLUSTER_SIZE,
                n_components=UMAP_N_COMPONENTS,
            )
            uniq = np.unique(labels)
            if uniq.size <= 1:
                print(f"    level {level}: 1 cluster — stop")
                break
            print(f"    level {level}: {len(cur_ids):,} -> {uniq.size} clusters "
                f"(silhouette={sil})")

            # 1) summarize each cluster (LLM)
            pending = []  # (title, summary, child_ids, n_desc, size)
            for c in uniq:
                idx = np.where(labels == c)[0]
                title, summary = summarizer.summarize([cur_texts[i] for i in idx])
                pending.append(
                    (
                        title,
                        summary,
                        [cur_ids[i] for i in idx],
                        int(sum(cur_desc[i] for i in idx)),
                        int(idx.size),
                    )
                )

            # 2) one batch encode for the whole level
            vecs = encoder.encode([p[1] for p in pending])

            # 3) insert + link
            new_ids, new_texts, new_desc = [], [], []
            for (title, summary, child_ids, n_desc, size), vec in zip(pending, vecs):
                meta = {
                    "method": "umap+hdbscan",
                    "level": level,
                    "size": size,
                    "silhouette": sil,
                }
                nid = _insert_internal(
                    conn, domain, source, level, title, summary, vec, n_desc, meta, child_ids
                )
                new_ids.append(nid)
                new_texts.append(summary)
                new_desc.append(n_desc)
            conn.commit()

            cur_ids, cur_emb, cur_texts, cur_desc = new_ids, vecs, new_texts, new_desc


def build_all(domains: list[str] | None = None, rebuild: bool = True) -> None:
    encoder = BGEM3Encoder()
    summarizer = ClusterSummarizer()
    targets = DOMAIN_SOURCES if domains is None else {d: DOMAIN_SOURCES[d] for d in domains}
    for domain, sources in targets.items():
        for source in sources:
            build_subtree(domain, source, encoder, summarizer, rebuild=rebuild)