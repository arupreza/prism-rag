"""Phase 4 — tree-guided retrieval.

Two strategies share one query encoding step (BGE-M3, L2-normalized, 1024-d):

  1) TOP-DOWN BEAM TRAVERSAL
     - Encode query → q.
     - Frontier = top-N roots (parent_id IS NULL AND level >= 1), domain-filtered.
       Cross-domain when domain=None: the tree itself routes the query by
       picking the best-matching root(s). Per-source filter is exact-match.
     - Loop: pull next-level children of the non-leaf frontier nodes, keep the
       top-`beam` by similarity. Leaves on the way are accumulated. Stop when
       no non-leaf nodes remain in the frontier or after MAX_HOPS.
     - Final result: top-`k` leaves under the descended subtree, ordered by
       leaf similarity to q, plus the per-level traversal path.

     Why beam, not greedy: greedy (beam=1) compounds early misclustering errors;
     beam=5-8 buys recall at trivial cost (HNSW + ANY(parent_id IN ...) is cheap).

  2) COLLAPSED + ANCESTOR REWEIGHT
     - Flat HNSW ANN over ALL tree_nodes (any level), domain-filtered, top-N.
     - Split into internal_hits and leaf_hits.
     - For each leaf candidate, walk parent_id up; if any ancestor is in the
       internal_hits set, add α * max(ancestor_sim) bonus to its score.
     - Re-rank leaves by combined score; return top-`k`.

     Why this is often more robust: a leaf that scores moderately on its own but
     sits under a strongly-on-topic cluster wins over a leaf that lexically
     matches the query but is in the wrong topic. The cluster summary acts as a
     topic prior.

SCORING CONVENTION
------------------
pgvector returns `embedding <#> q` = NEGATIVE inner product (so ASC == better).
Because embeddings are L2-normalized and `<#>` on unit vectors equals
-cosine_similarity, we convert to a similarity in [-1, 1] via `sim = -dist`.
Larger `sim` = better. All public outputs use `sim`, never raw distance.

HNSW NOTES
----------
- The index was built with `vector_ip_ops` (see scripts/03_embed_chunks.py).
- `SET LOCAL hnsw.ef_search = N` raises recall at modest latency cost. We set
  it per-query, scoped to the transaction, so it does not leak to other
  connections from a pool.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence
from uuid import UUID

import numpy as np

from agents.ingestion.db import connect
from agents.ingestion.encoder import BGEM3Encoder


# ── tunables ────────────────────────────────────────────────────────────────
HNSW_EF_SEARCH = 80          # 40 (default) is too low for 1024-d at our sizes
MAX_HOPS       = 6           # safety net; deepest source today is ≈ MAX_TREE_LEVELS
DEFAULT_BEAM   = 6
DEFAULT_K      = 5
DEFAULT_FANOUT = 50          # collapsed: how many flat ANN hits before re-rank
DEFAULT_ALPHA  = 0.30        # ancestor-boost weight in collapsed mode


# ── result types ────────────────────────────────────────────────────────────
@dataclass
class NodeHit:
    """One node returned by retrieval. `sim` is cosine similarity in [-1, 1]."""
    node_id: UUID
    level: int
    is_leaf: bool
    domain: str
    source: str | None
    title: str | None
    summary: str
    chunk_id: UUID | None
    parent_id: UUID | None
    n_descendants: int | None
    sim: float

    def to_dict(self) -> dict:
        return {
            "node_id": str(self.node_id),
            "level": self.level,
            "is_leaf": self.is_leaf,
            "domain": self.domain,
            "source": self.source,
            "title": self.title,
            "summary": self.summary,
            "chunk_id": str(self.chunk_id) if self.chunk_id else None,
            "parent_id": str(self.parent_id) if self.parent_id else None,
            "n_descendants": self.n_descendants,
            "sim": round(self.sim, 6),
        }


@dataclass
class RetrievalResult:
    query: str
    mode: str                          # "top_down" | "collapsed"
    domain: str | None
    source: str | None
    k: int
    leaves: list[NodeHit] = field(default_factory=list)
    path: list[list[NodeHit]] = field(default_factory=list)    # one list per descent level
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "mode": self.mode,
            "domain": self.domain,
            "source": self.source,
            "k": self.k,
            "leaves": [h.to_dict() for h in self.leaves],
            "path": [[h.to_dict() for h in lvl] for lvl in self.path],
            "extras": self.extras,
        }


# ── helpers ─────────────────────────────────────────────────────────────────
_SELECT_COLS = (
    "node_id, level, is_leaf, domain, source, title, summary, "
    "chunk_id, parent_id, n_descendants"
)


def _row_to_hit(row: tuple, sim: float) -> NodeHit:
    return NodeHit(
        node_id=row[0], level=row[1], is_leaf=row[2], domain=row[3],
        source=row[4], title=row[5], summary=row[6], chunk_id=row[7],
        parent_id=row[8], n_descendants=row[9], sim=sim,
    )


def _apply_ef_search(cur, ef: int = HNSW_EF_SEARCH) -> None:
    """LOCAL so it scopes to the current txn (pool-safe)."""
    cur.execute(f"SET LOCAL hnsw.ef_search = {int(ef)}")


# ── strategy 1: top-down beam ───────────────────────────────────────────────
def _fetch_roots(cur, q_vec: np.ndarray, *, domain: str | None,
                 source: str | None, beam: int) -> list[NodeHit]:
    """Top-`beam` source-tree roots by similarity to q.

    A root is `parent_id IS NULL AND level >= 1`. Phase 3 builds per-source,
    so domain=None lets the query route across domains by picking the best
    roots; an explicit domain restricts to that domain's roots only.
    """
    where = ["parent_id IS NULL", "level >= 1"]
    args: list = []
    if domain is not None:
        where.append("domain = %s"); args.append(domain)
    if source is not None:
        where.append("source = %s"); args.append(source)
    sql = f"""
        SELECT {_SELECT_COLS}, embedding <#> %s AS dist
        FROM tree_nodes
        WHERE {' AND '.join(where)}
        ORDER BY embedding <#> %s
        LIMIT %s
    """
    cur.execute(sql, (*args, q_vec, q_vec, beam))
    rows = cur.fetchall()
    return [_row_to_hit(r[:-1], -float(r[-1])) for r in rows]


def _fetch_children(cur, q_vec: np.ndarray, parent_ids: Sequence[UUID],
                    *, beam: int) -> list[NodeHit]:
    """Top-`beam` nodes whose parent is in `parent_ids`."""
    if not parent_ids:
        return []
    sql = f"""
        SELECT {_SELECT_COLS}, embedding <#> %s AS dist
        FROM tree_nodes
        WHERE parent_id = ANY(%s)
        ORDER BY embedding <#> %s
        LIMIT %s
    """
    cur.execute(sql, (q_vec, list(parent_ids), q_vec, beam))
    rows = cur.fetchall()
    return [_row_to_hit(r[:-1], -float(r[-1])) for r in rows]


def _fetch_leaves_under(cur, q_vec: np.ndarray, ancestor_ids: Sequence[UUID],
                        *, k: int) -> list[NodeHit]:
    """Top-`k` LEAVES whose tree path passes through any of `ancestor_ids`.

    Recursive CTE walks descendants down to level 0. The candidate set is
    typically small (subtree of a few clusters), so a recursive descent +
    ORDER BY ANN on the leaf set is cheap. We do NOT use the HNSW index here
    on purpose — the filter is the dominant constraint, not similarity.
    """
    if not ancestor_ids:
        return []
    sql = f"""
        WITH RECURSIVE descend AS (
            SELECT node_id FROM tree_nodes WHERE node_id = ANY(%s)
            UNION ALL
            SELECT c.node_id
            FROM tree_nodes c
            JOIN descend d ON c.parent_id = d.node_id
        )
        SELECT {_SELECT_COLS}, embedding <#> %s AS dist
        FROM tree_nodes
        WHERE node_id IN (SELECT node_id FROM descend)
            AND is_leaf = TRUE
        ORDER BY embedding <#> %s
        LIMIT %s
    """
    cur.execute(sql, (list(ancestor_ids), q_vec, q_vec, k))
    rows = cur.fetchall()
    return [_row_to_hit(r[:-1], -float(r[-1])) for r in rows]


def top_down_search(
    q_vec: np.ndarray,
    *,
    domain: str | None,
    source: str | None,
    k: int,
    beam: int,
    query: str,
) -> RetrievalResult:
    result = RetrievalResult(query=query, mode="top_down", domain=domain,
                                source=source, k=k)

    with connect() as conn, conn.cursor() as cur:
        _apply_ef_search(cur)

        frontier = _fetch_roots(cur, q_vec, domain=domain, source=source, beam=beam)
        if not frontier:
            return result
        result.path.append(frontier)

        # The beam descent: at each step, expand only non-leaf frontier members.
        # Leaf frontier members would already be answers, but in practice the
        # roots are internal — leaves appear only at the bottom step.
        for _ in range(MAX_HOPS):
            internal = [h for h in frontier if not h.is_leaf]
            if not internal:
                break
            children = _fetch_children(cur, q_vec, [h.node_id for h in internal],
                                        beam=beam)
            if not children:
                break
            result.path.append(children)
            frontier = children
            if all(h.is_leaf for h in frontier):
                break

        # The "answer set" for top-down is the top-k LEAVES under the final
        # internal frontier (or the leaves of the final frontier if we landed
        # exactly on leaves). We anchor on the LAST internal level so a tiny
        # final-step beam doesn't cap leaf recall artificially.
        last_internal_lvl = next(
            (lvl for lvl in reversed(result.path) if any(not h.is_leaf for h in lvl)),
            None,
        )
        if last_internal_lvl is not None:
            anchors = [h.node_id for h in last_internal_lvl if not h.is_leaf]
            result.leaves = _fetch_leaves_under(cur, q_vec, anchors, k=k)
        else:
            # Edge case: roots themselves were leaves (degenerate single-doc
            # source). Use the path's last level as the answer set.
            result.leaves = sorted(
                [h for h in result.path[-1] if h.is_leaf],
                key=lambda h: -h.sim,
            )[:k]

    return result


# ── strategy 2: collapsed + ancestor reweight ───────────────────────────────
def _flat_ann(cur, q_vec: np.ndarray, *, domain: str | None,
                source: str | None, fanout: int) -> list[NodeHit]:
    where = ["TRUE"]
    args: list = []
    if domain is not None:
        where.append("domain = %s"); args.append(domain)
    if source is not None:
        where.append("source = %s"); args.append(source)
    sql = f"""
        SELECT {_SELECT_COLS}, embedding <#> %s AS dist
        FROM tree_nodes
        WHERE {' AND '.join(where)}
        ORDER BY embedding <#> %s
        LIMIT %s
    """
    cur.execute(sql, (*args, q_vec, q_vec, fanout))
    rows = cur.fetchall()
    return [_row_to_hit(r[:-1], -float(r[-1])) for r in rows]


def _ancestor_map(cur, leaf_ids: Sequence[UUID]) -> dict[UUID, list[UUID]]:
    """For each leaf, return its list of ancestor node_ids (parent → root).

    Used by the collapsed re-rank to find which internal hits sit above which
    leaves. Recursive CTE walks parent_id upward.
    """
    if not leaf_ids:
        return {}
    sql = """
        WITH RECURSIVE up AS (
            SELECT node_id AS leaf_id, parent_id AS anc_id
            FROM tree_nodes
            WHERE node_id = ANY(%s)
            UNION ALL
            SELECT u.leaf_id, t.parent_id
            FROM up u
            JOIN tree_nodes t ON t.node_id = u.anc_id
            WHERE u.anc_id IS NOT NULL
        )
        SELECT leaf_id, anc_id FROM up WHERE anc_id IS NOT NULL
    """
    cur.execute(sql, (list(leaf_ids),))
    out: dict[UUID, list[UUID]] = {lid: [] for lid in leaf_ids}
    for leaf_id, anc_id in cur.fetchall():
        out[leaf_id].append(anc_id)
    return out


def collapsed_search(
    q_vec: np.ndarray,
    *,
    domain: str | None,
    source: str | None,
    k: int,
    fanout: int,
    alpha: float,
    query: str,
) -> RetrievalResult:
    result = RetrievalResult(query=query, mode="collapsed", domain=domain,
                                source=source, k=k,
                                extras={"alpha": alpha, "fanout": fanout})

    with connect() as conn, conn.cursor() as cur:
        _apply_ef_search(cur)

        hits = _flat_ann(cur, q_vec, domain=domain, source=source, fanout=fanout)
        if not hits:
            return result

        leaves = [h for h in hits if h.is_leaf]
        internals = [h for h in hits if not h.is_leaf]
        internal_sim: dict[UUID, float] = {h.node_id: h.sim for h in internals}

        # If the flat ANN brought back almost no leaves (rare but possible when
        # cluster summaries are dense), pull the top-`k*fanout/k` leaves under
        # the internal hits as a fallback candidate set.
        if len(leaves) < k and internals:
            extra = _fetch_leaves_under(
                cur, q_vec, [h.node_id for h in internals], k=max(k * 4, 20)
            )
            seen = {h.node_id for h in leaves}
            leaves += [h for h in extra if h.node_id not in seen]

        if not leaves:
            return result

        anc = _ancestor_map(cur, [h.node_id for h in leaves])

    # Re-rank
    ranked: list[NodeHit] = []
    for leaf in leaves:
        ancestor_sims = [internal_sim[a] for a in anc.get(leaf.node_id, [])
                            if a in internal_sim]
        boost = alpha * max(ancestor_sims) if ancestor_sims else 0.0
        combined = leaf.sim + boost
        ranked.append(NodeHit(
            node_id=leaf.node_id, level=leaf.level, is_leaf=leaf.is_leaf,
            domain=leaf.domain, source=leaf.source, title=leaf.title,
            summary=leaf.summary, chunk_id=leaf.chunk_id, parent_id=leaf.parent_id,
            n_descendants=leaf.n_descendants, sim=combined,
        ))

    ranked.sort(key=lambda h: -h.sim)
    result.leaves = ranked[:k]
    result.path = [internals]
    result.extras["n_internal_hits"] = len(internals)
    result.extras["n_leaf_candidates"] = len(leaves)
    return result


# ── public dispatcher ───────────────────────────────────────────────────────
class TreeSearcher:
    """One encoder, many queries. Reuse across CLI and FastAPI."""

    def __init__(self, encoder: BGEM3Encoder | None = None):
        self.encoder = encoder or BGEM3Encoder()

    def encode_query(self, query: str) -> np.ndarray:
        # encode() returns (1, 1024); pass the 1-D row to pgvector
        return self.encoder.encode([query])[0].astype(np.float32)

    def retrieve(
        self,
        query: str,
        *,
        mode: str = "top_down",
        domain: str | None = None,
        source: str | None = None,
        k: int = DEFAULT_K,
        beam: int = DEFAULT_BEAM,
        fanout: int = DEFAULT_FANOUT,
        alpha: float = DEFAULT_ALPHA,
    ) -> RetrievalResult:
        q_vec = self.encode_query(query)
        if mode == "top_down":
            return top_down_search(
                q_vec, domain=domain, source=source, k=k, beam=beam, query=query,
            )
        if mode == "collapsed":
            return collapsed_search(
                q_vec, domain=domain, source=source, k=k,
                fanout=fanout, alpha=alpha, query=query,
            )
        raise ValueError(f"unknown mode: {mode!r} (use 'top_down' or 'collapsed')")