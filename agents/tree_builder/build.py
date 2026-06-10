"""
RAPTOR tree builder.
Algorithm:
  level 0 = leaf chunks (already in DB)
  while n_nodes_at_level > MIN_CLUSTER:
     reduce dim (UMAP)
     soft-cluster (BIC-selected GMM, threshold prob)
     for each cluster: concat children text -> LLM summary -> embed -> store as parent at next level
"""
from __future__ import annotations
import os
import numpy as np
from dataclasses import dataclass
from typing import Callable

import umap
from sklearn.mixture import GaussianMixture

UMAP_DIM = int(os.getenv("RAPTOR_UMAP_DIM", "10"))
MAX_LEVELS = int(os.getenv("RAPTOR_MAX_LEVELS", "3"))
MIN_CLUSTER = int(os.getenv("RAPTOR_MIN_CLUSTER", "5"))
PROB_THRESHOLD = float(os.getenv("RAPTOR_PROB_THRESHOLD", "0.10"))
MAX_K = int(os.getenv("RAPTOR_MAX_K", "50"))
RNG = 42


@dataclass
class Node:
    id: int           # placeholder ID; real ID assigned after DB insert
    level: int
    content: str
    embedding: list[float]
    children: list[int]
    cluster_id: int | None = None


def _reduce(emb: np.ndarray, n_neighbors: int | None = None) -> np.ndarray:
    n = emb.shape[0]
    if n <= UMAP_DIM + 1:
        return emb
    nn = n_neighbors or max(2, min(int(n ** 0.5), n - 1))
    reducer = umap.UMAP(
        n_neighbors=nn,
        n_components=min(UMAP_DIM, n - 2),
        metric="cosine",
        random_state=RNG,
    )
    return reducer.fit_transform(emb)


def _best_k(emb: np.ndarray) -> int:
    n = emb.shape[0]
    max_k = min(MAX_K, n - 1)
    if max_k < 2:
        return 1
    bics = []
    ks = list(range(1, max_k + 1))
    for k in ks:
        gm = GaussianMixture(n_components=k, random_state=RNG, reg_covar=1e-4)
        gm.fit(emb)
        bics.append(gm.bic(emb))
    return ks[int(np.argmin(bics))]


def _soft_cluster(emb: np.ndarray) -> list[list[int]]:
    """Return list of cluster_id -> [node_indices] (soft assignment)."""
    if emb.shape[0] < 2:
        return [[i for i in range(emb.shape[0])]]
    k = _best_k(emb)
    if k <= 1:
        return [[i for i in range(emb.shape[0])]]
    gm = GaussianMixture(n_components=k, random_state=RNG, reg_covar=1e-4).fit(emb)
    probs = gm.predict_proba(emb)
    clusters: list[list[int]] = [[] for _ in range(k)]
    for i in range(emb.shape[0]):
        for c in range(k):
            if probs[i, c] >= PROB_THRESHOLD:
                clusters[c].append(i)
    # ensure each node lands somewhere
    assigned = {i for cl in clusters for i in cl}
    if len(assigned) < emb.shape[0]:
        argmax = probs.argmax(axis=1)
        for i in range(emb.shape[0]):
            if i not in assigned:
                clusters[int(argmax[i])].append(i)
    return [c for c in clusters if c]


# --- Summarizer hook (inject your LLM call) ---------------------------------
Summarizer = Callable[[list[str], str], str]
# signature: summarizer(child_texts, domain) -> summary_text


def build_tree(
    leaf_ids: list[int],
    leaf_embeddings: list[list[float]],
    leaf_contents: list[str],
    domain: str,
    summarizer: Summarizer,
    embed_fn: Callable[[list[str]], list[list[float]]],
) -> list[Node]:
    """
    Returns parent nodes only (levels 1..MAX_LEVELS). Leaves stay as already-inserted DB rows.
    Caller is responsible for inserting Node rows and wiring parent_ids/children_ids.
    """
    parents: list[Node] = []
    cur_ids = list(leaf_ids)
    cur_emb = np.array(leaf_embeddings, dtype=np.float32)
    cur_txt = list(leaf_contents)

    for level in range(1, MAX_LEVELS + 1):
        if len(cur_ids) <= MIN_CLUSTER:
            break
        reduced = _reduce(cur_emb)
        clusters = _soft_cluster(reduced)
        if len(clusters) <= 1:
            break

        next_ids: list[int] = []
        next_emb: list[list[float]] = []
        next_txt: list[str] = []
        summaries: list[str] = []
        cluster_children: list[list[int]] = []

        for c_idx, members in enumerate(clusters):
            child_texts = [cur_txt[i] for i in members]
            child_ids = [cur_ids[i] for i in members]
            summary = summarizer(child_texts, domain)
            summaries.append(summary)
            cluster_children.append(child_ids)

        emb_batch = embed_fn(summaries)
        # Node IDs here are negative placeholders; caller reassigns after INSERT RETURNING id
        for c_idx, (summary, emb, children) in enumerate(
            zip(summaries, emb_batch, cluster_children)
        ):
            placeholder = -(len(parents) + 1)
            node = Node(
                id=placeholder,
                level=level,
                content=summary,
                embedding=emb,
                children=children,
                cluster_id=c_idx,
            )
            parents.append(node)
            next_ids.append(placeholder)
            next_emb.append(emb)
            next_txt.append(summary)

        cur_ids = next_ids
        cur_emb = np.array(next_emb, dtype=np.float32)
        cur_txt = next_txt

    return parents