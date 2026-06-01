"""Phase 3 clustering — UMAP dimensionality reduction + HDBSCAN.

One level of clustering: reduce 1024-d BGE-M3 vectors to a low-d manifold,
discover clusters by density, then reassign HDBSCAN noise (-1) to the nearest
cluster centroid.

Why reassign noise: HDBSCAN labels low-density points as -1. If left as noise,
those leaves get no parent and silently vanish from every level above the
leaves — the tree would lose documents. Reassignment guarantees every node is
covered exactly once.

umap-learn and hdbscan are imported lazily so this module compiles without them
installed (and so `import` cost is paid only when a build actually runs).
"""
from __future__ import annotations

import numpy as np


def reduce_dim(
    emb: np.ndarray,
    n_components: int,
    n_neighbors: int = 15,
    metric: str = "cosine",
    random_state: int = 42,
) -> np.ndarray:
    """UMAP reduce. Skipped if too few points to form the manifold."""
    n = len(emb)
    if n <= n_components + 2:
        return emb.astype(np.float32)
    import umap

    reducer = umap.UMAP(
        n_neighbors=min(n_neighbors, n - 1),
        n_components=min(n_components, n - 2),
        metric=metric,
        random_state=random_state,
        low_memory=(n > 20_000),   # ← ADD
        verbose=(n > 20_000),      # ← ADD: see progress
    )
    return reducer.fit_transform(emb).astype(np.float32)


def _hdbscan_labels(reduced: np.ndarray, min_cluster_size: int) -> np.ndarray:
    import hdbscan
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=max(2, min_cluster_size),
        min_samples=None,
        metric="euclidean",
        cluster_selection_method="eom",
        core_dist_n_jobs=-1,       # ← ADD
    )
    labels = clusterer.fit_predict(reduced)
    n_valid = len(np.unique(labels[labels != -1]))
    n_noise = int((labels == -1).sum())
    print(f"      HDBSCAN: n={len(reduced)} mcs={min_cluster_size} "
            f"valid_clusters={n_valid} noise={n_noise}")
    return labels


def _reassign_noise(reduced: np.ndarray, labels: np.ndarray) -> np.ndarray:
    labels = labels.copy()
    valid = np.unique(labels[labels != -1])
    if valid.size == 0:
        return np.zeros_like(labels)            # all noise -> single cluster
    centroids = np.stack([reduced[labels == c].mean(axis=0) for c in valid])
    for i in np.where(labels == -1)[0]:
        d = np.linalg.norm(centroids - reduced[i], axis=1)
        labels[i] = valid[int(np.argmin(d))]
    return labels


def cluster_embeddings(
    emb: np.ndarray,
    *,
    min_cluster_size: int,
    n_components: int,
    n_neighbors: int = 15,
) -> tuple[np.ndarray, float | None]:
    """Return (labels, silhouette). Labels are dense ints, no -1 remaining."""
    reduced = reduce_dim(emb, n_components, n_neighbors)
    labels = _hdbscan_labels(reduced, min_cluster_size)
    labels = _reassign_noise(reduced, labels)

    sil: float | None = None
    if np.unique(labels).size > 1:
        try:
            from sklearn.metrics import silhouette_score

            sil = float(silhouette_score(reduced, labels))
        except Exception:
            sil = None
    return labels, sil