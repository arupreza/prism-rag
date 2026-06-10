"""
Encoder: BAAI/bge-m3 (1024-dim) by default. Single source of truth for embed dim.
Switch model via env EMB_MODEL.
"""
from __future__ import annotations
import os
from typing import Sequence

import torch
from sentence_transformers import SentenceTransformer

EMB_MODEL = os.getenv("EMB_MODEL", "BAAI/bge-m3")
EMB_DIM = int(os.getenv("EMB_DIM", "1024"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model: SentenceTransformer | None = None


def get_encoder() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMB_MODEL, device=DEVICE)
        if DEVICE == "cuda":
            _model.half()
    return _model


def embed(texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
    m = get_encoder()
    vecs = m.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    assert vecs.shape[1] == EMB_DIM, f"dim mismatch: {vecs.shape[1]} != {EMB_DIM}"
    return vecs.tolist()