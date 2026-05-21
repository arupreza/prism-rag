"""BGE-M3 dense encoder.

Returns L2-normalized 1024-d vectors. Uses FlagEmbedding (the canonical
BGE-M3 library) with fp16 on GPU. Falls back to CPU gracefully.
"""
import numpy as np
from FlagEmbedding import BGEM3FlagModel

from .config import EMBED_MODEL, EMBED_DEVICE, EMBED_BATCH


class BGEM3Encoder:
    def __init__(
        self,
        model_name: str = EMBED_MODEL,
        device: str = EMBED_DEVICE,
        batch_size: int = EMBED_BATCH,
    ):
        self.batch_size = batch_size
        use_fp16 = device.startswith("cuda")
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16, device=device)

    def encode(self, texts: list[str], max_length: int = 512) -> np.ndarray:
        """Encode texts → (N, 1024) L2-normalized float32 array."""
        out = self.model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )["dense_vecs"]  # already float32, (N, 1024)
        # enforce unit norm — BGE-M3 usually returns normalized, but be safe
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms