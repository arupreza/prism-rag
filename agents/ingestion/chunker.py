"""Token-aware overlapping text chunker.

We chunk on TOKENS (not characters or words) because the embedding model has
a hard token limit. The tokenizer used here MUST be the same one the encoder
will use later — that's why we load BGE-M3's tokenizer specifically.

Overlap preserves context across chunk boundaries: a sentence cut in half is
still retrievable from either chunk.

The tokenizer is loaded once at module import (small, CPU-only, fast).
"""
from transformers import AutoTokenizer

from .config import EMBED_MODEL, CHUNK_TOKENS, CHUNK_OVERLAP


_TOK = AutoTokenizer.from_pretrained(EMBED_MODEL)


def chunk_text(text: str,
            chunk_tokens: int = CHUNK_TOKENS,
            overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into <=chunk_tokens-token pieces with `overlap` token overlap.

    Returns [text] unchanged if it already fits in one chunk — most news
    articles will be a single chunk; full PubMed papers will be 10–30.
    """
    ids = _TOK.encode(text, add_special_tokens=False)
    if len(ids) <= chunk_tokens:
        return [text]

    step = chunk_tokens - overlap
    chunks: list[str] = []
    for i in range(0, len(ids), step):
        window = ids[i:i + chunk_tokens]
        chunks.append(_TOK.decode(window))
        if i + chunk_tokens >= len(ids):
            break
    return chunks


def count_tokens(text: str) -> int:
    """Return the BGE-M3 token count of `text`. Cheap; no need to cache."""
    return len(_TOK.encode(text, add_special_tokens=False))