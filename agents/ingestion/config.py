"""Ingestion pipeline configuration.

Single place for paths, model names, hyperparameters. Anything tunable lives
here. Environment variables override defaults where it makes sense.

NOTE: Despite living under agents/ingestion/, this is currently a BATCH JOB,
not a long-running service. The agents/ folder is preserved for symmetry with
the original repo layout — when retrieval-agent and generation-agent come
online in later phases, they'll share some of these constants. We'll refactor
into a top-level `common/` package then if duplication becomes a problem.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR     = Path(os.getenv("PRISM_DATA_DIR", "/home/lisa/Arupreza/PRISM-RAG/data"))


# ── Domain → source mapping ──────────────────────────────────────────────────
# Each source name must match the JSONL filename (without .jsonl) produced by
# your downloader. Edit here if you rename files.
DOMAIN_SOURCES: dict[str, list[str]] = {
    "politics": ["cc_news", "congressional_speeches"],
    "finance":  ["financial_news"],
    "ai_tech":  ["ml_arxiv_papers"],
    "medical":  ["pubmed_papers", "arxiv_papers"],
}


# ── Database ─────────────────────────────────────────────────────────────────
PG_DSN = os.getenv(
    "PG_DSN",
    "postgresql://postgres:postgres@localhost:5432/prism_rag",
)


# ── Embedding model (Phase 2) ────────────────────────────────────────────────
EMBED_MODEL  = "BAAI/bge-m3"
EMBED_DIM    = 1024
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cuda")   # "cpu" if no GPU
EMBED_BATCH  = 64


# ── Chunking ─────────────────────────────────────────────────────────────────
# 512 / 64 is the standard BGE-class sweet spot. Don't change without measuring
# retrieval quality on a held-out set.
CHUNK_TOKENS  = 512
CHUNK_OVERLAP = 64


# ── Tree build (Phase 3, here for visibility) ───────────────────────────────
MAX_TREE_LEVELS   = 4          # levels above the leaves
MIN_CLUSTER_SIZE  = 8          # HDBSCAN minimum
UMAP_N_COMPONENTS = 10         # dim reduction target before clustering


# ── Summarizer LLM (Phase 3) ─────────────────────────────────────────────────
SUMMARIZER_MODEL = "Qwen/Qwen2.5-32B-Instruct"
SUMMARIZER_URL   = os.getenv("SUMMARIZER_URL", "http://localhost:8000/v1")  # vLLM OpenAI-compat
SUMMARIZER_KEY   = os.getenv("SUMMARIZER_KEY", "EMPTY")                    # vLLM ignores it


# ── Prototype guard ──────────────────────────────────────────────────────────
# Cap docs per source so the full pipeline runs end-to-end in minutes, not
# days. Set to None ONLY after every phase has been validated on the sample.
PROTOTYPE_SAMPLE_PER_SOURCE: int | None = 5000