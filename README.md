# PRISM-RAG

### Progressive Retrieval with Indexed Summary Memory

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white) ![FastAPI](https://img.shields.io/badge/FastAPI-Latest-009688?style=flat-square&logo=fastapi&logoColor=white) ![Docker](https://img.shields.io/badge/Docker-Latest-2496ED?style=flat-square&logo=docker&logoColor=white) ![LangGraph](https://img.shields.io/badge/LangGraph-Latest-1C3C3C?style=flat-square&logo=langchain&logoColor=white) ![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black) ![pgvector](https://img.shields.io/badge/pgvector-sparsevec-336791?style=flat-square&logo=postgresql&logoColor=white) ![SPLADE](https://img.shields.io/badge/SPLADE-Vectorless-6C3483?style=flat-square&logoColor=white) ![Qwen](https://img.shields.io/badge/Qwen2.5--7B-AWQ%204bit-FF6F00?style=flat-square&logoColor=white) ![BEIR](https://img.shields.io/badge/BEIR-Benchmark-2ECC71?style=flat-square&logoColor=white) ![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

> Tree-guided multi-domain Retrieval-Augmented Generation. Build a hierarchical
> topic tree across a curated 4-domain corpus, embed every node (leaf chunks +
> LLM-summarized internal clusters) as dense vectors in pgvector HNSW, route
> queries by traversing the tree top-down, generate cited answers with Qwen2.5.

> **Status:** Phase 1 of 7 complete. Full architecture and roadmap in [`ARCHITECTURE.md`](./ARCHITECTURE.md).
> Performance numbers are intentionally absent — they will appear here once
> they are measured on a fixed eval set (Phase 7), not before.

---

## Project history

This repository originally described a SPLADE sparse multi-agent RAG over BEIR
datasets. That design was dropped in favor of tree-guided dense retrieval over
a curated multi-domain corpus. The original design is visible in early git
history. Reasoning for the pivot is documented in [`ARCHITECTURE.md`](./ARCHITECTURE.md).

---

## Idea in one paragraph

The corpus is four heterogeneous domains (politics, finance, AI/ML, medical).
Flat dense retrieval over a unified corpus mixes unrelated topics in the
top-k. PRISM-RAG instead builds a hierarchical topic tree: chunks are leaves;
HDBSCAN groups them into clusters; Qwen2.5-32B writes a title + summary for
each cluster; that summary becomes a higher-level node with its own
embedding. The tree has roughly 4–5 levels. At query time we traverse
top-down, eliminating irrelevant subtrees at each level, then ANN-search
inside the surviving subtree.

The approach follows RAPTOR (Sarthi et al., ICLR 2024) applied to a
multi-domain corpus, implemented end-to-end in PostgreSQL + pgvector HNSW
with no external vector service.

---

## Architecture

```
                ┌────────────────────────────────────────────────┐
                │            ONLINE  (Phases 4–5)                │
                ├────────────────────────────────────────────────┤
                │                                                │
   User Query ─►│  Gateway (FastAPI + LangGraph)                 │
                │      │                                         │
                │      ▼                                         │
                │  Retrieval Agent  ── tree-guided search        │
                │      │   (top-down traversal: domain → source  │
                │      │    → topic clusters → leaf chunks)      │
                │      ▼                                         │
                │  Generation Agent ── Qwen2.5-7B (+AWQ in v0.2) │
                │      │   (cited answer)                        │
                │      ▼                                         │
                │  Final Answer ───────────────────────► User    │
                └──────────────────────┬─────────────────────────┘
                                       │ reads
                                       ▼
                ┌────────────────────────────────────────────────┐
                │            OFFLINE  (Phases 1–3)               │
                ├────────────────────────────────────────────────┤
                │                                                │
                │  data/{domain}/*.jsonl                         │
                │      │                                         │
                │      ▼  ingest + token-aware chunk    (Phase 1)│
                │  documents + chunks tables                     │
                │      │                                         │
                │      ▼  BGE-M3 dense encode           (Phase 2)│
                │  tree_nodes  (level 0 = leaf chunks)           │
                │      │                                         │
                │      ▼  UMAP → HDBSCAN → Qwen-32B     (Phase 3)│
                │      │  summarize → embed → store             │
                │  tree_nodes  (levels 1..k = cluster summaries) │
                │      │                                         │
                │      ▼  one HNSW index over ALL levels         │
                │  pgvector tree (1 table, 1 index, level filter)│
                │                                                │
                └────────────────────────────────────────────────┘
```

---

## Corpus (4 domains)

Sourced via HuggingFace Datasets. Downloader script in `data/download.py`.

| Domain | Source | Description | Default cap |
|---|---|---|---|
| politics | `vblagoje/cc_news` | English news articles 2017–2019 | 708K (all) |
| politics | `Eugleo/us-congressional-speeches` | US Congressional speeches 1873–2024 | 700K |
| finance | `Brianferrell787/financial-news-multisource` | Yahoo / CNBC / S&P 500 financial news | 700K |
| ai_tech | `CShorten/ML-ArXiv-Papers` | ML & AI ArXiv titles + abstracts | 118K (all) |
| medical | `armanc/scientific_papers` (`pubmed`) | PubMed full-text papers | ~120K (all) |
| medical | `armanc/scientific_papers` (`arxiv`) | ArXiv full-text papers | ~203K (all) |

Total ≈ 2.5M documents at default caps.

A prototype subset of 5K documents per source is used through Phase 4 to
keep iteration cycles short.

---

## Stack

| Component | Technology |
|---|---|
| Database | PostgreSQL 14+ |
| Vector store | `pgvector` extension, `vector(1024)` columns, HNSW index |
| Embedding model | `BAAI/bge-m3` (1024-d, 8K context) |
| Clustering | UMAP (dim reduction) + HDBSCAN (cluster discovery) |
| Cluster summarizer | `Qwen/Qwen2.5-32B-Instruct` via vLLM (OpenAI-compatible endpoint) |
| Generator | `Qwen/Qwen2.5-7B-Instruct` (v0.1) → QLoRA + AWQ in v0.2 |
| Orchestration | LangGraph StateGraph (Phase 5) |
| API | FastAPI + uvicorn (Phase 5) |
| Container runtime | Docker + docker-compose (Phase 5) |
| Experiment tracking | Weights & Biases (Phase 6) |

---

## Build status

| Phase | Component | Status |
|---|---|---|
| 1 | DB schema + JSONL ingest + token-aware chunking | ✓ Done |
| 2 | BGE-M3 dense embedding of chunks as leaf nodes | → In progress |
| 3 | UMAP + HDBSCAN + Qwen-32B cluster summaries (tree build) | ⏳ Planned |
| 4 | Tree-guided retrieval (CLI, then FastAPI agent) | ⏳ Planned |
| 5 | Qwen-7B generation + LangGraph gateway + docker-compose | ⏳ Planned |
| 6 | QLoRA fine-tune + AWQ quantization per domain *(optional v0.2)* | ⏳ Planned |
| 7 | Synthetic eval set + retrieval & generation metrics | ⏳ Planned |

Phase-by-phase file list, verification queries, and decision points are in
[`ARCHITECTURE.md`](./ARCHITECTURE.md).

---

## Project structure

```
PRISM-RAG/
├── README.md                    ← this file
├── ARCHITECTURE.md              ← roadmap, schema, phase verifications
├── init.sql                     ← Postgres schema (documents, chunks, tree_nodes)
├── requirements.txt
├── .env.example
├── data/                        ← downloaded JSONL (politics/finance/ai_tech/medical)
├── checkpoints/                 ← Phase 6: LoRA adapters, AWQ models
├── agents/
│   ├── ingestion/               ← Phase 1–2: batch ingest + embed
│   ├── tree_builder/            ← Phase 3: cluster + summarize (planned)
│   ├── retrieval/               ← Phase 4: tree-guided search (planned)
│   └── generation/              ← Phase 5: Qwen-AWQ inference (planned)
├── gateway/                     ← Phase 5: FastAPI + LangGraph (planned)
├── training/                    ← Phase 6: QLoRA → merge → AWQ (planned)
├── evaluation/                  ← Phase 7: synthetic eval + metrics (planned)
└── scripts/                     ← numbered entry points (01_init_db, 02_ingest, ...)
```

---

## Quick start (Phase 1 — what currently works)

Prereqs: Python 3.10+, PostgreSQL 14+ with the `pgvector` extension
available, ~10 GB free disk.

```bash
git clone https://github.com/Arupreza/PRISM-RAG
cd PRISM-RAG

# Install deps
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Configure
cp .env.example .env
# edit .env: set PG_DSN

# (One-time) Download the corpus
python data/download.py             # produces data/{domain}/*.jsonl

# Initialize the database
createdb prism_rag
python scripts/01_init_db.py

# Ingest the prototype sample (5K docs per source)
python scripts/02_ingest.py
```

Verify Phase 1:

```sql
-- Document counts per (domain, source)
SELECT domain, source, COUNT(*) FROM documents GROUP BY 1,2 ORDER BY 1,2;

-- Chunk token-length distribution (max MUST be ≤ 512)
SELECT MIN(n_tokens), AVG(n_tokens)::int, MAX(n_tokens) FROM chunks;
```

Phase 2+ commands will appear here as each phase ships.

---

## What this project is and isn't

**Is:**
- A learning + research project exploring whether hierarchical topic trees
  built by RAPTOR-style recursive clustering improve retrieval over flat
  dense ANN on a heterogeneous multi-domain corpus.
- A fully open, single-database implementation (everything in PostgreSQL +
  pgvector — no external vector service).

**Isn't:**
- Production-ready.
- Yet benchmarked. No performance comparison against flat BM25, flat dense,
  or other RAG baselines is claimed here. Those numbers will be reported in
  Phase 7 on a synthetic evaluation set.
- A drop-in replacement for any commercial RAG product.

---

## License

MIT.

---

## Author

**Md Rezanur Islam (Reza)**
LLM Engineer & Agentic AI Developer
PhD Candidate, Software Convergence — Soonchunhyang University (BK21)