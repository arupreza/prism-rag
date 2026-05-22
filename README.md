# PRISM-RAG

### Progressive Retrieval with Indexed Summary Memory

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white) ![FastAPI](https://img.shields.io/badge/FastAPI-Latest-009688?style=flat-square&logo=fastapi&logoColor=white) ![Docker](https://img.shields.io/badge/Docker-Latest-2496ED?style=flat-square&logo=docker&logoColor=white) ![LangGraph](https://img.shields.io/badge/LangGraph-Latest-1C3C3C?style=flat-square&logo=langchain&logoColor=white) ![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=white) ![pgvector](https://img.shields.io/badge/pgvector-HNSW-336791?style=flat-square&logo=postgresql&logoColor=white) ![BGE-M3](https://img.shields.io/badge/BGE--M3-1024d-FF6F00?style=flat-square&logoColor=white) ![Qwen](https://img.shields.io/badge/Qwen2.5-7B%20%2F%2032B-FF6F00?style=flat-square&logoColor=white) ![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

> A tree-guided multi-domain Retrieval-Augmented Generation system.
> Instead of searching a flat pool of documents, PRISM-RAG organizes knowledge
> into a hierarchical topic tree — like a library with sections, shelves, and
> books — so queries find the right information faster and with less noise.

> **Status:** Phases 1–2 of 7 complete. Full roadmap in [`ARCHITECTURE.md`](./ARCHITECTURE.md).

---

## What problem does this solve?

Standard RAG systems dump all documents into one big pile and search through
everything for every query. When your corpus spans multiple unrelated domains
(politics, finance, AI research, medicine), this causes two problems:

1. **Topic contamination** — a question about drug side effects pulls in
   financially-themed documents that happen to share keywords like "risk" and
   "trial."

2. **Needle in a haystack** — at millions of documents, flat search wastes
   compute comparing your query against content that was never relevant.

PRISM-RAG solves both by building a **hierarchical topic tree** over the
corpus. Think of it like this:

```
Flat RAG:       "Search all 2.5 million documents"

PRISM-RAG:      "This query is about medicine"
                    → narrow to medical subtree
                    → "Specifically about drug trials"
                        → narrow to clinical research cluster
                        → search only the ~500 relevant chunks
```

The tree is built offline using clustering and LLM-generated summaries
(following the RAPTOR method from Sarthi et al., ICLR 2024). At query time,
the system walks the tree top-down, eliminating irrelevant branches at each
level before doing a final vector search on the surviving leaf chunks.

---

## How it works (the big picture)

PRISM-RAG has two main stages: **offline** (build the tree once) and
**online** (answer queries using the tree).

### Offline: Building the knowledge tree

```
Raw documents (JSONL files)
    │
    ▼
[Phase 1] Split into chunks (≤512 tokens each)
    │
    ▼
[Phase 2] Encode each chunk into a 1024-d vector using BGE-M3
           Store as level-0 "leaf" nodes in the tree
    │
    ▼
[Phase 3] Group similar chunks using UMAP + HDBSCAN clustering
           Ask Qwen-32B to write a title + summary for each cluster
           Encode that summary → becomes a level-1 node
           Repeat upward → level 2, 3, 4...
           Result: a tree with ~4–5 levels
    │
    ▼
All nodes (leaves + summaries) indexed in one pgvector HNSW index
```

### Online: Answering a query

The diagram below walks through a concrete example — *"What was the Bitcoin
price in 2018?"* — showing each step from user input to cited answer:

<p align="center">
  <img src="docs/query_flow.png" alt="PRISM-RAG Query Flow" width="550">
</p>

**Step by step:**

1. **Query Input** — The user sends a natural language question.
2. **Domain Routing** (Python / LangGraph) — An LLM reads the query and
   identifies the target domain (`finance`). BGE-M3 encodes the query into a
   1024-d vector.
3. **Level 1 Search** (PostgreSQL + pgvector) — HNSW finds the most similar
   cluster summaries within the `finance` domain. Returns the best-matching
   cluster (e.g. `cluster_id=42`, "Cryptocurrency cluster").
4. **Level 0 Search** (PostgreSQL + pgvector) — Searches only the leaf chunks
   inside the matching cluster. Returns the top-k most relevant chunks
   (e.g. chunks about Bitcoin prices in 2018).
5. **Answer Generation** (Qwen2.5-7B) — The retrieved chunks are passed to
   the language model, which generates a cited answer grounded in the evidence.
6. **Answer Returned** — The user receives the final answer with source
   citations pointing back to the original chunks.

---

## What has been built so far

### Phase 1 — Data ingestion & chunking ✅

**What it does:** Downloads documents from HuggingFace datasets, stores them
in PostgreSQL, and splits each document into overlapping chunks of ≤512 tokens.

**Why chunking matters:** The embedding model (BGE-M3) has a context window.
Documents longer than 512 tokens must be split, or content beyond the limit
gets silently ignored. Overlapping chunks (64 tokens overlap) ensure sentences
near a split boundary appear in both chunks, so nothing falls through the cracks.

**Key design decisions:**
- Tokenizer used for chunking is the *same* tokenizer BGE-M3 uses internally.
  This guarantees the token count is exact — no surprises at encoding time.
- Idempotent ingestion: re-running the script skips already-inserted documents
  (uses `ON CONFLICT DO NOTHING` on a unique key). Safe to restart after crashes.
- Prototype cap of 5,000 documents per source keeps iteration fast while
  developing. Removed for the full run.

**Database state after Phase 1:**
- `documents` table: one row per original article/paper/speech
- `chunks` table: one row per ≤512-token piece, linked to its parent document

---

### Phase 2 — Dense embedding & leaf node construction ✅

**What it does:** Encodes every chunk into a 1024-dimensional dense vector
using BGE-M3, stores each as a level-0 (leaf) node in the `tree_nodes` table,
and builds an HNSW vector index over all nodes.

**How embedding works:** BGE-M3 is a transformer encoder (568M parameters).
It reads the chunk text, processes it through attention layers, and outputs a
single 1024-number vector that captures the semantic meaning of that text.
Similar texts produce vectors that point in similar directions in 1024-d space.

**Why we normalize vectors:** Every vector is scaled to unit length (L2 norm = 1).
This lets us use inner product (IP) for similarity search instead of cosine
similarity. On normalized vectors, IP and cosine give identical rankings, but
IP skips the per-comparison normalization step — roughly 3× fewer floating-point
operations per comparison. At millions of vectors and hundreds of comparisons
per query, this matters.

**Why HNSW index is built after loading, not before:** HNSW is a graph-based
index. Inserting into an existing graph requires finding neighbors and updating
edges for each new row — ~10× slower than loading all rows first and
constructing the graph in one optimized pass.

**HNSW parameters chosen:**
- `m = 16` — each node connects to 16 neighbors. Standard for ≥256-d vectors.
- `ef_construction = 200` — how many candidates the builder evaluates when
  picking neighbors. Higher = better graph quality, slower build. 200 is the
  recommended setting for 1024-d.
- Operator class: `vector_ip_ops` — inner product, paired with normalized
  vectors (see above).

**Key design decisions:**
- Server-side cursor streams chunks from Postgres in pages of 2,000 rows.
  Python memory stays flat regardless of corpus size.
- `withhold=True` on the cursor so it survives periodic `COMMIT` calls
  (otherwise Postgres destroys server-side cursors when the transaction ends).
- Resumable: chunks that already have a tree_node are skipped via `NOT EXISTS`.

**Database state after Phase 2:**
- `tree_nodes` table: one row per chunk, `level=0`, `is_leaf=true`, with a
  1024-d embedding vector
- HNSW index over all tree_nodes (currently only leaves; internal nodes from
  Phase 3 will be added to the same index)

---

## Corpus (4 domains)

Sourced via HuggingFace Datasets. Downloader script in `data/download.py`.

| Domain | Source | Description | Default cap |
|---|---|---|---|
| politics | `vblagoje/cc_news` | English news articles 2017–2019 | all (~708K) |
| politics | `Eugleo/us-congressional-speeches` | US Congressional speeches 1873–2024 | 700K |
| finance | `ashraq/financial-news-articles` | Reuters / CNBC / WSJ financial news | all (~306K) |
| ai_tech | `CShorten/ML-ArXiv-Papers` | ML & AI ArXiv titles + abstracts | all (~118K) |
| medical | `ccdv/pubmed-summarization` | PubMed full-text papers | all (~120K) |
| medical | `ccdv/arxiv-summarization` | ArXiv full-text papers | all (~203K) |

A prototype subset of 5,000 documents per source is used through Phase 4 to
keep iteration cycles short.

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
                │  Generation Agent ── Qwen2.5-7B                │
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
                │      ▼  UMAP → HDBSCAN → Qwen-32B    (Phase 3)│
                │      │  summarize → embed → store              │
                │  tree_nodes  (levels 1..k = cluster summaries) │
                │      │                                         │
                │      ▼  one HNSW index over ALL levels         │
                │  pgvector tree (1 table, 1 index, level filter)│
                │                                                │
                └────────────────────────────────────────────────┘
```

---

## Stack

| Component | Technology |
|---|---|
| Database | PostgreSQL 16 + pgvector (Docker) |
| Vector index | HNSW with `vector_ip_ops` on normalized 1024-d vectors |
| Embedding model | `BAAI/bge-m3` (1024-d dense, 512-token context) |
| Clustering | UMAP (dimensionality reduction) + HDBSCAN (cluster discovery) |
| Cluster summarizer | `Qwen/Qwen2.5-32B-Instruct` via vLLM |
| Generator | `Qwen/Qwen2.5-7B-Instruct` (v0.1) → QLoRA + AWQ in v0.2 |
| Orchestration | LangGraph StateGraph (Phase 5) |
| API | FastAPI + uvicorn (Phase 5) |
| Container runtime | Docker + docker-compose |
| Experiment tracking | Weights & Biases (Phase 6) |

---

## Build status

| Phase | Component | Status |
|---|---|---|
| 1 | Data ingestion + token-aware chunking | ✅ Done |
| 2 | BGE-M3 dense embedding + HNSW index | ✅ Done |
| 3 | UMAP + HDBSCAN clustering + LLM summaries (tree build) | ⏳ Next |
| 4 | Tree-guided retrieval (CLI → FastAPI) | ⏳ Planned |
| 5 | Qwen-7B generation + LangGraph gateway + Docker stack | ⏳ Planned |
| 6 | QLoRA fine-tune + AWQ quantization per domain *(optional)* | ⏳ Planned |
| 7 | Synthetic eval set + retrieval & generation metrics | ⏳ Planned |

---

## Project structure

```
PRISM-RAG/
├── README.md                    ← this file
├── ARCHITECTURE.md              ← detailed roadmap, schema, phase verifications
├── init.sql                     ← PostgreSQL schema (documents, chunks, tree_nodes)
├── docker-compose.yml           ← Postgres + pgvector container
├── requirements.txt
├── .env.example
│
├── docs/                        ← diagrams and documentation assets
│   └── query_flow.png           ← end-to-end query flow diagram
│
├── data/                        ← downloaded JSONL (politics/finance/ai_tech/medical)
│   └── download.py              ← HuggingFace dataset downloader
│
├── agents/
│   ├── ingestion/               ← Phases 1–2: batch ingest + embed
│   │   ├── config.py            ← paths, model names, hyperparameters
│   │   ├── db.py                ← Postgres + pgvector connection helper
│   │   ├── chunker.py           ← token-aware overlapping text splitter
│   │   ├── loader.py            ← JSONL → documents + chunks (idempotent)
│   │   ├── encoder.py           ← BGE-M3 dense encoder wrapper
│   │   └── embed_leaves.py      ← chunks → tree_nodes level 0
│   ├── tree_builder/            ← Phase 3: cluster + summarize
│   ├── retrieval/               ← Phase 4: tree-guided search
│   └── generation/              ← Phase 5: Qwen inference
│
├── gateway/                     ← Phase 5: FastAPI + LangGraph orchestrator
├── training/                    ← Phase 6: QLoRA → merge → AWQ
├── evaluation/                  ← Phase 7: synthetic eval + metrics
├── checkpoints/                 ← Phase 6: LoRA adapters, AWQ models
│
└── scripts/                     ← numbered entry points
    ├── 01_init_db.py            ← apply schema to Postgres
    ├── 02_ingest.py             ← ingest JSONL → documents + chunks
    ├── 03_embed_chunks.py       ← embed chunks + build HNSW index
    ├── 04_build_tree.py         ← (Phase 3)
    ├── 05_query_cli.py          ← (Phase 4)
    └── 06_benchmark.py          ← (Phase 7)
```

---

## Quick start

**Prerequisites:** Python 3.10+, Docker, ~10 GB free disk, GPU recommended
for embedding (CPU works but slower).

```bash
git clone https://github.com/Arupreza/PRISM-RAG
cd PRISM-RAG

# Install dependencies
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Configure
cp .env.example .env
# edit .env: set PG_DSN, HF_TOKEN, PRISM_DATA_DIR

# Start Postgres + pgvector
docker compose up -d postgres

# Download the corpus (all 4 domains)
python data/download.py

# Phase 1: Initialize DB + ingest documents
python scripts/01_init_db.py
python scripts/02_ingest.py

# Phase 2: Embed chunks + build HNSW index
python scripts/03_embed_chunks.py
```

**Verify everything worked:**

```sql
-- Connect to the database
psql "postgresql://prism:prism@localhost:5433/prism_rag"

-- Phase 1: document and chunk counts
SELECT domain, source, COUNT(*) FROM documents GROUP BY 1,2 ORDER BY 1,2;
SELECT MIN(n_tokens), AVG(n_tokens)::int, MAX(n_tokens) FROM chunks;

-- Phase 2: every chunk has a leaf node
SELECT
  (SELECT COUNT(*) FROM chunks) AS chunks,
  (SELECT COUNT(*) FROM tree_nodes WHERE level=0) AS leaves;

-- Phase 2: HNSW index exists
SELECT indexname FROM pg_indexes
WHERE tablename = 'tree_nodes' AND indexname LIKE '%hnsw%';
```

---

## Project history

This repository originally described a SPLADE-based sparse multi-agent RAG
over BEIR benchmark datasets. That design was dropped in favor of tree-guided
dense retrieval over a curated multi-domain corpus. The original design is
visible in early git history. Reasoning for the pivot is documented in
[`ARCHITECTURE.md`](./ARCHITECTURE.md).

---

## What this project is and isn't

**Is:**
- A research project exploring whether RAPTOR-style hierarchical topic trees
  improve retrieval over flat dense search on a heterogeneous multi-domain corpus.
- A fully open, single-database implementation (everything in PostgreSQL +
  pgvector — no external vector service).

**Isn't:**
- Production-ready.
- Yet benchmarked. Performance comparisons against flat baselines will be
  reported in Phase 7 on a synthetic evaluation set.

---

## License

MIT.

---

## Author

**Md Rezanur Islam (Reza)**
LLM Engineer & Agentic AI Developer
PhD Candidate, Software Convergence — Soonchunhyang University (BK21)