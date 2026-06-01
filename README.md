# PRISM-RAG

### Progressive Retrieval with Indexed Summary Memory

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white) ![FastAPI](https://img.shields.io/badge/FastAPI-Latest-009688?style=flat-square&logo=fastapi&logoColor=white) ![Docker](https://img.shields.io/badge/Docker-Latest-2496ED?style=flat-square&logo=docker&logoColor=white) ![LangGraph](https://img.shields.io/badge/LangGraph-Latest-1C3C3C?style=flat-square&logo=langchain&logoColor=white) ![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=white) ![pgvector](https://img.shields.io/badge/pgvector-HNSW-336791?style=flat-square&logo=postgresql&logoColor=white) ![BGE-M3](https://img.shields.io/badge/BGE--M3-1024d-FF6F00?style=flat-square&logoColor=white) ![Qwen](https://img.shields.io/badge/Qwen2.5-7B%20%2F%2032B-FF6F00?style=flat-square&logoColor=white) ![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

> A tree-guided multi-domain Retrieval-Augmented Generation system.
> Instead of searching a flat pool of documents, PRISM-RAG organizes knowledge
> into a hierarchical topic tree — like a library with sections, shelves, and
> books — so queries find the right information faster and with less noise.

> **Status:** Phases 1–4 of 7 complete. Full roadmap in [`ARCHITECTURE.md`](./ARCHITECTURE.md).

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
           Ask Qwen2.5 to write a title + summary for each cluster
           Encode that summary → becomes a level-1 node
           Repeat upward → level 2, 3, 4...
           Result: a per-source tree with up to 4 levels above the leaves
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

### Phase 1 — Ingestion & chunking ✅
Pulls documents from HuggingFace into PostgreSQL, then cuts each one into pieces
of at most 512 tokens. Why cut them? The embedding model can only "read" 512
tokens at a time — anything longer gets ignored. Each piece overlaps the next by
64 tokens so a sentence sitting on a cut line still appears whole in one of them.
Re-running the script is safe: it skips documents already loaded.
**Result:** a `documents` table (full articles) and a `chunks` table (the pieces).

### Phase 2 — Embedding the chunks ✅
Turns every chunk into a list of 1024 numbers (a "vector") using BGE-M3. Chunks
with similar meaning get similar vectors, which is what lets us search by meaning
instead of keywords. Each vector is scaled to length 1, a small trick that makes
similarity search faster. After all vectors are stored, we build one HNSW index —
a structure that finds nearest vectors quickly. We build it *after* loading
everything because building it once at the end is far faster than updating it row
by row.
**Result:** every chunk now lives in `tree_nodes` as a level-0 "leaf" with its
vector, plus a search index over them.

### Phase 3 — Building the topic tree ✅
This is where the "tree" gets built, bottom-up, separately for each source. The
idea (from the RAPTOR paper): group related chunks, summarize each group, then
treat those summaries as a higher layer and repeat — like turning thousands of
notes into chapter summaries, then a book summary.

- **Group similar chunks.** 1024 numbers is too many dimensions for grouping to
  work well, so UMAP first squeezes them down to 10. Then HDBSCAN finds the
  natural groups by density — we don't tell it how many groups to expect, it
  figures that out. A few chunks land in no group ("noise"); we attach each to
  its closest group so no chunk is ever left behind and lost from the tree.
- **Summarize each group.** A local Qwen2.5-7B model writes a short title and
  summary for every group. It runs inside the same program (no separate server to
  start), and big groups are summarized in batches so they never overflow the
  model's reading limit.
- **Repeat upward.** Those summaries get embedded and grouped again into a higher
  level, up to 4 levels, stopping once everything folds into a single group.
- **Safe to re-run.** The database is wired so that deleting a summary node would
  also delete its child chunks. To avoid wiping Phase 2 on a rebuild, we
  disconnect the links first, then delete only the summary nodes.

**Result:** `tree_nodes` now holds the original chunks (level 0) *and* the
summary nodes above them (level 1+), all connected parent-to-child and searchable
through the same index.

### Phase 4 — Tree-guided retrieval ✅
The tree built in Phase 3 only pays off if queries actually walk it correctly at
run time. This phase ships two ways to do that walk, plus a CLI for debugging
and a FastAPI service for everything downstream to call.

- **Top-down beam traversal.** Encode the query, find the best-matching tree
  roots (one root per source), then keep the top-`beam` children at each step
  down to the leaves. Beam width is set to 6 instead of 1 because greedy descent
  compounds early clustering mistakes — keeping a handful of candidates per level
  lets the search recover from a wrong turn near the top. The final answer set
  is the top-`k` leaves under the last internal frontier, fetched in one
  recursive descent so a tight final-step beam never caps leaf recall.
- **Collapsed + ancestor boost.** Search the whole tree in one flat HNSW pass,
  then for each leaf candidate add a bonus proportional to its strongest
  ancestor-cluster similarity: `combined = leaf_sim + α · max(ancestor_sim)`.
  A leaf that sits under a strongly on-topic cluster wins over a leaf that
  lexically matches but belongs to the wrong topic. The cluster summary acts as
  a learned topic prior.
- **One round trip per descent step.** Children at the next level are fetched
  with `WHERE parent_id = ANY(frontier)`, hitting the b-tree parent index — no
  extra ANN call per branch. `hnsw.ef_search` is raised per transaction (not
  per session), so the bump stays scoped and never leaks across pooled
  connections.
- **Same code, two surfaces.** `scripts/05_query_cli.py` calls the search code
  directly so retrieval can be debugged without HTTP. `agents/retrieval/main.py`
  wraps the same code in FastAPI (`POST /retrieve`, plus `/healthz` and
  `/readyz`) and ships in a uv-based Docker image. The encoder is loaded once
  in a lifespan handler so the first request doesn't pay startup tax.

Which strategy wins on real data is the open question for Phase 7's evaluation.
We deliberately ship both — picking one before measuring is the kind of decision
that haunts you later.

**Result:** retrieval service at `http://localhost:8001`. `POST /retrieve`
returns the top-`k` leaf chunks plus the tree path walked to find them, so
downstream generation can both cite the chunks and explain why they were chosen.

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
   User Query ─►│  Gateway (FastAPI + LangGraph)        ⏳ Ph 5  │
                │      │                                         │
                │      ▼                                         │
                │  Retrieval Agent  ── tree-guided search ✅ Ph 4│
                │      │   (top-down beam OR collapsed re-rank)  │
                │      │    domain → source → clusters → leaves  │
                │      ▼                                         │
                │  Generation Agent ── Qwen2.5-7B       ⏳ Ph 5  │
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
                │      ▼  UMAP → HDBSCAN → Qwen2.5-7B   (Phase 3)│
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
| Clustering | UMAP (dimensionality reduction) + HDBSCAN (cluster discovery) + noise reassignment |
| Cluster summarizer | Local `Qwen2.5-7B-Instruct` checkpoint via `transformers` (in-process, greedy) |
| Retrieval service | FastAPI + uvicorn, uv-based Docker image (Phase 4) |
| Generator | `Qwen/Qwen2.5-7B-Instruct` (v0.1) → QLoRA + AWQ in v0.2 |
| Orchestration | LangGraph StateGraph (Phase 5) |
| Container runtime | Docker + docker-compose |
| Experiment tracking | Weights & Biases (Phase 6) |

---

## Build status

| Phase | Component | Status |
|---|---|---|
| 1 | Data ingestion + token-aware chunking | ✅ Done |
| 2 | BGE-M3 dense embedding + HNSW index | ✅ Done |
| 3 | UMAP + HDBSCAN clustering + LLM summaries (tree build) | ✅ Done |
| 4 | Tree-guided retrieval (CLI + FastAPI + Docker) | ✅ Done |
| 5 | Qwen-7B generation + LangGraph gateway + Docker stack | ⏳ Next |
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
├── checkpoints/                 ← model weights
│   └── source_model/qwen_2_5/   ← local Qwen2.5-7B-Instruct (Phase 3 summarizer)
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
│   │   ├── cluster.py           ← UMAP reduce + HDBSCAN + noise reassignment
│   │   ├── summarizer.py        ← in-process Qwen2.5 summarizer (map-reduce)
│   │   └── build.py             ← recursive per-source tree construction
│   ├── retrieval/               ← Phase 4: tree-guided search
│   │   ├── tree_search.py       ← top-down beam + collapsed re-rank (DB + numpy)
│   │   ├── main.py              ← FastAPI: POST /retrieve, GET /healthz, /readyz
│   │   └── Dockerfile           ← uv-based slim image (CPU default, CUDA via build-arg)
│   └── generation/              ← Phase 5: Qwen inference
│
├── gateway/                     ← Phase 5: FastAPI + LangGraph orchestrator
├── training/                    ← Phase 6: QLoRA → merge → AWQ
├── evaluation/                  ← Phase 7: synthetic eval + metrics
│
└── scripts/                     ← numbered entry points
    ├── 01_init_db.py            ← apply schema to Postgres
    ├── 02_ingest.py             ← ingest JSONL → documents + chunks
    ├── 03_embed_chunks.py       ← embed chunks + build HNSW index
    ├── 04_build_tree.py         ← cluster + summarize → internal nodes (Phase 3)
    ├── 05_query_cli.py          ← tree-guided retrieval CLI (Phase 4)
    └── 06_benchmark.py          ← (Phase 7)
```

---

## Quick start

**Prerequisites:** Python 3.10+, Docker, ~10 GB free disk, GPU recommended
for embedding and summarization (CPU works but slower). A local
Qwen2.5-7B-Instruct checkpoint under `checkpoints/source_model/qwen_2_5/` for
Phase 3 (point `SUMMARIZER_MODEL` in `agents/ingestion/config.py` at it).

```bash
git clone https://github.com/Arupreza/PRISM-RAG
cd PRISM-RAG

# Install dependencies
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt        # note: uv venvs have no `pip`; use `uv pip`

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

# Phase 3: Cluster + summarize → build the topic tree
python scripts/04_build_tree.py
#   options:
#     --domain medical     build a subset of domains
#     --no-rebuild         keep existing internal nodes
#   tip: set EMBED_DEVICE=cpu if the summarizer and BGE-M3 compete for VRAM

# Phase 4a: Query the tree from the CLI (no service)
python scripts/05_query_cli.py "What did Congress say about voter ID laws?"
python scripts/05_query_cli.py "mRNA vaccine R&D financial impact" \
    --domain finance --mode collapsed --k 5

# Phase 4b: Run the retrieval service in Docker
docker build -f agents/retrieval/Dockerfile \
  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 \
  -t prism-retrieval:gpu .

docker run --rm --gpus all --network host \
  -v ~/.cache/huggingface:/cache/hf \
  --env-file .env \
  prism-retrieval:gpu

# Then in another terminal — readiness check + a query
curl -s http://localhost:8001/readyz
curl -s -X POST http://localhost:8001/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"query":"voter ID laws","k":5}' | python -m json.tool

# Browser-based playground (auto-generated Swagger UI):
#   http://localhost:8001/docs
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

-- Phase 3: nodes per level (level 0 = leaves, level ≥1 = cluster summaries)
SELECT level, COUNT(*) FROM tree_nodes GROUP BY level ORDER BY level;

-- Phase 3: no leaf was dropped (must return 0)
SELECT COUNT(*) FROM tree_nodes WHERE level=0 AND parent_id IS NULL;

-- Phase 3: tree shape per source
SELECT domain, source, MAX(level) AS depth FROM tree_nodes
GROUP BY domain, source ORDER BY domain, source;

-- Phase 3: inspect the largest clusters
SELECT title, n_descendants FROM tree_nodes
WHERE level >= 1 ORDER BY n_descendants DESC LIMIT 10;
```

**Phase 4 smoke checks** (against the running service):

```bash
# liveness
curl -s http://localhost:8001/healthz
# expect: {"ok":true}

# readiness (encoder loaded + DB reachable + leaves present)
curl -s http://localhost:8001/readyz
# expect: {"ok":true,"n_leaves":<your leaf count>}

# top-down beam (default)
curl -s -X POST http://localhost:8001/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"query":"voter ID laws","mode":"top_down","k":5}'

# collapsed + ancestor boost
curl -s -X POST http://localhost:8001/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"query":"voter ID laws","mode":"collapsed","alpha":0.3,"k":5}'
```

The `scripts/04_build_tree.py` run prints a `n_descendants` vs leaf-count
reconciliation per source. `scripts/05_query_cli.py` prints the full
traversal path and the top-`k` leaves with similarity scores — use it to
sanity-check that retrieval picks the right domain and source before wiring
up Phase 5.

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