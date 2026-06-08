# PRISM-RAG

### Progressive Retrieval with Indexed Summary Memory

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white) ![FastAPI](https://img.shields.io/badge/FastAPI-Latest-009688?style=flat-square&logo=fastapi&logoColor=white) ![Docker](https://img.shields.io/badge/Docker-Latest-2496ED?style=flat-square&logo=docker&logoColor=white) ![LangGraph](https://img.shields.io/badge/LangGraph-Latest-1C3C3C?style=flat-square&logo=langchain&logoColor=white) ![pgvector](https://img.shields.io/badge/pgvector-HNSW-336791?style=flat-square&logo=postgresql&logoColor=white) ![BM25](https://img.shields.io/badge/BM25-tsvector%20GIN-336791?style=flat-square&logo=postgresql&logoColor=white) ![BGE-M3](https://img.shields.io/badge/BGE--M3-1024d-FF6F00?style=flat-square&logoColor=white) ![Qwen](https://img.shields.io/badge/Qwen2.5-7B-FF6F00?style=flat-square&logoColor=white) ![AWQ](https://img.shields.io/badge/AWQ-W4A16-4B0082?style=flat-square&logoColor=white) ![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

> A tree-guided multi-domain Retrieval-Augmented Generation system with
> hybrid dense+BM25 retrieval, structure-aware chunking, and domain-specialist
> quantized worker models. Documents are organized into a hierarchical topic
> tree; queries walk the tree, fuse dense and lexical signals at the leaves,
> then route the evidence to a domain-expert LLM worker fine-tuned for that
> subject area.

> **Status:** Phases 1–5.5 of 7 complete. Phase 1.5 (hybrid retrieval +
> structure-aware chunking) and Phase 6 (gateway orchestration) in progress.
> Phase 7 (evaluation) next. Full roadmap in [`ARCHITECTURE.md`](./ARCHITECTURE.md).

---

## What problem does this solve?

Standard RAG fails in three compounding ways on heterogeneous corpora:

1. **Topic contamination at retrieval.** A query about Korean visa renewal
   pulls back trading documents that share surface vocabulary ("application,"
   "filing," "deadline"). Flat dense search has no representation of subject
   domain — it only sees vector similarity.

2. **Dense-only retrieval misses exact terms.** Dense embeddings struggle
   with proper nouns, identifiers, codes, and rare tokens — exactly the
   things users include when they know what they're looking for ("F-2-7-7
   visa," ticker symbols, paper titles). Pure cosine similarity buries the
   correct chunk behind topically-related but lexically-wrong ones.

3. **Generic generation.** Even with correct retrieval, a single
   general-purpose LLM hedges on domain-specific questions. A finance
   question needs a model trained on finance reasoning. A legal question
   needs a model trained to refuse when the answer isn't in context — a
   behavior generic instruction-tuned models do poorly.

**PRISM-RAG fixes all three layers:**

- **Routing:** documents are organized into a RAPTOR-style hierarchical
  topic tree (Sarthi et al., ICLR 2024). Queries walk down a tree of
  cluster summaries before touching leaf chunks — domain filtering is
  structural, not learned.
- **Retrieval:** at the leaves, dense ANN and BM25 lexical search are
  fused via Reciprocal Rank Fusion (RRF) so semantic similarity and
  exact-term matches both vote on the final ranking.
- **Generation:** three domain-specialist worker models (AI / Trading /
  Korean Immigration Law), each fine-tuned from a shared Qwen2.5 base and
  quantized to AWQ W4A16 for efficient serving.

```
Standard RAG:    Search all documents → one generic LLM answers

PRISM-RAG:       "This query is about Korean immigration law"
                     → narrow to the law subtree
                     → hybrid (dense + BM25) search on the relevant cluster
                     → AWQ-quantized law worker generates the answer
                       (trained to refuse when answer not in context)
```

---

## How it works (the big picture)

PRISM-RAG has three stages: **offline tree build**, **offline worker training**,
and **online query serving**.

### Offline — Stage A: Building the knowledge tree

```
Raw documents (JSONL + ingested PDFs)
    │
    ▼
[Phase 1 + 1.5] Structure-aware chunking
                  - Split on paragraph boundaries (atomic units)
                  - Oversized paragraphs → bisect on sentence boundaries
                  - Short adjacent paragraphs → merge up to target size
                  - Overlap (64 tokens) only at artificial split points
                  - Result: chunks in [100, 512] tokens preserving meaning
    │
    ▼
[Phase 2] Encode each chunk into a 1024-d vector using BGE-M3
           Store as level-0 "leaf" nodes in the tree
           Build PostgreSQL tsvector + GIN index on leaf summary text
           (powers the BM25 side of hybrid retrieval)
    │
    ▼
[Phase 3] Group similar chunks using UMAP + HDBSCAN clustering
           Ask Qwen2.5-7B to write a title + summary for each cluster
           Encode that summary → becomes a level-1 node
           Repeat upward → level 2, 3, 4...
    │
    ▼
All nodes indexed in one pgvector HNSW index
Leaves additionally indexed by tsvector + GIN for BM25
```

### Offline — Stage B: Training the domain workers

```
Shared Qwen2.5 base
    │
    ├──► Trader worker:  TRL SFT on Sujet-Finance-177k + finance-alpaca
    │                    QLoRA r=32 → LoRA merge → AWQ W4A16
    │
    ├──► Coder worker:   TRL GRPO with 4 executable rewards:
    │                      (1) output format
    │                      (2) Python syntax compile
    │                      (3) sandboxed unit-test correctness
    │                      (4) length band
    │                    LoRA merge → AWQ W4A16
    │
    └──► Law worker:     law_llm base reused (no SFT)
                         → AWQ W4A16 with 30% refusal-injection
                            (preserves "not-in-context" behavior post-quant)
```

### Online: Answering a query

```
                    User query
                        │
                        ▼
            ┌───────────────────────┐
            │  LangGraph Gateway    │
            │  (StateGraph)         │
            └───────────┬───────────┘
                        │ domain routing
                        │ (cosine-argmax of query
                        │  embedding vs. tree roots)
                        ▼
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
   AI subtree     Trading subtree   Law subtree
        │               │               │
        ▼               ▼               ▼
   Tree walk      Tree walk         Tree walk         ← top-down beam OR
   (dense ANN)    (dense ANN)       (dense ANN)         collapsed re-rank
        │               │               │
        ▼               ▼               ▼
   Candidate      Candidate         Candidate
   leaf set       leaf set          leaf set
        │               │               │
        ▼               ▼               ▼
   Hybrid fuse    Hybrid fuse       Hybrid fuse       ← dense + BM25 via
   (RRF k=60)     (RRF k=60)        (RRF k=60)          Reciprocal Rank Fusion
        │               │               │
        ▼               ▼               ▼
   AWQ Coder      AWQ Trader        AWQ Law
   worker         worker            worker
        │               │               │
        └───────────────┼───────────────┘
                        ▼
                 Cited answer → User
```

---

## What has been built so far

### Phase 1 — Ingestion ✅ + Phase 1.5 — Structure-aware chunking ⏳

The chunker is being upgraded from fixed-token splitting to **structure-aware
chunking**:

- **Paragraphs are atomic.** Each paragraph (double-newline boundary, or
  numbered clause for legal PDFs) is one chunk by default.
- **Oversized paragraphs split on sentence boundaries.** A paragraph
  exceeding 512 tokens is recursively bisected at the nearest sentence
  boundary — never mid-sentence — until each piece fits. 64-token overlap
  is applied only at these artificial split points, NOT across natural
  paragraph boundaries (overlap between semantically-separate paragraphs
  is noise, not signal).
- **Short paragraphs merge.** Adjacent paragraphs under 100 tokens are
  combined up to the target size band [100, 512]. Trading news articles
  in particular have many 1–3-sentence paragraphs that benefit from
  merging.
- **Per-source rules.** Trading news → aggressive merging; ArXiv papers →
  preserve paragraphs as-is; Korean Immigration Law PDFs → chunk on
  numbered clauses when detected.

Re-running the script is safe: content-hash dedup skips documents already
loaded.

**Result:** a `documents` table (full articles) and a `chunks` table whose
size distribution sits cleanly in [100, 512] tokens, with chunk boundaries
that respect actual document structure.

### Phase 2 — Embedding ✅ + BM25 index ⏳

Turns every chunk into a 1024-d vector using BGE-M3 (L2-normalized). HNSW
index is built once after all vectors are loaded.

**Phase 1.5 adds a lexical index** on the same `tree_nodes` table:

```sql
ALTER TABLE tree_nodes
  ADD COLUMN tsv tsvector
  GENERATED ALWAYS AS (to_tsvector('english', summary)) STORED;

CREATE INDEX tree_nodes_tsv_idx
  ON tree_nodes USING GIN(tsv)
  WHERE level = 0;
```

PostgreSQL's `ts_rank_cd` provides BM25-like scoring out of the box;
`pg_search` (formerly ParadeDB) is a drop-in for true BM25 if needed
later. The index is **leaf-only** — cluster summaries are LLM-generated
and don't necessarily contain the user's exact query terms, so BM25 over
them adds noise rather than signal.

**Result:** every chunk lives in `tree_nodes` as a level-0 leaf with both
a 1024-d dense embedding (HNSW) and a tsvector (GIN). Hybrid retrieval
works on the same row, single source of truth.

### Phase 3 — Building the topic tree ✅

Built bottom-up per `(domain, source)`. UMAP reduces 1024-d → 10-d for
density-based clustering; HDBSCAN finds natural clusters; noise points are
reassigned to the nearest cluster so zero leaves are lost. Qwen2.5-7B
writes a title + summary per cluster (in-process via `transformers`,
greedy, JSON-mode); summaries are re-embedded as the next level. Repeats
up to 4 levels.

**Result:** `tree_nodes` holds leaves (level 0) and cluster summary nodes
(level 1+), connected parent-to-child and searchable via the same HNSW
index. Rebuild scripts disconnect parent links before deleting summary
nodes so the leaf layer survives.

### Phase 4 — Tree-guided retrieval ✅ → hybrid extension ⏳

The retrieval service exposes two tree-walk strategies and (with Phase 1.5)
three fusion modes on top of them.

**Tree-walk strategies:**

- **Top-down beam traversal** (beam=6): ANN at level k → keep top-`beam`
  children → fetch children via `WHERE parent_id = ANY(frontier)` → repeat
  to leaves. Beam=6 (not 1) recovers from early clustering mistakes;
  final step uses one recursive descent so beam doesn't cap leaf recall.
- **Collapsed + ancestor boost**: flat HNSW pass over ALL nodes, then
  re-rank with `combined = leaf_sim + α · max(ancestor_sim)`. Cluster
  summary acts as a learned topic prior.

**Fusion modes** (Phase 1.5):

- **`fusion: "dense"`** — current default, pure HNSW cosine.
- **`fusion: "bm25"`** — pure `ts_rank_cd` over the leaf tsvector.
- **`fusion: "hybrid"`** — tree walk returns top-`k·4` candidate leaves;
  dense and BM25 are computed independently within that candidate set;
  results fused via **Reciprocal Rank Fusion**:

  ```
  score_rrf(doc) = Σ over rankers  1 / (k_rrf + rank_ranker(doc))
  ```

  RRF (k_rrf=60) is rank-based, scale-free, and avoids the score-
  normalization problem of weighted-sum fusion. Weighted-sum
  (`α · dense + (1-α) · bm25`) needs per-query normalization to make
  cosine and tsvector scores comparable — fragile in practice.

**Performance optimization:** `hnsw.ef_search` is raised **per transaction**
(`SET LOCAL`), not per session, so the bump stays scoped and never leaks
across pooled connections.

**API:**

```http
POST /retrieve
{
    "query": "F-2 visa renewal requirements",
    "mode": "top_down",         // "top_down" | "collapsed"
    "fusion": "hybrid",         // "dense" | "bm25" | "hybrid"
    "rrf_k": 60,                // optional, default 60
    "alpha": 0.3,               // for collapsed mode only
    "k": 5
}
```

`fusion: "dense"` is the default to preserve existing behavior; switch to
`"hybrid"` once Phase 1.5 ships.

**Result:** retrieval service at `http://localhost:8001`. `POST /retrieve`
returns top-`k` leaf chunks plus the tree path walked plus the fusion
method used.

### Phase 5 — Domain-specialist quantized workers ✅

Three domain experts on a shared Qwen2.5 base, all shipped as AWQ W4A16.

- **Trader worker** — TRL `SFTTrainer` on Sujet-Finance-Instruct-177k +
  finance-alpaca. QLoRA r=32, nf4 double-quant, bf16, cosine LR 2e-4,
  effective batch 16, packed chat template.
- **Coder worker** — TRL `GRPOTrainer` on verifiable-coding-problems +
  LeetCodeDataset with four executable reward heads:
  (1) **format reward** — output matches
  `<reasoning>...</reasoning><code>...</code>`;
  (2) **syntax reward** — code passes Python `compile()`;
  (3) **correctness reward** — sandboxed subprocess execution against
  unit tests, 8-second timeout;
  (4) **length reward** — 100–800 tokens.
- **Law worker** — `law_llm` base reused directly; value-add in AWQ
  calibration.

**AWQ pipeline:**
1. CPU-side LoRA → FP16 merge with tokenizer-aware embedding resize.
2. GPU AWQ calibration on 128 task-matched samples:
   verifiable-coding-problems for coder, CUAD-QA + LegalQAEval for law.
3. **30% of law calibration samples are "answer-not-in-context" cases**
   so 4-bit quantization doesn't degrade refusal behavior — a real
   failure mode when quantizing models that need to say "I don't know."

### Phase 5.5 — Self-updating ingestion ✅

Drop a PDF into a watched folder; the system:

1. Computes SHA-256 hash; skip if already ingested.
2. PDF → text → structure-aware chunks (Phase 1.5 chunker).
3. BGE-M3 embeds chunks; cosine-argmax vs. cached tree-root embeddings
   assigns domain.
4. Inserts chunks as level-0 leaves; tsvector auto-generated by the
   schema; HNSW updated incrementally.
5. Each new leaf is reassigned to the nearest level-1 cluster within its
   classified domain.
6. If any cluster grows past `CLUSTER_RESUMMARIZE_THRESHOLD`, its
   summary is regenerated.

Live-corpus expansion takes seconds per document — no full rebuild.

---

## Corpus (3 domains)

| Domain | Source | Description |
|---|---|---|
| AI | `CShorten/ML-ArXiv-Papers` | ML & AI ArXiv titles + abstracts |
| Trading | `ashraq/financial-news-articles` | Reuters / CNBC / WSJ financial news |
| Trading | User-ingested PDFs | Trading strategy docs, market analyses |
| Korean Immigration Law | User-ingested PDFs | Visa, residency, naturalization regulations |

The corpus intentionally mixes curated HuggingFace datasets with
user-contributed PDFs to validate that the tree structure, routing logic,
and hybrid retrieval hold up under heterogeneous, evolving inputs.

---

## Architecture

```
                ┌────────────────────────────────────────────────┐
                │            ONLINE  (Phases 4–6)                │
                ├────────────────────────────────────────────────┤
                │                                                │
   User Query ─►│  Gateway (FastAPI + LangGraph)        ⏳ Ph 6  │
                │      │     ── domain routing                   │
                │      ▼                                         │
                │  Retrieval Agent  ── tree walk + hybrid ✅ Ph 4│
                │      │   tree walk: top-down beam OR collapsed │
                │      │   leaf fusion: dense + BM25 via RRF ⏳1.5│
                │      ▼                                         │
                │  Generation Agent ── AWQ worker       ✅ Ph 5  │
                │      │   (Trader / Coder / Law)                │
                │      │    cited answer                         │
                │      ▼                                         │
                │  Final Answer ───────────────────────► User    │
                └──────────────────────┬─────────────────────────┘
                                       │ reads
                                       ▼
                ┌────────────────────────────────────────────────┐
                │            OFFLINE  (Phases 1–3, 5)            │
                ├────────────────────────────────────────────────┤
                │                                                │
                │  data/{domain}/*.jsonl + user PDFs             │
                │      │                                         │
                │      ▼  structure-aware chunk   (Ph 1 + 1.5)   │
                │  documents + chunks tables                     │
                │      │                                         │
                │      ▼  BGE-M3 dense + tsvector   (Ph 2 + 1.5) │
                │  tree_nodes (level 0 = leaves, HNSW + GIN)     │
                │      │                                         │
                │      ▼  UMAP → HDBSCAN → Qwen2.5-7B   (Ph 3)   │
                │      │  summarize → embed → store              │
                │  tree_nodes (levels 1..k = cluster summaries)  │
                │      │                                         │
                │      ▼  one HNSW index over ALL levels         │
                │      │  + leaf-only tsvector GIN for BM25      │
                │  pgvector tree (single DB, dense + lexical     │
                │  in the same table)                            │
                │                                                │
                │  ┌────────────────────────────────────────┐    │
                │  │  Qwen2.5 base ── per-domain workers    │    │
                │  │     ├─ Trader: SFT (QLoRA r=32)        │    │
                │  │     ├─ Coder: GRPO (4 exec rewards)    │    │
                │  │     └─ Law: base reused                │    │
                │  │  All → AWQ W4A16 calibration  (Phase 5)│    │
                │  └────────────────────────────────────────┘    │
                │                                                │
                └────────────────────────────────────────────────┘
```

---

## Stack

| Component | Technology |
|---|---|
| Database | PostgreSQL 16 + pgvector (Docker) |
| Dense index | HNSW with `vector_ip_ops` on normalized 1024-d vectors |
| Lexical index | `tsvector` + GIN, scored via `ts_rank_cd` (BM25-style) |
| Hybrid fusion | Reciprocal Rank Fusion (RRF, k=60) |
| Embedding model | `BAAI/bge-m3` (1024-d dense, 512-token context) |
| Chunker | Structure-aware: paragraph-atomic, sentence-bisect on overflow, merge-on-short, per-source rules |
| Clustering | UMAP (dim-reduction) + HDBSCAN + noise reassignment |
| Cluster summarizer | Local `Qwen2.5-7B-Instruct` via `transformers` (in-process, greedy) |
| Retrieval service | FastAPI + uvicorn, uv-based Docker image |
| Worker base model | `Qwen/Qwen2.5-7B-Instruct` (+ `law_llm` for law worker) |
| Worker fine-tuning | TRL `SFTTrainer` (trader), TRL `GRPOTrainer` (coder) |
| PEFT | QLoRA r=32, nf4 double-quant via BitsAndBytes |
| Worker quantization | AWQ W4A16 via `autoawq` |
| Orchestration | LangGraph StateGraph (Phase 6) |
| Container runtime | Docker + docker-compose |
| Experiment tracking | Weights & Biases |

---

## Build status

| Phase | Component | Status |
|---|---|---|
| 1 | Data ingestion (fixed-token chunking) | ✅ Done |
| 1.5 | Structure-aware chunking + BM25 tsvector/GIN + hybrid RRF fusion | ⏳ In progress |
| 2 | BGE-M3 dense embedding + HNSW index | ✅ Done |
| 3 | UMAP + HDBSCAN clustering + LLM summaries (tree build) | ✅ Done |
| 4 | Tree-guided retrieval (CLI + FastAPI + Docker) | ✅ Done |
| 5 | Domain workers (Trader SFT / Coder GRPO / Law) + AWQ W4A16 | ✅ Done |
| 5.5 | Self-updating PDF ingestion with zero-shot domain routing | ✅ Done |
| 6 | LangGraph gateway + full Docker Compose stack | ⏳ Next |
| 7 | Synthetic eval set + retrieval & generation metrics | ⏳ Planned |

---

## Project structure

```
PRISM-RAG/
├── README.md                    ← this file
├── ARCHITECTURE.md              ← detailed roadmap, schema, phase verifications
├── init.sql                     ← PostgreSQL schema (incl. tsvector + GIN for BM25)
├── docker-compose.yml
├── pyproject.toml
├── .env.example
│
├── data/                        ← downloaded JSONL + user-ingested PDFs
│   └── download.py
│
├── checkpoints/
│   ├── source_model/            ← Qwen2.5-7B, law_llm bases
│   ├── clallibration_data/      ← trader / coder / legal AWQ calibration sets
│   └── awq_models/              ← shipped W4A16 worker checkpoints
│
├── agents/
│   ├── ingestion/               ← Phases 1, 1.5, 2, 5.5
│   │   ├── chunker.py           ← structure-aware paragraph chunker (Ph 1.5)
│   │   ├── encoder.py           ← BGE-M3 wrapper
│   │   ├── embed_leaves.py      ← chunks → tree_nodes level 0
│   │   ├── pdf_loader.py        ← Ph 5.5
│   │   ├── domain_classifier.py ← Ph 5.5
│   │   └── watcher.py           ← Ph 5.5
│   ├── tree_builder/            ← Phase 3
│   ├── retrieval/               ← Phase 4 + 1.5 hybrid fusion
│   │   └── tree_search.py       ← top-down beam, collapsed, dense/bm25/hybrid (RRF)
│   └── generation/              ← Phase 6 AWQ worker inference
│
├── gateway/                     ← Phase 6: LangGraph StateGraph
├── training/                    ← Phase 5: SFT, GRPO, AWQ
│   ├── qwen_trader_SFT_fine_tune.py
│   ├── qwen_coder_GRPO_fine_tune.py
│   ├── awq_quantize_coder_worker.py
│   └── awq_quantize_law_worker.py
├── evaluation/                  ← Phase 7
│
└── scripts/
    ├── 01_init_db.py
    ├── 02_ingest.py
    ├── 03_embed_chunks.py
    ├── 04_build_tree.py
    ├── 05_query_cli.py
    └── 06_benchmark.py
```

---

## Quick start

**Prerequisites:** Python 3.10+, Docker, ~30 GB free disk, GPU with ≥24 GB
VRAM recommended for training and AWQ quantization. A local
Qwen2.5-7B-Instruct checkpoint under `checkpoints/source_model/qwen_2_5/`.

```bash
git clone https://github.com/Arupreza/PRISM-RAG && cd PRISM-RAG

uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

cp .env.example .env   # set PG_DSN, HF_TOKEN, PRISM_DATA_DIR

docker compose up -d postgres
python data/download.py

# Phase 1 + 1.5: schema (with tsvector GIN) + structure-aware ingest
python scripts/01_init_db.py
python scripts/02_ingest.py

# Phase 2: embed leaves
python scripts/03_embed_chunks.py

# Phase 3: build the topic tree
python scripts/04_build_tree.py

# Phase 4: query from CLI (try all three fusion modes)
python scripts/05_query_cli.py "What does the F-2 visa allow?" \
    --domain law --mode collapsed --fusion hybrid --k 5

python scripts/05_query_cli.py "AAPL Q3 earnings" \
    --domain trading --fusion bm25 --k 5

# Phase 4 service in Docker
docker build -f agents/retrieval/Dockerfile -t prism-retrieval:gpu .
docker run --rm --gpus all --network host --env-file .env prism-retrieval:gpu

# Phase 5: train workers
python training/qwen_trader_SFT_fine_tune.py
python training/qwen_coder_GRPO_fine_tune.py

# Phase 5: quantize to AWQ W4A16
python training/awq_quantize_coder_worker.py
python training/awq_quantize_law_worker.py
```

**Smoke checks:**

```bash
curl -s http://localhost:8001/healthz
curl -s http://localhost:8001/readyz

# Dense (current default)
curl -s -X POST http://localhost:8001/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"query":"F-2 visa renewal","mode":"top_down","fusion":"dense","k":5}'

# Pure BM25 — best for exact-term queries (codes, identifiers, proper nouns)
curl -s -X POST http://localhost:8001/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"query":"F-2-7-7","mode":"top_down","fusion":"bm25","k":5}'

# Hybrid (RRF) — recommended default once Phase 1.5 ships
curl -s -X POST http://localhost:8001/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"query":"F-2 visa renewal","mode":"top_down","fusion":"hybrid","rrf_k":60,"k":5}'
```

---

## Project history

This repository originally described a SPLADE-based sparse multi-agent RAG
over BEIR benchmark datasets. That design was dropped in favor of
tree-guided dense retrieval over a curated multi-domain corpus. The corpus
shifted from the original 4-domain mix (politics, finance, AI, medical) to
a focused 3-domain build (AI, Trading, Korean Immigration Law) aligned
with the domain-specialist worker architecture. Phase 1.5 brings back
lexical search — not as a return to sparse-first, but as the **BM25 half
of a hybrid retriever** alongside the existing dense path. Original design
visible in early git history; reasoning for each pivot in
[`ARCHITECTURE.md`](./ARCHITECTURE.md).

---

## What this project is and isn't

**Is:**
- A research project exploring whether RAPTOR-style hierarchical topic
  trees + hybrid (dense + BM25) leaf retrieval + domain-specialist
  quantized workers outperform flat dense RAG with a generic LLM on
  heterogeneous multi-domain corpora.
- A fully open, single-database implementation (everything in PostgreSQL +
  pgvector + tsvector/GIN — no external vector service, no separate search
  engine).
- A practical reference for the full LLM-engineering pipeline: data
  ingestion, structure-aware chunking, dense+lexical indexing, tree
  construction, fine-tuning (SFT + GRPO), quantization (AWQ), and serving.

**Isn't:**
- Production-ready.
- Yet benchmarked. Performance comparisons against flat baselines,
  dense-only baselines, and generic-LLM baselines will be reported in
  Phase 7 on a synthetic evaluation set.

---

## License

MIT.

---

## Author

**Md Rezanur Islam (Reza)**
LLM Engineer & Agentic AI Developer
PhD Candidate, Software Convergence — Soonchunhyang University (BK21)