# PRISM-RAG
### Parallel Retrieval with Intelligent Sparse Multi-agent RAG

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-009688?style=flat-square&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Latest-2496ED?style=flat-square&logo=docker&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Latest-1C3C3C?style=flat-square&logo=langchain&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![pgvector](https://img.shields.io/badge/pgvector-sparsevec-336791?style=flat-square&logo=postgresql&logoColor=white)
![SPLADE](https://img.shields.io/badge/SPLADE-Vectorless-6C3483?style=flat-square&logoColor=white)
![Qwen](https://img.shields.io/badge/Qwen2.5--7B-AWQ%204bit-FF6F00?style=flat-square&logoColor=white)
![BEIR](https://img.shields.io/badge/BEIR-Benchmark-2ECC71?style=flat-square&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

> A distributed multi-agent Retrieval-Augmented Generation system with domain-specialized agents,
> **vectorless** SPLADE sparse retrieval (no dense embedding model), dual-role pgvector memory,
> LangGraph orchestration, and task-specific Qwen2.5-7B-AWQ fine-tuned models —
> evaluated on BEIR benchmark datasets.

---

## Problem

A user query may require knowledge from multiple heterogeneous domains simultaneously.
No single retriever, no single corpus, and no single agent can answer it correctly.

Existing RAG systems have two additional problems:

1. **Dense embedding dependency** — standard RAG requires a separate GPU embedding server
   (e.g. text-embedding-ada-002, all-MiniLM) running continuously to encode queries and documents.
   This adds latency, cost, and an extra failure point.

2. **Unified corpus** — dumping all domains into one vector store forces a single retrieval model
   to serve science, finance, medical, and general queries equally — it cannot.

**PRISM-RAG** solves both. It routes each query to domain-specialized agents, retrieves grounded
evidence using **SPLADE sparse vectors — no dense embedding model required (vectorless)** —
stores them natively in pgvector `sparsevec`, generates cited answers via task-fine-tuned
Qwen2.5-7B-AWQ, and caches repeated query patterns for sub-100ms response.

---

## Architecture

```
User Query
    │
    ▼
FastAPI Gateway (api-gateway container)
    │
    ▼
LangGraph StateGraph Orchestrator
    │
    ├──→ [Cache Agent]       → pgvector semantic_cache table
    │         │ HIT → return cached answer
    │         │ MISS ↓
    ├──→ [Router Agent]      → Qwen2.5-7B-AWQ (router fine-tune)
    │         │               classify domain(s): science/finance/medical/multihop
    │         ↓
    ├──→ [Retrieval Agents]  → SPLADE sparse vectors → pgvector long_term_memory
    │    ├── Science Agent   → BeIR/scifact corpus
    │    ├── Finance Agent   → BeIR/fiqa corpus
    │    ├── Medical Agent   → BeIR/trec-covid corpus
    │    └── MultiHop Agent  → BeIR/hotpotqa corpus
    │         ↓
    └──→ [Generation Agent]  → Qwen2.5-7B-AWQ (domain fine-tune)
              │               merge retrieved evidence → cited answer
              ↓
         [Cache Write]       → store query+answer in semantic_cache
              │
              ▼
         Final Answer
```

---

## Datasets (HuggingFace BEIR)

| Agent | Dataset | Corpus Size | Queries | Task |
|---|---|---|---|---|
| Science Agent | `BeIR/scifact` | 5,183 docs | 300 | Fact verification |
| Finance Agent | `BeIR/fiqa` | 57,638 docs | 648 | Financial QA |
| Medical Agent | `BeIR/trec-covid` | 171,332 docs | 50 | COVID-19 IR |
| MultiHop Agent | `BeIR/hotpotqa` | 5.2M docs | 7,405 | Multi-hop QA |

---

## Model Stack

| Role | Model | Method |
|---|---|---|
| Sparse encoder | `naver/splade-cocondenser-ensembledistil` | Inference only |
| Router | `Qwen2.5-7B-Instruct` → QLoRA → Merge → AWQ | Task fine-tuned |
| Science generator | `Qwen2.5-7B-Instruct` → QLoRA → Merge → AWQ | Domain fine-tuned |
| Finance generator | `Qwen2.5-7B-Instruct` → QLoRA → Merge → AWQ | Domain fine-tuned |
| Medical generator | `Qwen2.5-7B-Instruct` → QLoRA → Merge → AWQ | Domain fine-tuned |
| MultiHop generator | `Qwen2.5-7B-Instruct` → QLoRA → Merge → AWQ | Domain fine-tuned |

**Fine-tune order (mandatory):**
```
QLoRA Fine-tune → Merge Adapter → AWQ Quantize (domain-calibrated) → Save INT4
```

---

## Container Services

| Container | Port | Role |
|---|---|---|
| `api-gateway` | 8080 | FastAPI entry + LangGraph orchestrator |
| `cache-agent` | 8001 | Semantic cache check and write |
| `retrieval-agent` | 8002 | Domain-routed SPLADE retrieval |
| `generation-agent` | 8003 | AWQ model inference |
| `ingestion-agent` | 8004 | BEIR corpus ingestion + indexing |
| `pgvector-db` | 5432 | PostgreSQL + pgvector (long-term + cache) |
| `redis` | 6379 | Session state + rate limiting |

---

## Project Structure

```
prism-rag/
├── README.md
├── docker-compose.yml
├── init.sql                        # pgvector schema
├── gateway/
│   ├── Dockerfile
│   ├── main.py                     # FastAPI app
│   ├── graph.py                    # LangGraph StateGraph
│   ├── nodes.py                    # Agent HTTP calls
│   ├── models.py                   # Pydantic schemas
│   └── requirements.txt
├── agents/
│   ├── cache/
│   │   ├── Dockerfile
│   │   ├── main.py
│   │   └── cache_service.py
│   ├── retrieval/
│   │   ├── Dockerfile
│   │   ├── main.py
│   │   ├── splade.py
│   │   └── retrieval_service.py
│   ├── generation/
│   │   ├── Dockerfile
│   │   ├── main.py
│   │   └── llm_service.py
│   └── ingestion/
│       ├── Dockerfile
│       ├── main.py
│       ├── splade.py
│       └── chunker.py
├── training/
│   ├── prepare_data.py             # Build QLoRA datasets from BEIR
│   ├── finetune.py                 # QLoRA fine-tuning script
│   ├── merge_adapter.py            # Merge LoRA → FP16
│   ├── quantize_awq.py             # AWQ quantization per task
│   └── configs/
│       ├── router.yaml
│       ├── scifact.yaml
│       ├── fiqa.yaml
│       ├── trec_covid.yaml
│       └── hotpotqa.yaml
├── evaluation/
│   ├── eval_retrieval.py           # nDCG@10, Recall@100
│   ├── eval_generation.py          # EM, F1, Citation accuracy
│   └── eval_cache.py               # Cache hit rate, latency
└── scripts/
    ├── ingest_beir.py              # CLI: load BEIR → pgvector
    └── benchmark.py                # End-to-end latency benchmark
```

---

## Evaluation Metrics

| Metric | Tool | Target |
|---|---|---|
| nDCG@10 per domain | BEIR qrels | > BM25 baseline |
| Recall@100 | BEIR qrels | > dense embedding baseline |
| Answer F1 | BEIR ground truth | Fine-tuned > base Qwen |
| Citation accuracy | Manual / qrels overlap | > 80% |
| Cache hit rate | Internal counter | Measure on HotpotQA |
| E2E latency cache hit | p95 ms | < 100ms |
| E2E latency cache miss | p95 ms | Measure and report |
| Memory (GPU) | nvidia-smi | AWQ < 5GB vs FP16 14GB |

---

---

# 7-Day Build Plan

---

## Day 1 — Infrastructure Setup
**Goal: Docker stack running, pgvector schema initialized, all containers healthy**

### Tasks
- [ ] Create project repository `prism-rag/`
- [ ] Write `docker-compose.yml` with all 7 services
- [ ] Write `init.sql` — create `long_term_memory` and `semantic_cache` tables with `sparsevec(30522)`
- [ ] Verify `pgvector/pgvector:pg16` image supports `sparsevec` type
- [ ] Write stub `main.py` for each agent (health endpoint only)
- [ ] Write `Dockerfile` for each agent container
- [ ] Run `docker compose up` — all containers must return 200 on `/health`

### Deliverable
```bash
curl http://localhost:8080/health     # api-gateway
curl http://localhost:8001/health     # cache-agent
curl http://localhost:8002/health     # retrieval-agent
curl http://localhost:8003/health     # generation-agent
curl http://localhost:8004/health     # ingestion-agent
# all return {"status": "ok"}
```

### Key Files
```
docker-compose.yml
init.sql
agents/*/Dockerfile
agents/*/main.py  (stub)
```

---

## Day 2 — SPLADE Encoder + Ingestion Agent
**Goal: Ingest all 4 BEIR corpora into pgvector as SPLADE sparsevec**

### Tasks
- [ ] Implement `SPLADEEncoder` class using `naver/splade-cocondenser-ensembledistil`
- [ ] Verify sparse vector output: indices + values + dim=30522
- [ ] Implement `ingestion-agent` document chunker
- [ ] Implement batch ingestion endpoint `POST /ingest`
- [ ] Write `scripts/ingest_beir.py` — load each BEIR dataset and push to ingestion agent
- [ ] Ingest `BeIR/scifact` corpus (5,183 docs) — verify in pgvector
- [ ] Ingest `BeIR/fiqa` corpus (57,638 docs)
- [ ] Ingest `BeIR/trec-covid` corpus (171,332 docs)
- [ ] Sample ingest `BeIR/hotpotqa` corpus (start with 100K docs, full later)
- [ ] Verify row counts in `long_term_memory` table

### Deliverable
```sql
SELECT source, COUNT(*) FROM long_term_memory GROUP BY source;
-- scifact:    5183
-- fiqa:      57638
-- trec_covid: 171332
-- hotpotqa:  100000+
```

### Key Files
```
agents/ingestion/splade.py
agents/ingestion/main.py
scripts/ingest_beir.py
```

---

## Day 3 — Retrieval Agent
**Goal: Domain-routed SPLADE retrieval returning top-k docs with nDCG@10 measured**

### Tasks
- [ ] Implement `retrieval-agent` with `POST /retrieve` endpoint
- [ ] Accept: `{query, domain, top_k}`
- [ ] SPLADE encode query → sparsevec → pgvector IP search filtered by `source` field
- [ ] Return: `[{doc_id, title, text, score}]`
- [ ] Write `evaluation/eval_retrieval.py`
- [ ] Load BEIR qrels for each domain
- [ ] Compute nDCG@10 and Recall@100 for all 4 domains
- [ ] Compute BM25 baseline nDCG@10 for comparison
- [ ] Record baseline numbers in results table

### Deliverable
```
Retrieval Baseline Results:
  SciFact   nDCG@10:  BM25=? | PRISM-RAG=?
  FiQA      nDCG@10:  BM25=? | PRISM-RAG=?
  TREC-COVID nDCG@10: BM25=? | PRISM-RAG=?
  HotpotQA  nDCG@10:  BM25=? | PRISM-RAG=?
```

### Key Files
```
agents/retrieval/main.py
agents/retrieval/retrieval_service.py
agents/retrieval/splade.py
evaluation/eval_retrieval.py
```

---

## Day 4 — Cache Agent + Semantic Cache
**Goal: Semantic cache operational with threshold-based hit/miss, latency measured**

### Tasks
- [ ] Implement `cache-agent` with `POST /check` and `POST /write` endpoints
- [ ] `POST /check`: SPLADE encode query → IP search `semantic_cache` → return hit/miss
- [ ] `POST /write`: store `{query_text, query_vec, answer}` in `semantic_cache`
- [ ] Set similarity threshold = 0.92 (tunable via env var)
- [ ] Write `evaluation/eval_cache.py`
- [ ] Run all 7,405 HotpotQA queries twice — measure cache hit rate on second pass
- [ ] Measure p50/p95 latency: cache hit vs cache miss
- [ ] Tune threshold: 0.85 / 0.90 / 0.92 / 0.95 — record hit rate vs precision tradeoff

### Deliverable
```
Cache Evaluation:
  Threshold 0.92:
    Hit rate (HotpotQA 2nd pass): ?%
    Latency cache hit  p95: ? ms
    Latency cache miss p95: ? ms
    Answer consistency: ?%
```

### Key Files
```
agents/cache/main.py
agents/cache/cache_service.py
evaluation/eval_cache.py
```

---

## Day 5 — QLoRA Fine-tune + AWQ Quantization
**Goal: 5 task-specific Qwen2.5-7B-AWQ models saved and loadable**

### Tasks

**Data Preparation**
- [ ] Write `training/prepare_data.py`
- [ ] Build router training data: auto-label BEIR queries by source domain → 4-class JSON format
- [ ] Build domain QA pairs for each generator: query + retrieved docs + ground truth answer
- [ ] Split each dataset: 80% train / 10% val / 10% calibration (no overlap)

**QLoRA Fine-tuning (per task: router, scifact, fiqa, trec_covid, hotpotqa)**
- [ ] Write `training/finetune.py` with BitsAndBytesConfig + LoraConfig
- [ ] Fine-tune router model on domain classification data
- [ ] Fine-tune 4 generator models on domain QA data
- [ ] Track training loss with W&B per task

**Merge + AWQ**
- [ ] Write `training/merge_adapter.py` — merge LoRA into FP16
- [ ] Write `training/quantize_awq.py` — AWQ quantize with domain calibration data
- [ ] Run AWQ for all 5 merged models
- [ ] Verify each AWQ model loads and generates correctly
- [ ] Measure GPU memory: FP16 vs AWQ per model

### Deliverable
```
training/
  awq_models/
    router/       # ~4GB
    scifact/      # ~4GB
    fiqa/         # ~4GB
    trec_covid/   # ~4GB
    hotpotqa/     # ~4GB

GPU Memory:
  Qwen2.5-7B FP16:  ~14GB
  Qwen2.5-7B AWQ:   ~4GB
  Reduction:         3.5×
```

### Key Files
```
training/prepare_data.py
training/finetune.py
training/merge_adapter.py
training/quantize_awq.py
training/configs/*.yaml
```

---

## Day 6 — Router Agent + Generation Agent + LangGraph Wiring
**Goal: Full agent pipeline working end-to-end without FastAPI gateway**

### Tasks

**Router Agent**
- [ ] Load router AWQ model in `generation-agent` container (shared container, task param)
- [ ] Implement `POST /route`: query → domain classification JSON
- [ ] Test on 50 queries across 4 domains — verify routing accuracy

**Generation Agent**
- [ ] Implement `POST /generate`: `{query, domain, retrieved_docs}` → cited answer
- [ ] Load correct AWQ model based on `domain` param
- [ ] Implement structured prompt per domain (science / finance / medical / multihop)
- [ ] Test generation quality on 10 samples per domain

**LangGraph Wiring**
- [ ] Write `gateway/graph.py` — full StateGraph with all nodes
- [ ] Implement all node functions in `gateway/nodes.py` (HTTP calls to agents)
- [ ] Define conditional edges: cache hit → END, cache miss → router → retrieval → generation → cache_write → END
- [ ] Test graph execution with mock inputs — verify state transitions
- [ ] Test graph with real query — verify full pipeline

### Deliverable
```python
# Direct graph test (no FastAPI yet)
graph = build_graph()
result = graph.invoke({
    "query": "What are the effects of mRNA vaccines on T-cell response?",
    "cache_hit": False,
    "retrieved_docs": [],
    "final_answer": ""
})
print(result["final_answer"])  # must print cited answer
```

### Key Files
```
gateway/graph.py
gateway/nodes.py
agents/generation/main.py
agents/generation/llm_service.py
```

---

## Day 7 — FastAPI Gateway + Evaluation + Benchmark
**Goal: Full system live, all metrics measured, results table complete**

### Tasks

**FastAPI Gateway**
- [ ] Implement `gateway/main.py` — wire LangGraph into FastAPI endpoint
- [ ] `POST /query`: `{query}` → full pipeline → `{answer, sources, latency_ms, cache_hit}`
- [ ] `POST /ingest`: trigger ingestion agent
- [ ] `GET /stats`: cache hit rate, query count, avg latency
- [ ] Add request logging middleware
- [ ] Run `docker compose up` — full stack

**End-to-End Evaluation**
- [ ] Write `evaluation/eval_generation.py` — compute F1, EM, citation accuracy
- [ ] Run 50 queries per domain through full pipeline
- [ ] Record: retrieval nDCG@10, answer F1, citation accuracy, cache hit rate
- [ ] Compare fine-tuned AWQ vs base Qwen2.5-7B (no fine-tune) on answer F1

**Latency Benchmark**
- [ ] Write `scripts/benchmark.py` — run 100 queries, measure p50/p95
- [ ] Measure separately: cache hit latency, single-domain miss, multi-domain miss

**Final Results Table**
- [ ] Fill complete results table (see Evaluation Metrics section)
- [ ] Screenshot working API on Postman or curl
- [ ] Tag git commit as `v0.1.0`

### Deliverable
```bash
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the financial impact of COVID-19 on biotech firms?"}'

# Returns:
{
  "answer": "According to [fiqa_doc_1432] and [trec_covid_doc_882], ...",
  "sources": ["fiqa_doc_1432", "trec_covid_doc_882"],
  "domains_routed": ["finance", "medical"],
  "cache_hit": false,
  "latency_ms": 1240
}
```

### Key Files
```
gateway/main.py
evaluation/eval_generation.py
scripts/benchmark.py
```

---

## Daily Checklist Summary

| Day | Focus | Done When |
|---|---|---|
| 1 | Infrastructure | All 7 containers return `/health` 200 |
| 2 | Ingestion | All BEIR corpora in pgvector, row counts verified |
| 3 | Retrieval | nDCG@10 measured, beats BM25 baseline |
| 4 | Cache | Hit rate measured, p95 latency < 100ms on cache hit |
| 5 | Fine-tune + AWQ | 5 AWQ models saved, GPU memory verified |
| 6 | Agents + LangGraph | Full pipeline runs query-to-answer without gateway |
| 7 | Gateway + Eval | API live, all metrics in results table |

---

## Quick Start

```bash
# Clone and start infrastructure
git clone https://github.com/rezanur/prism-rag
cd prism-rag

# Start all containers
docker compose up --build

# Ingest BEIR datasets
python scripts/ingest_beir.py --domains scifact fiqa trec_covid hotpotqa

# Run evaluation
python evaluation/eval_retrieval.py --domain scifact
python evaluation/eval_cache.py --dataset hotpotqa

# Query the system
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Your question here"}'
```

---

## Vectorless Design

**Vectorless** means: no dense embedding model anywhere in this pipeline.

Standard RAG architecture requires a dense embedding model running as a live service:

```
Standard RAG:
  Query → [Embedding Server: all-MiniLM / text-ada-002] → dense vector → FAISS/Weaviate → docs
              ↑
        GPU server, always-on, extra latency, extra cost, extra failure point
```

PRISM-RAG eliminates this entirely:

```
PRISM-RAG (Vectorless):
  Query → [SPLADE: MLM forward pass] → sparse lexical vector → pgvector sparsevec → docs
              ↑
        Lightweight, CPU-capable, no separate embedding service
```

### How SPLADE Produces Vectors Without a Dense Embedding Model

SPLADE uses a **Masked Language Model head** over the full BERT vocabulary (30,522 tokens).
For each token position in the input, it outputs a weight across all vocab terms.
The final sparse vector = max pooling + ReLU + log saturation across all positions.

```python
# What SPLADE produces — not a dense 768-dim vector
# A sparse vector with ~200 non-zero entries out of 30,522 dimensions
{
  "indices": [1045, 2182, 4327, 8901, ...],   # vocab token IDs with activation
  "values":  [0.82, 0.61, 0.44, 0.38, ...],  # activation weights
  "dim": 30522
}
```

This is stored directly in pgvector as `sparsevec(30522)`.
No embedding server. No FAISS index. No Weaviate cluster.
Inner product search runs natively inside PostgreSQL.

### What This Removes From Your Infrastructure

| Component | Standard RAG | PRISM-RAG (Vectorless) |
|---|---|---|
| Embedding server | Required (GPU/API) | Not needed |
| Vector index service | FAISS / Weaviate / Pinecone | pgvector native |
| Encoding latency | 20–100ms per query | SPLADE ~15ms CPU |
| Infrastructure cost | Extra service + GPU | Zero extra |
| Failure points | +1 (embedding API) | Removed |

### pgvector sparsevec Storage

```sql
-- Documents stored as sparse vectors — no dense embedding column
CREATE TABLE long_term_memory (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content     TEXT NOT NULL,
    sparse_vec  sparsevec(30522),          -- SPLADE output, ~200 non-zero
    source      TEXT,                      -- scifact / fiqa / trec_covid / hotpotqa
    metadata    JSONB,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- IP similarity search — inner product on sparse vectors
SELECT content, sparse_vec <#> query_vec::sparsevec AS score
FROM long_term_memory
WHERE source = 'scifact'
ORDER BY score ASC
LIMIT 5;
```

---

## Stack

| Component | Technology |
|---|---|
| Orchestration | LangGraph |
| API | FastAPI + uvicorn |
| Containers | Docker + docker-compose |
| Vector DB | PostgreSQL + pgvector (`sparsevec` type) |
| Vector Strategy | **Vectorless** — SPLADE sparse lexical vectors, no dense embedding model |
| Sparse Encoder | `naver/splade-cocondenser-ensembledistil` (MLM-based, CPU-capable) |
| LLM | Qwen2.5-7B-Instruct |
| Fine-tuning | QLoRA (PEFT + BitsAndBytes) |
| Quantization | AWQ (4-bit, domain-calibrated) |
| Cache | Redis (session) + pgvector semantic_cache (vectorless lookup) |
| Evaluation | BEIR qrels, nDCG@10, Recall@100, F1 |
| Tracking | Weights & Biases |

---

## Author

**Md Rezanur Islam (Reza)**
LLM Engineer & Agentic AI Developer
PhD Candidate, Software Convergence — Soonchunhyang University (BK21)