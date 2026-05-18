# PRISM-RAG — Architecture & Build Roadmap

Tree-guided dense RAG over a multi-domain corpus (politics, finance, ai_tech,
medical). This document supersedes the SPLADE multi-agent description in
`README.md` — that README will be rewritten when Phase 5 ships.

---

## Why the architecture changed

| Aspect | Original (SPLADE) | Current (Tree-RAG) |
|---|---|---|
| Retrieval encoder | SPLADE sparse (`sparsevec(30522)`) | Dense BGE-M3 (`vector(1024)`) |
| Corpus | BEIR (scifact, fiqa, trec-covid, hotpotqa) | Your downloaded JSONL (politics, finance, ai_tech, medical) |
| Index structure | Flat per-source | Hierarchical tree (RAPTOR-style) |
| Domain routing | Qwen router agent | Tree traversal (top-down filter) |
| Eval framework | BEIR qrels | Synthetic LLM-generated QA + tree quality metrics |

Folder structure from the original repo is preserved where it makes sense. The
SPLADE-specific content inside those folders is replaced.

---

## Status legend

| Marker | Meaning |
|---|---|
| ✓ | Built and validated |
| → | Current phase |
| ⏳ | Planned, not started |
| ✗ | Dropped from original design |

---

## Full target tree

```
PRISM-RAG/
├── README.md                          ⏳ rewrite after Phase 5
├── ARCHITECTURE.md                    ✓ this file
├── .gitignore                         (existing)
├── .env.example                       ✓ Phase 1
├── .env                               (you create from .env.example)
├── docker-compose.yml                 ⏳ Phase 5 — rewrite (currently SPLADE stub)
├── init.sql                           ✓ Phase 1 — dense schema
├── requirements.txt                   ✓ Phase 1
│
├── data/                              (your downloader output, untouched)
│   ├── politics/
│   │   ├── cc_news.jsonl
│   │   └── congressional_speeches.jsonl
│   ├── finance/
│   │   └── financial_news.jsonl
│   ├── ai_tech/
│   │   └── ml_arxiv_papers.jsonl
│   └── medical/
│       ├── pubmed_papers.jsonl
│       └── arxiv_papers.jsonl
│
├── checkpoints/                       ⏳ Phase 6 — LoRA adapters, AWQ models
│
├── agents/
│   ├── __init__.py                    ✓ Phase 1
│   │
│   ├── ingestion/                     batch pipeline (Phases 1-2)
│   │   ├── __init__.py                ✓ Phase 1
│   │   ├── config.py                  ✓ Phase 1
│   │   ├── db.py                      ✓ Phase 1
│   │   ├── chunker.py                 ✓ Phase 1
│   │   ├── loader.py                  ✓ Phase 1   (JSONL → documents + chunks)
│   │   ├── encoder.py                 → Phase 2   (BGE-M3 wrapper)
│   │   └── embed_leaves.py            → Phase 2   (chunks → tree_nodes level 0)
│   │
│   ├── tree_builder/                  ⏳ Phase 3 — NEW folder, batch pipeline
│   │   ├── __init__.py
│   │   ├── cluster.py                 UMAP + HDBSCAN
│   │   ├── summarizer.py              LLM cluster summaries (vLLM client)
│   │   └── build.py                   recursive tree builder
│   │
│   ├── retrieval/                     ⏳ Phase 4 — FastAPI service (replaces SPLADE)
│   │   ├── __init__.py
│   │   ├── tree_search.py             top-down + collapsed retrieval
│   │   ├── main.py                    FastAPI app, POST /retrieve
│   │   └── Dockerfile
│   │
│   ├── generation/                    ⏳ Phase 5 — FastAPI service
│   │   ├── __init__.py
│   │   ├── prompts.py                 per-domain answer prompts
│   │   ├── llm_service.py             Qwen-AWQ inference
│   │   ├── main.py                    FastAPI app, POST /generate
│   │   └── Dockerfile
│   │
│   └── cache/                         ✗ DROP — was SPLADE-specific. Re-add later if needed.
│
├── gateway/                           ⏳ Phase 5 — orchestrator
│   ├── __init__.py
│   ├── main.py                        FastAPI app, POST /query
│   ├── graph.py                       LangGraph StateGraph
│   ├── nodes.py                       node functions (HTTP calls to agents)
│   ├── models.py                      Pydantic schemas
│   └── Dockerfile
│
├── training/                          ⏳ Phase 6 — OPTIONAL for v0.1
│   ├── prepare_data.py                build QLoRA datasets from retrieval+answers
│   ├── finetune.py                    QLoRA (PEFT + BitsAndBytes)
│   ├── merge_adapter.py               LoRA → FP16
│   ├── quantize_awq.py                AWQ 4-bit, domain-calibrated
│   └── configs/
│       ├── politics.yaml
│       ├── finance.yaml
│       ├── ai_tech.yaml
│       └── medical.yaml
│
├── evaluation/                        ⏳ Phase 7
│   ├── synthetic_qa_gen.py            LLM-generate eval QA pairs from your docs
│   ├── eval_tree_quality.py           silhouette, cluster coherence, depth balance
│   ├── eval_retrieval.py              Recall@k, MRR@k, nDCG@k on synthetic set
│   └── eval_generation.py             F1, faithfulness, citation accuracy
│
└── scripts/                           orchestration entry points (run from repo root)
    ├── 01_init_db.py                  ✓ Phase 1
    ├── 02_ingest.py                   ✓ Phase 1
    ├── 03_embed_chunks.py             → Phase 2
    ├── 04_build_tree.py               ⏳ Phase 3
    ├── 05_query_cli.py                ⏳ Phase 4 (CLI before gateway)
    └── 06_benchmark.py                ⏳ Phase 7
```

---

## Phase rollout

Each phase has: (1) new files created, (2) expected database state after,
(3) verification queries / commands. **Do not skip a phase's verification.**

### Phase 1 — Ingest ✓ DONE

**New files:** `init.sql`, `requirements.txt`, `.env.example`,
`agents/__init__.py`, `agents/ingestion/{__init__,config,db,chunker,loader}.py`,
`scripts/{01_init_db,02_ingest}.py`.

**DB state:** `documents` and `chunks` populated for sampled JSONL files.
`tree_nodes` empty.

**Verify:**
```sql
SELECT domain, source, COUNT(*) FROM documents GROUP BY 1,2 ORDER BY 1,2;
SELECT MIN(n_tokens), AVG(n_tokens)::int, MAX(n_tokens) FROM chunks;
-- max chunk n_tokens MUST be <= 512
```

---

### Phase 2 — Embed chunks as leaf nodes →

**New files:**
- `agents/ingestion/encoder.py` — `class BGEM3Encoder` wrapping
  `sentence-transformers`. Batched, GPU, normalized vectors.
- `agents/ingestion/embed_leaves.py` — reads chunks not yet embedded, encodes
  in batches, inserts into `tree_nodes` with `level=0`, `is_leaf=true`.
- `scripts/03_embed_chunks.py` — orchestrator.

**DB state after:** every row in `chunks` has a corresponding row in
`tree_nodes` with `is_leaf=true`, `level=0`, `summary = chunks.text`,
`embedding` populated.

**Verify:**
```sql
-- Every chunk has exactly one leaf node
SELECT
  (SELECT COUNT(*) FROM chunks) AS n_chunks,
  (SELECT COUNT(*) FROM tree_nodes WHERE level=0) AS n_leaves,
  (SELECT COUNT(*) FROM chunks)
    - (SELECT COUNT(*) FROM tree_nodes WHERE level=0) AS missing;

-- Embedding sanity: vectors normalized? (cosine cluster needs this)
SELECT AVG(sqrt(embedding <-> embedding))::numeric(6,4) AS avg_norm
FROM tree_nodes WHERE level=0;
-- expect ≈ 0 for L2 self-distance; sqrt(<#>) approach for norm
```

**Don't move on until:** every chunk has a leaf node, no nulls in embedding.

---

### Phase 3 — Build the tree ⏳

**New files:**
- `agents/tree_builder/cluster.py` — UMAP dim-reduce, HDBSCAN cluster, returns
  cluster assignments. Pure numpy, no DB.
- `agents/tree_builder/summarizer.py` — vLLM client. Takes a list of child
  summaries, returns `{title, summary}` JSON. Strict JSON-mode, temp=0.
- `agents/tree_builder/build.py` — main loop: for each `(domain, source)`,
  iterate L=0→max_levels: fetch level-L nodes → cluster → summarize each
  cluster → embed summary → insert as level-L+1 node → set children's
  `parent_id`. Stop when one cluster remains or max depth hit.
- `scripts/04_build_tree.py` — runs for each (domain, source), prints tree shape.

**Prereq:** vLLM server running.
```bash
# Start vLLM with Qwen2.5-32B-Instruct (adjust GPU count to your hardware)
vllm serve Qwen/Qwen2.5-32B-Instruct \
  --tensor-parallel-size 2 \
  --max-model-len 8192 \
  --port 8000
```

**DB state after:** `tree_nodes` has rows at multiple levels. Parent pointers
form a forest (one tree per `(domain, source)` pair, or one tree per domain
if you collapse the level above source).

**Verify:**
```sql
-- Tree shape per domain/source
SELECT domain, source, level, COUNT(*) AS n_nodes,
       AVG(n_descendants)::int AS avg_leaves_under
FROM tree_nodes
GROUP BY 1,2,3 ORDER BY 1,2,3;

-- No orphans below the top level
SELECT COUNT(*) FROM tree_nodes WHERE parent_id IS NULL AND level < (
  SELECT MAX(level) FROM tree_nodes
);

-- Read 5 internal node summaries — do the titles name coherent topics?
SELECT domain, source, level, title, LEFT(summary, 300) AS preview
FROM tree_nodes WHERE NOT is_leaf
ORDER BY RANDOM() LIMIT 5;
```

**Don't move on until:** internal node titles read like topic names a human
would write. If they're vague ("various political topics"), the summarizer
prompt or cluster size is wrong. Fix before Phase 4.

---

### Phase 4 — Tree-guided retrieval ⏳

**New files:**
- `agents/retrieval/tree_search.py` — implements TWO retrieval strategies:
  1. **Top-down traversal**: ANN at level k → keep top-N children → ANN
     restricted to their subtree at level k-1 → repeat until leaves.
  2. **Collapsed**: flat ANN over all nodes (any level), then re-rank to favor
     leaves under the highest-scoring internal nodes. Often more robust.
- `agents/retrieval/main.py` — FastAPI service, `POST /retrieve` returns
  top-k leaf chunks + the traversal path used.
- `agents/retrieval/Dockerfile`.
- `scripts/05_query_cli.py` — CLI that calls `tree_search` directly (skipping
  FastAPI) so you can debug retrieval logic without service overhead.

**Verify:**
```bash
python scripts/05_query_cli.py "What did Congress say about voter ID laws?"
# Expected: traversal path through politics → congressional_speeches →
# some voter-rights cluster → top-5 chunks. Path printed; chunks printed.
```

**Open question for Phase 4:** which strategy (traversal vs collapsed) wins
on your data? Measure on the synthetic eval set in Phase 7 — don't decide
ahead of time.

---

### Phase 5 — Generation + Gateway ⏳

**New files:**
- `agents/generation/{prompts,llm_service,main,Dockerfile}.py` — Qwen-AWQ
  inference (or base Qwen if you skip Phase 6).
- `gateway/{main,graph,nodes,models,Dockerfile}.py` — FastAPI + LangGraph
  StateGraph wiring retrieval → generation.
- `docker-compose.yml` — REWRITE to compose: pgvector-db, retrieval-agent,
  generation-agent, gateway. (No cache-agent, no SPLADE.)

**Verify:**
```bash
docker compose up --build
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the financial impact of mRNA vaccine R&D on biotech firms?"}'
# Expected: {answer, sources, latency_ms, tree_path}
```

---

### Phase 6 — Domain fine-tuning ⏳ (optional for v0.1)

This phase mirrors your original Day 5 plan. The mechanics
(QLoRA → merge → AWQ) are unchanged. Only the training data shape changes:
inputs are `(query, retrieved_chunks_from_tree, ground_truth_answer)`
triples instead of BEIR-derived ones.

**Skip-or-do decision:** base Qwen2.5-7B-Instruct with a good prompt may be
"good enough" for a learning v0.1. Fine-tune in v0.2 once you have eval
numbers showing where it falls short.

---

### Phase 7 — Evaluation ⏳

**Eval set construction** (your data has no qrels):
- `evaluation/synthetic_qa_gen.py` — for each domain, sample 100 chunks →
  prompt Qwen to generate `(question, answer, source_chunk_id)` triples.
  Manually review ~20% for quality. Discard junk. This is your gold set.

**Metrics:**
- `eval_tree_quality.py` — silhouette score per cluster level, % singleton
  clusters, average depth, % leaves with valid parent path.
- `eval_retrieval.py` — Recall@{1,5,10}, MRR@10 on the synthetic gold set.
  Run both traversal and collapsed strategies; compare.
- `eval_generation.py` — F1 vs gold answer, faithfulness (does the answer
  cite real retrieved chunks?), citation precision.

---

## Sequencing rule

Validate each phase against its verification queries BEFORE starting the
next. If you skip ahead, you'll spend more time debugging cascading failures
than you would have spent verifying. This is the most common mistake on
multi-phase pipelines.

---

## Where you are now

- Phase 1 files have been generated.
- Next action: place them in the repo, run the Phase 1 commands, verify with
  the SQL queries above.
- After verification passes, Phase 2 (BGE-M3 encoder + leaf insertion).