# PRISM-RAG — Architecture & Build Roadmap

Tree-guided dense + BM25 hybrid RAG over a 3-domain corpus (AI, Trading,
Korean Immigration Law), paired with three AWQ-quantized domain-specialist
worker LLMs sharing a Qwen2.5 base. This document is the canonical roadmap;
`README.md` is the user-facing entry point.

---

## Architecture summary

| Aspect | Implementation |
|---|---|
| Dense encoder | BGE-M3 (`vector(1024)`, L2-normalized) |
| Lexical index | PostgreSQL `tsvector` + GIN (BM25-style via `ts_rank_cd`) |
| Hybrid fusion | Reciprocal Rank Fusion (RRF, k=60) — rank-based, scale-free |
| Chunker | Structure-aware: paragraph-atomic, sentence-bisect on overflow, merge-on-short |
| Corpus | AI, Trading, Korean Immigration Law (JSONL + user PDFs) |
| Index structure | Hierarchical RAPTOR-style tree, single HNSW + leaf-only GIN |
| Domain routing | Cosine-argmax of query embedding vs. cached tree-root embeddings |
| Ingestion | Self-updating — watched-folder PDF ingest with hash-dedup + incremental HNSW |
| Generation | Three AWQ W4A16 domain workers (Trader SFT / Coder GRPO / Law base) |
| Eval framework | Synthetic per-domain QA + worker-vs-base + fusion-mode ablations |

---

## Status legend

| Marker | Meaning |
|---|---|
| ✓ | Built and validated |
| → | Current phase |
| ⏳ | Planned, not started |

---

## Full target tree

```
PRISM-RAG/
├── README.md                          ✓ user-facing entry point
├── ARCHITECTURE.md                    ✓ this file
├── .gitignore
├── .env.example                       ✓ Phase 1
├── .env                               (you create from .env.example)
├── docker-compose.yml                 → Phase 6 — extends with generation + gateway services
├── init.sql                           → Phase 1.5 — adds tsvector + GIN to existing schema
├── pyproject.toml                     ✓ uv-managed dependencies
│
├── data/                              JSONL corpus + user-ingested PDFs
│   ├── ai/
│   │   └── ml_arxiv_papers.jsonl
│   ├── trading/
│   │   ├── financial_news.jsonl
│   │   └── user_pdfs/                 ← Phase 5.5 watched ingestion target
│   └── law/
│       └── user_pdfs/                 ← Korean Immigration Law PDFs
│
├── checkpoints/
│   ├── source_model/                  base model weights
│   │   ├── qwen_2_5/                  ← Qwen2.5-7B-Instruct (summarizer + trader/coder base)
│   │   ├── qwen_coder/                ← Qwen2.5-Coder base (optional separate coder base)
│   │   └── law_llm/                   ← law worker base
│   ├── clallibration_data/            AWQ calibration corpora
│   │   ├── trader/                    SujetFinance + finance_alpaca
│   │   ├── coder/                     verifiable-coding-problems + LeetCodeDataset
│   │   └── legal/                     CUAD-QA + LegalQAEval
│   └── awq_models/                    shipped W4A16 worker checkpoints
│       ├── qwen_trader_sft_lora/      LoRA adapters (intermediate)
│       ├── qwen_trader_sft_merged_fp16/
│       ├── qwen_coder_grpo_lora/      LoRA adapters (intermediate)
│       ├── qwen_coder_merged_fp16/
│       ├── qwen_coder_awq_w4a16/      ← final shipped coder worker
│       └── law_llm_awq_w4a16/         ← final shipped law worker
│
├── agents/
│   ├── __init__.py                    ✓ Phase 1
│   │
│   ├── ingestion/                     batch + streaming pipeline
│   │   ├── __init__.py                ✓ Phase 1
│   │   ├── config.py                  ✓ Phase 1 (extended in Ph 1.5 with chunker config)
│   │   ├── db.py                      ✓ Phase 1
│   │   ├── chunker.py                 → Phase 1.5 — structure-aware (paragraph + sentence-bisect + merge)
│   │   ├── sentence_splitter.py       → Phase 1.5 — NLTK/spaCy boundary detection
│   │   ├── loader.py                  ✓ Phase 1   (JSONL → documents + chunks)
│   │   ├── encoder.py                 ✓ Phase 2   (BGE-M3 wrapper, L2-normalized)
│   │   ├── embed_leaves.py            ✓ Phase 2   (chunks → tree_nodes level 0)
│   │   ├── pdf_loader.py              ✓ Phase 5.5 (PDF → text → chunks)
│   │   ├── domain_classifier.py       ✓ Phase 5.5 (cosine-argmax vs. tree roots)
│   │   └── watcher.py                 ✓ Phase 5.5 (watched-folder daemon, hash-dedup)
│   │
│   ├── tree_builder/                  ✓ Phase 3 — batch pipeline
│   │   ├── __init__.py
│   │   ├── cluster.py                 ✓ UMAP + HDBSCAN + noise reassignment
│   │   ├── summarizer.py              ✓ in-process Qwen2.5-7B summarizer (greedy, JSON-mode)
│   │   ├── build.py                   ✓ recursive tree builder
│   │   └── incremental.py             ✓ Phase 5.5 (nearest-cluster reassign + threshold re-summarize)
│   │
│   ├── retrieval/                     ✓ Phase 4 — FastAPI service (extended in Ph 1.5)
│   │   ├── __init__.py
│   │   ├── tree_search.py             ✓ top-down beam + collapsed re-rank
│   │   ├── bm25_search.py             → Phase 1.5 — ts_rank_cd over leaf tsvector
│   │   ├── fusion.py                  → Phase 1.5 — Reciprocal Rank Fusion
│   │   ├── main.py                    ✓ FastAPI: POST /retrieve (extended with fusion params)
│   │   └── Dockerfile                 ✓ uv-based slim image (CPU default, CUDA via build-arg)
│   │
│   └── generation/                    → Phase 6 — FastAPI worker service
│       ├── __init__.py
│       ├── prompts.py                 per-domain answer prompts (trader / coder / law)
│       ├── worker_registry.py         maps domain → AWQ checkpoint path
│       ├── awq_service.py             autoawq inference wrapper
│       ├── main.py                    FastAPI app, POST /generate
│       └── Dockerfile
│
├── gateway/                           → Phase 6 — orchestrator
│   ├── __init__.py
│   ├── main.py                        FastAPI app, POST /query
│   ├── graph.py                       LangGraph StateGraph (route → retrieve → generate)
│   ├── nodes.py                       node functions (HTTP calls to agents)
│   ├── domain_router.py               cosine-argmax routing using cached tree-root embeddings
│   ├── models.py                      Pydantic schemas
│   └── Dockerfile
│
├── training/                          ✓ Phase 5 — worker training + AWQ quantization
│   ├── qwen_trader_SFT_fine_tune.py   ✓ TRL SFTTrainer + QLoRA r=32 (trader)
│   ├── qwen_coder_GRPO_fine_tune.py   ✓ TRL GRPOTrainer + 4 exec rewards (coder)
│   ├── awq_quantize_coder_worker.py   ✓ LoRA merge + AWQ W4A16 calibration (coder)
│   └── awq_quantize_law_worker.py     ✓ AWQ W4A16 with 30% refusal injection (law)
│
├── evaluation/                        ⏳ Phase 7
│   ├── synthetic_qa_gen.py            per-domain LLM-generated QA pairs
│   ├── eval_tree_quality.py           silhouette, singleton %, depth balance
│   ├── eval_retrieval.py              Recall@k, MRR@k, nDCG@k; dense vs bm25 vs hybrid
│   ├── eval_generation.py             F1, faithfulness, citation accuracy
│   └── eval_worker_vs_base.py         ablation: domain worker vs. generic Qwen2.5
│
└── scripts/                           orchestration entry points (run from repo root)
    ├── 01_init_db.py                  ✓ Phase 1 (re-run in Ph 1.5 to add tsvector + GIN)
    ├── 02_ingest.py                   ✓ Phase 1 (uses new chunker after Ph 1.5)
    ├── 03_embed_chunks.py             ✓ Phase 2
    ├── 04_build_tree.py               ✓ Phase 3
    ├── 05_query_cli.py                ✓ Phase 4 (extended with --fusion flag in Ph 1.5)
    └── 06_benchmark.py                ⏳ Phase 7
```

---

## Phase rollout

Each phase has: (1) new files created, (2) expected database state after,
(3) verification queries / commands. **Do not skip a phase's verification.**

### Phase 1 — Ingest ✓ DONE

**New files:** `init.sql`, `pyproject.toml`, `.env.example`,
`agents/__init__.py`, `agents/ingestion/{__init__,config,db,chunker,loader}.py`,
`scripts/{01_init_db,02_ingest}.py`.

**DB state:** `documents` and `chunks` populated for the 3-domain corpus.
`tree_nodes` empty.

**Verify:**
```sql
SELECT domain, source, COUNT(*) FROM documents GROUP BY 1,2 ORDER BY 1,2;
SELECT MIN(n_tokens), AVG(n_tokens)::int, MAX(n_tokens) FROM chunks;
-- max chunk n_tokens MUST be <= 512
```

---

### Phase 1.5 — Structure-aware chunking + BM25 hybrid retrieval → IN PROGRESS

Two coupled upgrades that ship together. Both touch the ingestion path and
the retrieval path, so they're scheduled as one phase to avoid an
intermediate state where chunks have changed but retrieval hasn't been
re-tuned for them.

#### 1.5a — Structure-aware chunker

**New files:**
- `agents/ingestion/chunker.py` — rewritten. New logic:
  1. **Paragraph segmentation** — split on `\n\n+` for text/JSONL;
     numbered-clause regex (`/^Article \d+|Section \d+\.\d+/m`) for legal
     PDFs.
  2. **Token-counting per paragraph** with the BGE-M3 tokenizer.
  3. **Size routing:**
     - `tokens ≤ MAX_CHUNK_TOKENS (512)`: emit as one chunk.
     - `tokens > MAX_CHUNK_TOKENS`: recursively bisect on the nearest
       sentence boundary (binary-search the midpoint sentence) until each
       piece fits. Apply `OVERLAP_TOKENS_ON_SPLIT (64)` only across these
       artificial splits.
     - `tokens < MIN_CHUNK_TOKENS (100)`: merge with the next paragraph if
       the combined size stays under `MAX_CHUNK_TOKENS`.
  4. **Per-source rules** dispatch:
     - `trading_news`: aggressive merging (typical paragraphs are
       1–3 sentences).
     - `ml_arxiv`: preserve paragraphs as-is (paper paragraphs are
       semantically meaningful).
     - `law_pdf`: clause-based chunking when clause structure detected;
       fall back to paragraph chunking otherwise.
- `agents/ingestion/sentence_splitter.py` — wrapper around NLTK
  `sent_tokenize` (default) or spaCy (fallback for Korean text). Cached
  per process.

**Config additions** (`agents/ingestion/config.py`):
```python
MAX_CHUNK_TOKENS = 512
MIN_CHUNK_TOKENS = 100
OVERLAP_TOKENS_ON_SPLIT = 64    # ONLY across artificial sentence-bisect splits
SENTENCE_SPLITTER = "nltk"
PER_SOURCE_CHUNK_RULES = {
    "trading_news": {"merge_aggressive": True},
    "ml_arxiv":     {"merge_aggressive": False},
    "law_pdf":      {"chunk_on_clauses": True},
}
```

**Critical design decision: no overlap across natural paragraph boundaries.**
Overlap exists to protect against meaning loss at *artificial* cuts. Copying
the end of paragraph N into the start of paragraph N+1 — when those are
semantically distinct paragraphs — just adds noise to N+1's embedding. The
current naive overlap-everywhere chunker damages embedding quality on
clean-paragraph corpora (most ArXiv papers); the new chunker fixes this.

**Verify:**
```sql
-- Chunk size distribution should sit cleanly in [100, 512]
SELECT
  width_bucket(n_tokens, 0, 600, 12) AS bucket,
  MIN(n_tokens) AS lo, MAX(n_tokens) AS hi,
  COUNT(*)
FROM chunks
GROUP BY 1 ORDER BY 1;
-- expect: most chunks in buckets covering [100, 512]
-- expect: very few chunks under 50 tokens (single-sentence outliers only)

-- Per-source chunking sanity
SELECT source, AVG(n_tokens)::int AS avg_size, COUNT(*) AS n_chunks
FROM chunks GROUP BY source ORDER BY 1;
-- expect trading_news avg ~= 300 (after merge), arxiv ~= 200,
-- law_pdf varies by clause length
```

**Don't move on until:** chunk size distribution looks reasonable per
source. If trading_news still has thousands of tiny chunks, merge logic
isn't firing — debug before re-embedding.

#### 1.5b — BM25 lexical index

**Schema migration** (apply to `init.sql` and re-run `01_init_db.py` on a
fresh DB, or run as ALTER on existing):
```sql
ALTER TABLE tree_nodes
  ADD COLUMN tsv tsvector
  GENERATED ALWAYS AS (to_tsvector('english', summary)) STORED;

CREATE INDEX tree_nodes_tsv_idx
  ON tree_nodes USING GIN(tsv)
  WHERE level = 0;

-- For exact-term phrase queries on identifiers (visa codes, tickers)
CREATE INDEX tree_nodes_summary_trgm_idx
  ON tree_nodes USING GIN(summary gin_trgm_ops)
  WHERE level = 0;
-- Requires: CREATE EXTENSION IF NOT EXISTS pg_trgm;
```

**Why leaf-only:** cluster summaries (level ≥ 1) are LLM-generated text
that's *about* the underlying chunks but doesn't necessarily contain the
user's exact query terms. BM25 over summary text adds false-positive
matches without recall benefit. Lexical signal belongs at the document
level (leaves), semantic signal works everywhere.

**Why generated column over manual updates:** the tsvector stays in sync
with `summary` automatically. Phase 5.5's incremental insertion doesn't
need to worry about updating the tsvector — Postgres does it.

**Why also `pg_trgm`:** `to_tsvector` lemmatizes and stems. For exact
identifier matches like "F-2-7-7" or "AAPL", trigram matching is more
reliable than the lemmatized tsvector.

#### 1.5c — Retrieval fusion

**New files:**
- `agents/retrieval/bm25_search.py` — single function:
  ```python
  def bm25_search_within_ids(
      query: str,
      candidate_ids: list[int],
      k: int,
  ) -> list[tuple[int, float]]:
      """
      Returns (node_id, ts_rank_cd score) sorted desc.
      Uses websearch_to_tsquery for user-friendly query syntax.
      """
  ```
- `agents/retrieval/fusion.py` — Reciprocal Rank Fusion:
  ```python
  def reciprocal_rank_fusion(
      ranked_lists: list[list[int]],   # each list is node_ids in rank order
      k: int,                           # final top-k to return
      k_rrf: int = 60,                  # RRF dampening constant
  ) -> list[tuple[int, float]]:
      """
      score(doc) = sum over ranked_lists of 1 / (k_rrf + rank_in_list(doc))
      Returns top-k by RRF score.
      """
  ```

**Tree-search integration** — `agents/retrieval/tree_search.py` extended:

```python
def hybrid_search(
    query: str,
    query_embedding: np.ndarray,
    *,
    mode: Literal["top_down", "collapsed"],
    fusion: Literal["dense", "bm25", "hybrid"],
    k: int,
    rrf_k: int = 60,
    alpha: float = 0.3,                  # collapsed mode only
) -> SearchResult:
    # 1. Tree walk produces candidate leaf set
    if mode == "top_down":
        candidate_leaves = top_down_beam_traversal(query_embedding, beam=6)
    else:  # collapsed
        candidate_leaves = collapsed_with_ancestor_boost(query_embedding, alpha)

    # If pure dense or pure bm25, no fusion needed
    if fusion == "dense":
        return rank_dense(query_embedding, candidate_leaves, k)
    if fusion == "bm25":
        return rank_bm25(query, candidate_leaves, k)

    # Hybrid: get top k*4 from each ranker, fuse via RRF
    dense_ranked = rank_dense(query_embedding, candidate_leaves, k * 4)
    bm25_ranked  = rank_bm25(query, candidate_leaves, k * 4)
    return reciprocal_rank_fusion(
        [dense_ranked.ids, bm25_ranked.ids],
        k=k, k_rrf=rrf_k,
    )
```

**Why RRF over weighted-sum** (`α · dense + (1-α) · bm25`):
- Dense cosine returns values in `[-1, 1]`; `ts_rank_cd` returns
  unbounded positive floats. Comparing them requires per-query
  min-max normalization, which is brittle (single outlier scores skew
  the whole list).
- RRF only uses rank position; scale-free, parameter-light. The single
  `k_rrf` parameter (default 60, from the original RRF paper) is robust
  across very different rankers — this is why Elasticsearch, Vespa, and
  ColBERT-X use it as their default hybrid strategy.

**API extension** (`agents/retrieval/main.py`):

```python
class RetrieveRequest(BaseModel):
    query: str
    mode: Literal["top_down", "collapsed"] = "top_down"
    fusion: Literal["dense", "bm25", "hybrid"] = "dense"  # backwards-compatible default
    rrf_k: int = 60
    alpha: float = 0.3       # used only when mode == "collapsed"
    k: int = 5
    domain: Optional[str] = None
```

**Default stays `fusion: "dense"`** so any existing client code continues
to work. Switch to `"hybrid"` after Phase 7 evaluation confirms it wins.

**Verify:**
```bash
# Exact-term recall — dense should miss, bm25 should hit
python scripts/05_query_cli.py "F-2-7-7" --domain law --fusion dense  --k 5
python scripts/05_query_cli.py "F-2-7-7" --domain law --fusion bm25   --k 5
python scripts/05_query_cli.py "F-2-7-7" --domain law --fusion hybrid --k 5
# Expect: bm25 and hybrid surface the correct visa-code chunk in top-5;
# dense alone may rank it outside top-5.

# Semantic-paraphrase recall — bm25 should miss, dense should hit
python scripts/05_query_cli.py "what paperwork is needed when extending residency status" \
    --domain law --fusion dense  --k 5
python scripts/05_query_cli.py "what paperwork is needed when extending residency status" \
    --domain law --fusion bm25   --k 5
# Expect: dense finds the F-2 renewal docs even without keyword overlap;
# bm25 may struggle with paraphrase.

# Hybrid should perform >= max(dense, bm25) on both query types.
```

**Don't move on until:** the three test queries above demonstrate the
expected complementary behavior. Hybrid Recall@5 ≥ max(dense, bm25)
Recall@5 on the Phase 7 synthetic eval is the formal acceptance bar, but
the manual smoke test above is the daily-development sanity check.

---

### Phase 2 — Embed chunks as leaf nodes ✓ DONE

**New files:**
- `agents/ingestion/encoder.py` — `class BGEM3Encoder` wrapping
  `sentence-transformers`. Batched, GPU, L2-normalized vectors.
- `agents/ingestion/embed_leaves.py` — reads chunks not yet embedded, encodes
  in batches, inserts into `tree_nodes` with `level=0`, `is_leaf=true`.
- `scripts/03_embed_chunks.py` — orchestrator that also builds the HNSW index
  *after* all leaves are loaded (one-shot index build is far faster than
  incremental).

**DB state after:** every row in `chunks` has a corresponding row in
`tree_nodes` with `is_leaf=true`, `level=0`, `summary = chunks.text`,
`embedding` populated. HNSW index `tree_nodes_embedding_hnsw_idx` exists.
After Phase 1.5, the tsvector column also exists on every row (auto-
populated by the generated-column expression).

**Verify:**
```sql
-- Every chunk has exactly one leaf node
SELECT
  (SELECT COUNT(*) FROM chunks) AS n_chunks,
  (SELECT COUNT(*) FROM tree_nodes WHERE level=0) AS n_leaves,
  (SELECT COUNT(*) FROM chunks)
    - (SELECT COUNT(*) FROM tree_nodes WHERE level=0) AS missing;

-- HNSW index exists
SELECT indexname FROM pg_indexes
WHERE tablename = 'tree_nodes' AND indexname LIKE '%hnsw%';

-- After Phase 1.5: GIN index exists and tsvector populated
SELECT indexname FROM pg_indexes
WHERE tablename = 'tree_nodes' AND indexname LIKE '%tsv%';
SELECT COUNT(*) FROM tree_nodes WHERE level=0 AND tsv IS NOT NULL;
```

**Don't move on until:** every chunk has a leaf node, no nulls in embedding,
HNSW index built. After Phase 1.5: GIN index exists, every leaf has a
non-null tsvector.

---

### Phase 3 — Build the tree ✓ DONE

**New files:**
- `agents/tree_builder/cluster.py` — UMAP dim-reduce (1024 → 10), HDBSCAN
  cluster, noise-point reassignment to nearest cluster. Pure numpy, no DB.
- `agents/tree_builder/summarizer.py` — in-process Qwen2.5-7B summarizer
  (greedy, JSON-mode). Takes a list of child summaries, returns
  `{title, summary}` JSON. Runs via `transformers` in the same process as
  the build script — one fewer service to manage, throughput-optimized for
  a batch job.
- `agents/tree_builder/build.py` — main loop: for each `(domain, source)`,
  iterate L=0→max_levels: fetch level-L nodes → cluster → summarize each
  cluster → embed summary → insert as level-L+1 node → set children's
  `parent_id`. Stop when one cluster remains or max depth (4) hit.
- `scripts/04_build_tree.py` — runs per `(domain, source)`, prints tree shape
  and `n_descendants` vs leaf-count reconciliation.

**DB state after:** `tree_nodes` has rows at levels 0..k (k ≤ 4). Parent
pointers form a forest (one tree per `(domain, source)`). All leaves have a
non-null `parent_id` (noise reassignment guarantees this).

**Verify:**
```sql
-- Tree shape per domain/source
SELECT domain, source, level, COUNT(*) AS n_nodes,
       AVG(n_descendants)::int AS avg_leaves_under
FROM tree_nodes
GROUP BY 1,2,3 ORDER BY 1,2,3;

-- Zero leaf orphans (noise reassignment guarantee)
SELECT COUNT(*) FROM tree_nodes WHERE level=0 AND parent_id IS NULL;
-- MUST be 0

-- Read 5 internal node summaries — do the titles name coherent topics?
SELECT domain, source, level, title, LEFT(summary, 300) AS preview
FROM tree_nodes WHERE NOT is_leaf
ORDER BY RANDOM() LIMIT 5;
```

**Don't move on until:** internal node titles read like topic names a human
would write. If they're vague ("various topics"), the summarizer prompt or
cluster min-size is wrong. Fix before Phase 4.

---

### Phase 4 — Tree-guided retrieval ✓ DONE (hybrid extension in Phase 1.5)

**New files:**
- `agents/retrieval/tree_search.py` — implements two tree-walk strategies:
  1. **Top-down beam traversal** (beam=6): ANN at level k → keep top-`beam`
     children → fetch their children via `WHERE parent_id = ANY(frontier)` →
     repeat to leaves. Final step uses one recursive descent so beam doesn't
     cap leaf recall.
  2. **Collapsed + ancestor boost**: flat HNSW pass over ALL nodes (any
     level), then re-rank with `combined = leaf_sim + α · max(ancestor_sim)`.
     Cluster summary acts as a learned topic prior.
- `agents/retrieval/main.py` — FastAPI service, `POST /retrieve` returns
  top-k leaf chunks + the traversal path used + the scoring mode used.
  `GET /healthz` (liveness), `GET /readyz` (encoder loaded + DB reachable +
  leaves present).
- `agents/retrieval/Dockerfile` — uv-based slim image, CPU default,
  CUDA via build-arg.
- `scripts/05_query_cli.py` — CLI that calls `tree_search` directly (skipping
  FastAPI) so retrieval logic can be debugged without service overhead.

Phase 1.5 extends this with `bm25_search.py`, `fusion.py`, and a `fusion`
parameter on the API. The tree-walk strategies are unchanged; fusion
operates on the candidate leaf set the tree walk produces.

**Performance optimization:** `hnsw.ef_search` is raised **per transaction**
(`SET LOCAL hnsw.ef_search = ...`), not per session, so the bump stays
scoped and never leaks across pooled connections.

**Verify:**
```bash
python scripts/05_query_cli.py "What did the latest paper on RLHF show?"
# Expected: traversal path through ai → ml_arxiv_papers →
# some RLHF/alignment cluster → top-5 chunks. Path printed; chunks printed.

python scripts/05_query_cli.py "What does the F-2 visa allow?" \
  --domain law --mode collapsed --fusion hybrid --k 5
```

**Open question for Phase 7:** which combination of (mode, fusion) wins on
real data? Six combinations to test: `{top_down, collapsed} × {dense, bm25,
hybrid}`. We deliberately ship all six — decide on measured data, not
preference.

---

### Phase 5 — Domain-specialist quantized workers ✓ DONE

**New files** (`training/`):
- `qwen_trader_SFT_fine_tune.py` — TRL `SFTTrainer` on
  Sujet-Finance-Instruct-177k + finance-alpaca. QLoRA r=32, nf4
  double-quant, bf16, cosine LR 2e-4, effective batch 16, packed chat
  template. Saves LoRA adapters then merges to FP16.
- `qwen_coder_GRPO_fine_tune.py` — TRL `GRPOTrainer` on
  verifiable-coding-problems + LeetCodeDataset, with four programmatic
  reward heads:
  1. **format reward** (0.5): output matches
     `<reasoning>...</reasoning><code>```python ... ```</code>`.
  2. **syntax reward** (0.25): extracted code passes Python `compile()`.
  3. **correctness reward** (2.0): code runs in a sandboxed `subprocess`
     with 8-second timeout against the problem's unit tests.
  4. **length reward** (0.1): completion tokens in [100, 800).
  Structured `<reasoning>/<code>` contract forces reasoning before code.
- `awq_quantize_coder_worker.py` — two-stage:
  1. **Merge stage (CPU):** load base + adapter on CPU,
     `resize_token_embeddings` if vocab grew, `merge_and_unload`, save FP16
     with `max_shard_size=4GB`.
  2. **Quantize stage (GPU):** `AutoAWQForCausalLM` with W4A16 calibration
     on 128 task-matched samples drawn from
     `verifiable-coding-problems-python`. System prompt during calibration
     **matches GRPO training prompt** to keep calibration distribution
     aligned with serving distribution.
- `awq_quantize_law_worker.py` — single-stage GPU quantize (no SFT step
  needed). Calibration uses CUAD-QA + LegalQAEval. **30% of calibration
  samples are constructed as "answer-not-in-context" cases** with the
  refusal string `"The answer is not contained in the provided context."`
  This is the key insight: AWQ calibration shapes which weight precisions
  preserve which behaviors. If refusal cases are absent from calibration,
  the 4-bit model will hallucinate confident answers where the FP16 model
  would refuse.

**Artifacts produced:**
- `checkpoints/awq_models/qwen_coder_awq_w4a16/`
- `checkpoints/awq_models/law_llm_awq_w4a16/`
- (trader merged FP16 ready for AWQ — quantize script analogous to coder
  not yet shipped as a dedicated worker file; same `autoawq` recipe applies)

**Verify:**
```bash
# Sanity-check each worker loads and produces reasonable output
python -c "
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
m = AutoAWQForCausalLM.from_quantized('checkpoints/awq_models/qwen_coder_awq_w4a16')
t = AutoTokenizer.from_pretrained('checkpoints/awq_models/qwen_coder_awq_w4a16')
print(m.generate(**t('def fibonacci(n):', return_tensors='pt').to(m.model.device),
                 max_new_tokens=64))
"

# Specifically for law: verify refusal behavior survives quantization
# Prompt: a question whose answer is NOT in the supplied context.
# Expected: model emits the refusal string (or a paraphrase), NOT a
# fabricated answer.
```

**Don't move on until:** law worker refuses on out-of-context prompts at a
rate comparable to its FP16 source. If 4-bit refusal rate drops sharply,
increase REFUSAL_FRAC in the calibration script.

---

### Phase 5.5 — Self-updating ingestion ✓ DONE

**New files** (`agents/ingestion/`):
- `pdf_loader.py` — PDF → text extraction (pypdf for text PDFs;
  unstructured-io fallback for scanned/table-heavy PDFs). After Phase 1.5,
  the extracted text passes through the structure-aware chunker, not the
  fixed-token chunker.
- `domain_classifier.py` — given a document embedding, returns the
  best-matching domain by cosine-argmax against cached top-level cluster
  summary embeddings. Cache is rebuilt whenever Phase 3 reruns.
- `watcher.py` — watchdog-based daemon on a configured directory. On
  `created` events:
  1. Compute SHA-256 hash; if hash exists in `documents.content_hash`, skip.
  2. PDF → text → structure-aware chunks.
  3. Embed each chunk; embed the full doc for domain classification.
  4. Insert document + chunks; insert chunks as level-0 `tree_nodes`.
     **tsvector is auto-populated** by the generated-column expression
     introduced in Phase 1.5 — no extra work in the watcher.
  5. For each new leaf, find nearest level-1 cluster *within the
     classified domain* and set `parent_id`.
  6. If any updated cluster now exceeds `CLUSTER_RESUMMARIZE_THRESHOLD`
     descendants, enqueue it for re-summarization.
- `agents/tree_builder/incremental.py` — re-summarizer worker that pops
  enqueued clusters, regenerates title + summary via Qwen2.5-7B, re-embeds,
  and updates the node in place.

**DB schema additions:** `documents.content_hash` (unique index),
`incremental_resummarize_queue` table. Phase 1.5 adds `tree_nodes.tsv`
generated column + GIN index — backwards compatible with Phase 5.5
since the column generates itself on insert.

**Verify:**
```bash
# Drop a test PDF, watch it appear
cp test_visa_law.pdf data/law/user_pdfs/
# tail the watcher log; expect: classified=law, chunks=N, parent_cluster=...

# Confirm it's queryable immediately, including via BM25 (after Phase 1.5)
python scripts/05_query_cli.py "what's in the test_visa_law document" --fusion hybrid
```

**Don't move on until:** dropping a PDF results in retrievable chunks
within ~5 seconds for a 20-page document, and the same PDF dropped twice
doesn't double-ingest. After Phase 1.5: the new chunks are also queryable
via `fusion=bm25`.

---

### Phase 6 — Generation + Gateway → IN PROGRESS

**New files:**
- `agents/generation/{prompts,worker_registry,awq_service,main,Dockerfile}.py`
  — autoawq inference service. `worker_registry` maps
  `{ai: <generic Qwen2.5-7B>, trading: <trader AWQ>, law: <law AWQ>,
  code: <coder AWQ>}` to checkpoint paths.
- `gateway/{main,graph,nodes,domain_router,models,Dockerfile}.py` —
  LangGraph StateGraph orchestrator:
  1. **Route node** — embed query, cosine-argmax vs. cached tree-root
     embeddings → domain.
  2. **Retrieve node** — HTTP call to retrieval service with
     `(query, domain, mode, fusion, k)`. **The gateway picks the
     `fusion` mode** based on either client request or a heuristic
     (e.g., short ALL-CAPS query → bm25-favored; long natural-language
     question → hybrid default).
  3. **Generate node** — HTTP call to generation service with
     `(query, retrieved_chunks, domain)` → routes to the matching AWQ
     worker.
  4. **Return** — `{answer, sources, tree_path, fusion_used, worker_used, latency_ms}`.
- `docker-compose.yml` rewrite: services `postgres`, `retrieval-agent`,
  `generation-agent`, `gateway`.

**Verify:**
```bash
docker compose up --build
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What are the requirements for F-2 visa renewal?"}'
# Expected: {answer, sources, latency_ms, tree_path, fusion_used: "hybrid",
#            worker_used: "law"}
```

---

### Phase 7 — Evaluation ⏳

**Eval set construction** (no qrels exist for this corpus):
- `evaluation/synthetic_qa_gen.py` — for each domain, sample 100 chunks →
  prompt Qwen2.5-7B to generate `(question, answer, source_chunk_id)`
  triples. Manually review ~20% per domain for quality. Discard junk.
  Within each domain, mix two query types:
  1. **Semantic queries** — natural-language paraphrases of the chunk
     content. Stress test dense retrieval.
  2. **Exact-term queries** — questions citing specific identifiers,
     codes, proper nouns from the chunk. Stress test BM25.
  This stratification is what makes the fusion-mode ablation meaningful.

**Metrics:**
- `eval_tree_quality.py` — silhouette score per cluster level, % singleton
  clusters, average depth, % leaves with valid parent path.
- `eval_retrieval.py` — Recall@{1,5,10}, MRR@10, nDCG@10 on the synthetic
  gold set. Run all six `{mode} × {fusion}` combinations; report a
  results matrix. The expected pattern (to be confirmed):
  - dense wins on semantic queries.
  - bm25 wins on exact-term queries.
  - hybrid wins on average and never substantially loses to either.
  If hybrid loses on a slice, that's a real finding — investigate which
  RRF parameter or candidate-set size is wrong.
- `eval_generation.py` — F1 vs gold answer, faithfulness (does the
  answer cite real retrieved chunks?), citation precision.
- `eval_worker_vs_base.py` — the critical ablation: for each domain, run
  the same retrieved chunks through (a) the domain AWQ worker and (b) the
  generic Qwen2.5-7B-Instruct base. Report Δ on F1, faithfulness, and (for
  law specifically) refusal precision/recall on out-of-context questions.

The worker-vs-base ablation and the fusion-mode matrix are the two
experiments that justify the v2 architecture decisions empirically. If
either comes back null, the corresponding design choice should be
reconsidered.

---

## Sequencing rule

Validate each phase against its verification queries BEFORE starting the
next. If you skip ahead, you'll spend more time debugging cascading failures
than you would have spent verifying. This is the most common mistake on
multi-phase pipelines, and the cost compounds: a missed Phase 3 verification
(vague cluster titles) wastes weeks of Phase 4 retrieval tuning chasing what
is actually a clustering problem.

Phase 1.5 deserves special caution: changing the chunker invalidates every
downstream artifact. Re-running 02 → 03 → 04 to rebuild the corpus on the
new chunks is mandatory before Phase 1.5's retrieval changes can be
fairly evaluated. **Do not** A/B-test old-chunker dense vs. new-chunker
hybrid; the chunking change confounds the fusion change.

---

## Where you are now

- Phases 1, 2, 3, 4, 5, 5.5 complete and validated.
- Phase 1.5 in progress (structure-aware chunking + BM25 + RRF hybrid).
  Ships as one phase because the chunker change and the retrieval change
  must be evaluated together on a freshly-rebuilt corpus.
- Phase 6 next: generation FastAPI wrapper around the shipped AWQ
  workers, plus the LangGraph gateway that routes queries through
  `route → retrieve (with fusion) → generate`.
- Phase 7 (synthetic eval with fusion-mode matrix + worker-vs-base
  ablation) is the next research milestone after Phase 6 ships.

---

## Known caveats

- **Trader AWQ script not in repo yet.** `awq_quantize_coder_worker.py` and
  `awq_quantize_law_worker.py` exist; the analogous trader script is a
  straight copy of the coder version with paths and calibration data
  swapped, and is on the immediate todo list.
- **Generation service and gateway are scaffolded but empty.** The
  `agents/generation/` and `gateway/` directories exist with placeholder
  files; full implementation is Phase 6.
- **Self-updating ingestion assumes domain has at least one existing
  cluster.** A brand-new domain with no tree roots has nothing to
  cosine-argmax against. Bootstrap a domain with a batch ingest before
  enabling the watcher for it.
- **HDBSCAN is not incremental.** Phase 5.5's "nearest-cluster reassign"
  is a nearest-centroid heuristic against frozen Phase 3 clusters. Cluster
  drift accumulates; periodic full Phase 3 rebuilds (currently manual) are
  needed to keep cluster summaries representative.
- **Phase 1.5 chunker change invalidates the current tree.** Once the
  new chunker ships, the existing corpus must be fully re-ingested (drop
  `chunks` and `tree_nodes`, run 02 → 03 → 04). Hybrid retrieval evaluation
  on old chunks would conflate two effects.
- **`ts_rank_cd` is BM25-style, not true BM25.** If retrieval evaluation
  shows BM25 underperforming literature baselines on the synthetic eval,
  swap to `pg_search` (formerly ParadeDB) which provides true BM25 in the
  same Postgres deployment.
- **Korean text segmentation.** NLTK English tokenization works for the AI
  and Trading corpora but may not split Korean Immigration Law PDFs
  correctly. The Phase 1.5 chunker dispatches to KSS (Korean Sentence
  Splitter) for clause text identified as Korean by langdetect; this is a
  fragile path and may need rework once real Korean PDFs are tested.