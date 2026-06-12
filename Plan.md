# PRISM-RAG — Canonical Build Plan (Opus Handoff)

**Repo:** https://github.com/Arupreza/PRISM-RAG @ `3a397be` (2026-06-11)
**Rule for the implementing agent:** This document overrides README.md and ARCHITECTURE.md wherever they conflict. The code on disk is ground truth for what exists; this plan is ground truth for what to build.

---

## 0. Ground-truth audit (code vs. docs)

| Module | Docs claim | Actual state @ 3a397be |
|---|---|---|
| `agents/ingestion/` (loader, chunker, captioner, db, encoder, embed_leaves, config) | Ph 1 ✅ | **EXISTS, working.** Parent-child chunker + Qwen2.5-VL image captioner. ~918 LOC |
| `agents/tree_builder/` (build, summarizer) | Ph 3 ✅, UMAP+HDBSCAN | **EXISTS** — but uses UMAP + **BIC-selected soft GMM** (original RAPTOR), multi-membership via `chunks.parent_ids[]`. Docs wrong about HDBSCAN |
| `agents/retrieval/main.py` | Ph 4 ✅ | **BROKEN.** Imports `agents.retrieval.tree_search` which does not exist. Only `parent_child.py` (child-search → parent-return) works |
| `agents/retrieval/{tree_search,bm25_search,fusion}.py` | Ph 4 ✅ / Ph 1.5 ⏳ | **MISSING** |
| `agents/generation/` | Ph 6 scaffold | **0-line stubs** (`__init__`, `llm_service`, `main`, `prompts`) |
| `agents/cache/` | not in docs at all | **0-line stubs** — decide: implement or delete |
| `gateway/` | Ph 6 scaffold | **0-line stubs** (graph, nodes, main, models) |
| Ph 5.5 (`pdf_loader`, `domain_classifier`, `watcher`, `incremental`) | ✅ Done | **DO NOT EXIST.** PDF loading lives inside `loader.py`; no watcher, no incremental tree update |
| `training/` | Ph 5 ✅ (3 workers) | EXISTS — SFT (trader), GRPO (coder), AWQ ×3 **plus undocumented `awq_quantize_vision_worker.py`** |
| `init.sql` | `tree_nodes` + tsvector GIN | Actual schema: single `chunks` table, parent/child/RAPTOR levels via `level` + `is_searchable` + `parent_chunk_id` + `parent_ids[]`; per-domain partial HNSW (`vector_cosine_ops`); `pg_trgm` GIN on content; **no tsvector column** |
| `scripts/` | 6 scripts | 4: `01_init_db`, `02_ingest_and_build`, `03_query_cli`, `04_benchmark` (empty) |
| `evaluation/` | Ph 7 plan | 3 files exist (`eval_retrieval`, `eval_generation`, `eval_cache`) — verify contents before trusting |

**True status: Phases 1 (incl. multimodal), 2, 3, 5 exist. Phase 4 is broken (missing module). Phases 1.5, 5.5, 6, 7 not built.**

---

## 1. Canonical architecture decisions (resolve all doc/code conflicts)

These are final. Do not re-litigate inside implementation.

1. **Schema:** keep the code's single `chunks` table. Semantics (already in `init.sql`):
   - `level=0, parent_chunk_id NULL, is_searchable=FALSE` → PARENT paragraph (≤1200 tok)
   - `level=0, parent_chunk_id NOT NULL, is_searchable=TRUE` → CHILD shard (350 tok, 60 overlap)
   - `level≥1, is_searchable=TRUE` → RAPTOR summary node
   - `content_type='image'` → VLM caption row, `image_path` → figure on disk
2. **Clustering:** UMAP → BIC-selected soft GMM with probability-threshold multi-membership (`parent_ids[]`). This is the original RAPTOR recipe and is what `build.py` implements. Update docs; delete every HDBSCAN mention.
3. **What gets clustered:** **PARENT paragraphs** are the level-0 units fed to RAPTOR clustering (children are embedding shards of parents, clustering them would duplicate membership). Summaries at level≥1 embed their own text. *(Verify `build.py` selects parents; if it clusters children, fix to parents.)*
4. **Retrieval unit contract:** dense ANN and lexical search run over **searchable rows** (children + summary nodes). Results **dedup to parents**; the parent's full paragraph is the context unit sent to generation. `parent_child.py` already implements the child→parent half; tree-walk and fusion must respect the same contract.
5. **Lexical search:** PostgreSQL FTS. Add a generated `tsv tsvector` column on searchable rows + GIN partial index (`WHERE is_searchable`), keep existing `pg_trgm` index for exact identifiers (visa codes, tickers). **Never call `ts_rank_cd` "BM25"** in code, docs, or the paper — it lacks TF saturation and length normalization. Label: `lexical`. If Phase 7 shows it underperforming, swap to ParadeDB `pg_search` (true BM25) — same Postgres.
6. **Fusion:** Reciprocal Rank Fusion, `k_rrf=60`, over rank lists from (dense, lexical) computed **within the tree-walk candidate set** (top `k·4` each). Modes exposed: `dense | lexical | hybrid`. Default `dense` until Phase 7 says otherwise.
7. **Multimodal is a first-class feature** (it's already built): PDF figures → Qwen2.5-VL captions → searchable rows carrying `image_path`. Document it; the vision AWQ worker is part of Phase 5.
8. **`agents/cache/`:** implement as an optional semantic cache in Phase 6b (embed query → cosine ≥ τ against cached (query, answer) → return cached answer) **or delete the directory.** Empty stubs may not ship.

---

## 2. Phase plan

Sequencing rule (keep from ARCHITECTURE.md): each phase's verification gate passes before the next starts. Chunker/schema changes invalidate downstream artifacts → full re-ingest (drop chunks beyond level 0 parents if chunking unchanged; full drop if changed).

### Phase R0 — Repair + reconcile (FIRST, blocking)
- Fix or restore `agents/retrieval/tree_search.py` so `retrieval/main.py` starts. Implement `TreeSearcher` with:
  - `top_down(query_emb, beam=6, fanout)` — ANN at top level → keep top-beam → descend via `parent_ids`/`children_ids` → searchable leaves.
  - `collapsed(query_emb, alpha=0.3)` — flat ANN over all searchable rows; re-rank `child_sim + α·max(ancestor_sim)`.
  - Both end with **dedup-to-parent** per §1.4. `SET LOCAL hnsw.ef_search` per transaction.
- Rewrite README.md + ARCHITECTURE.md to match §1 and the real status table. Delete claims of completed phases that aren't.
- Decide cache: implement later or `git rm agents/cache`.
- **Gate:** `uvicorn agents.retrieval.main:app` starts; `POST /retrieve` returns parents with tree path on all three domains; docs contain zero references to HDBSCAN, `tree_nodes`, or "BM25" for ts_rank_cd.

### Phase R1 — Lexical index + hybrid fusion (was "1.5")
- `init.sql`: add `tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED`; partial GIN `WHERE is_searchable`. Note: English config — Korean law text relies on the existing `pg_trgm` path (FTS 'english' will not stem Korean; acceptable, document it).
- `agents/retrieval/lexical_search.py`: `lexical_search_within_ids(query, candidate_ids, k)` via `websearch_to_tsquery` + `ts_rank_cd`, union with a trgm `similarity()` pass for identifier-like tokens (regex: contains digit or all-caps len≤8).
- `agents/retrieval/fusion.py`: RRF as specified in §1.6.
- Extend `RetrieveRequest` with `fusion: Literal["dense","lexical","hybrid"]="dense"`, `rrf_k:int=60`. Backwards compatible.
- **Gate (smoke):** `"F-2-7-7"` → lexical/hybrid hit top-5, dense may miss; paraphrase query ("paperwork to extend residency") → dense/hybrid hit, lexical may miss. Formal acceptance deferred to Phase R4 metrics.

### Phase R2 — Generation service
- `agents/generation/llm_service.py`: AWQ inference via `autoawq` (or vLLM if VRAM plan allows — decide once, vLLM strongly preferred for throughput; autoawq `generate` is fine for single-GPU dev).
- `worker_registry`: `{ai: Qwen2.5-7B-Instruct base, trading: trader AWQ, law: law AWQ, code: coder AWQ, vision: vision AWQ}` → checkpoint paths. Trader AWQ checkpoint must exist — `awq_quantize_trader_worker.py` is in repo; run it and check the artifact in.
- `prompts.py`: per-domain answer templates; law template hard-requires the refusal string when context lacks the answer; all templates require inline source citations `[chunk_id]`.
- FastAPI `POST /generate {query, chunks[], domain}` → `{answer, citations[], worker_used, latency_ms}`. `GET /healthz`, `/readyz`.
- **Gate:** each worker loads, answers from supplied chunks, cites real chunk_ids; law worker refuses on an out-of-context probe.

### Phase R3 — Gateway (LangGraph)
- `gateway/graph.py`: StateGraph `route → retrieve → generate → respond`.
  - **route:** embed query; cosine-argmax vs cached top-level summary-node embeddings per domain (cache rebuilt on tree rebuild). Below margin τ → search all domains, skip worker specialization (use base).
  - **retrieve:** HTTP to retrieval service. Fusion heuristic: query contains identifier-like token → `hybrid` with lexical weight; else client-specified or `hybrid` default post-R4.
  - **generate:** HTTP to generation service.
- `docker-compose.yml`: `postgres`, `retrieval`, `generation`, `gateway` services; healthcheck-gated startup order.
- Optional R3b: semantic cache node before retrieve (cosine ≥ 0.95 on cached queries) — only if `agents/cache` kept.
- **Gate:** `docker compose up --build`; one `POST /query` returns `{answer, sources, tree_path, fusion_used, worker_used, latency_ms}` end-to-end for one query per domain.

### Phase R4 — Evaluation
- `synthetic_qa_gen.py`: per domain, 100 sampled parents → Qwen2.5-7B generates `(question, answer, source_parent_id)`; stratify 50% semantic-paraphrase / 50% exact-term queries; manually review ≥20%/domain.
- `eval_retrieval.py`: Recall@{1,5,10}, MRR@10, nDCG@10 over the 6-cell matrix `{top_down, collapsed} × {dense, lexical, hybrid}`. Acceptance: hybrid ≥ max(dense, lexical) on the pooled set and within −2 pts on every slice.
- `eval_generation.py`: token-F1 vs gold, faithfulness (answer claims grounded in retrieved parents), citation precision.
- `eval_worker_vs_base.py`: same retrieved context → (a) domain AWQ worker, (b) generic Qwen2.5-7B. Report Δ F1/faithfulness; **for law: refusal precision/recall on out-of-context questions, FP16 vs W4A16** — this is the experiment that turns the "30% refusal calibration preserves refusal" claim from hypothesis into result. Until measured, docs must phrase it as hypothesis.
- `eval_tree_quality.py`: silhouette per level, % singleton clusters, depth, orphan count (must be 0).
- **Gate:** results matrix + worker-vs-base table committed to `evaluation/results/`; README updated with numbers; if hybrid or workers lose, the corresponding default is reverted — decisions on data, not preference.

### Phase R5 — Self-updating ingestion (was "5.5"; build it for real)
- `watcher.py` (watchdog daemon): SHA-256 dedup via `documents.content_hash` (add column + unique index) → loader (PDF text + figures → captioner) → chunker → embed → insert; tsv auto-generates.
- `domain_classifier.py`: cosine-argmax vs cached domain-root embeddings; bootstrap rule: a domain must have a tree before watcher activates for it.
- `incremental.py`: nearest level-1 cluster assignment within classified domain (append to `parent_ids`); clusters past `CLUSTER_RESUMMARIZE_THRESHOLD` enqueued for re-summarization. Known limitation stays documented: GMM is not incremental, periodic full rebuild required.
- **Gate:** dropped 20-page PDF retrievable (dense + lexical) in ≤ ~10 s; double-drop is a no-op; figures captioned and retrievable.

---

## 3. Standing constraints for the implementing agent

1. Code on disk wins over README/ARCHITECTURE prose; this plan wins over both.
2. Never mark a phase done without its gate output (SQL result / curl response / metric table) pasted into the PR description.
3. No new tables; extend `chunks`/`documents` only.
4. All scores/claims in docs must be measured (Phase R4) or labeled hypothesis.
5. Single GPU (≥24 GB) dev assumption; AWQ workers load one-at-a-time in dev, registry lazy-loads.
6. W&B logging for all training/eval runs; commit run URLs.
7. Keep `fusion="dense"` the API default until R4 acceptance flips it.

---

## 4. Immediate task order

1. R0: restore `tree_search.py`, fix broken import, rewrite both docs, cache decision.
2. R1: tsv column + lexical + RRF.
3. Run `awq_quantize_trader_worker.py`; verify all 4 AWQ artifacts (trader, coder, law, vision).
4. R2 → R3 → R4 → R5.