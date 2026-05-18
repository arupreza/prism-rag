-- ============================================================================
-- PRISM-RAG schema  (tree-guided dense RAG)
--
-- Run once after creating the database:
--   createdb prism_rag
--   psql "$PG_DSN" -f init.sql
--
-- Idempotent — safe to re-run.
--
-- NOTE: This replaces the SPLADE/sparsevec schema that was originally planned.
-- We now use dense embeddings (BGE-M3, 1024-d) with HNSW. No sparsevec column.
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;   -- gen_random_uuid()

-- ----------------------------------------------------------------------------
-- documents : one row per JSONL record (original article / paper / speech)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    doc_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    external_id  TEXT,                                  -- "id" field from JSONL
    domain       TEXT NOT NULL,                         -- politics / finance / ai_tech / medical
    source       TEXT NOT NULL,                         -- cc_news, pubmed_papers, ...
    title        TEXT,
    text         TEXT NOT NULL,
    metadata     JSONB,
    n_tokens     INT,
    created_at   TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (source, external_id)                        -- idempotent ingest key
);
CREATE INDEX IF NOT EXISTS idx_documents_domain_source
    ON documents (domain, source);

-- ----------------------------------------------------------------------------
-- chunks : encoder-sized pieces of each document. Become tree leaves.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id     UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    chunk_idx  INT NOT NULL,                            -- 0..N-1 within doc
    text       TEXT NOT NULL,
    n_tokens   INT,
    UNIQUE (doc_id, chunk_idx)
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks (doc_id);

-- ----------------------------------------------------------------------------
-- tree_nodes : ONE table for leaves AND internal cluster nodes.
--   - level 0  = leaf (is_leaf=true, chunk_id set, summary = chunk text)
--   - level 1+ = internal (is_leaf=false, chunk_id NULL, summary = LLM output)
-- A single HNSW index serves every level. Retrieval filters by level/parent_id.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS tree_nodes (
    node_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    domain         TEXT NOT NULL,
    source         TEXT,                                -- NULL only at root
    level          INT NOT NULL,                        -- 0 = leaf
    is_leaf        BOOLEAN NOT NULL,
    parent_id      UUID REFERENCES tree_nodes(node_id) ON DELETE CASCADE,
    chunk_id       UUID REFERENCES chunks(chunk_id),    -- non-null iff is_leaf
    title          TEXT,                                -- LLM-generated for clusters
    summary        TEXT,                                -- LLM-gen for clusters; chunk text for leaves
    n_descendants  INT,                                 -- # of level-0 leaves under this node
    cluster_meta   JSONB,                               -- {method, size, silhouette, ...}
    embedding      vector(1024) NOT NULL,               -- BGE-M3 = 1024-d
    embed_input    TEXT,                                -- exact text fed to encoder (reproducibility)
    created_at     TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_tree_nodes_dom_src_lvl
    ON tree_nodes (domain, source, level);
CREATE INDEX IF NOT EXISTS idx_tree_nodes_parent
    ON tree_nodes (parent_id);
CREATE INDEX IF NOT EXISTS idx_tree_nodes_chunk
    ON tree_nodes (chunk_id);
CREATE INDEX IF NOT EXISTS idx_tree_nodes_hnsw
    ON tree_nodes USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);